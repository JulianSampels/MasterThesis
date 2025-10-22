from abc import ABC, abstractmethod
from functools import partial
import math
import statistics
import multiprocessing as mp
import os

import numpy as np
import torch
import torch_scatter
from tqdm import tqdm
from .pathdata import RelationMaps

import logging
logger = logging.getLogger(__name__)

class BaseCandidateGenerator(ABC):
    """Abstract base class for all candidate generation strategies."""
    
    def __init__(self, max_num_workers: int = 1):
        self.max_num_workers = max_num_workers if max_num_workers and max_num_workers > 0 else os.cpu_count() // 2
        self.pool = None

    def _get_or_create_pool(self, num_processes):
        """Generate or reuse the multiprocessing pool if the pool is empty or has fewer processes."""
        num_processes = min(num_processes, self.max_num_workers)
        if self.pool is None or self.pool._processes < num_processes:
            if self.pool is not None:
                self.close_pool()
            if num_processes >= 64:
                logger.warning(f"Creating a large pool with {num_processes} processes. This may take a while and lead to high memory usage.")
            else:
                logger.info(f"Creating multiprocessing pool with {num_processes} processes.")
            self.pool = mp.Pool(processes=num_processes)
        return self.pool

    def close_pool(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None

    def __del__(self):
        self.close_pool()

    @abstractmethod
    def generate_candidates(self,
        tuples: torch.Tensor,
        scores_rp: torch.Tensor,
        relation_maps: RelationMaps,
        num_groups: int,
        scores_tp: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate candidate triples from tuple predictions.
        
        Args:
            tuples: (N, 2) tensor with head entities in column 0
            scores_rp: (N, R_total) tensor of relation logits
            relation_maps: RelationMaps object with original/inverse relation mappings
            
        Returns:
            candidates: (M, 3) tensor of (head_id, relation_id, tail_id) triples
            scores: (M,) tensor of confidence scores
        """
        pass

    # -------------------------
    # Helpers
    # -------------------------
    @staticmethod
    def _aggregate_logits_per_head(tuples: torch.Tensor, scores_rp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Aggregate (mean) relation logits for each unique head entity using torch_scatter.
        Args:
            tuples: (num_samples, 2) tensor with head entities in column 0 and relations in column 1
            scores_rp: (num_samples, R_total) tensor of relation logits
        Returns:
            unique_heads: (H,) long tensor of unique entity ids;
            scores_rp_grouped: (H, R_total) mean logits per head
        """
        # 1. Collect unique head entities (local indexing)
        unique_heads, inverse_entity_indices = tuples[:, 0].unique(return_inverse=True, sorted=False)
        if unique_heads.size(0) == 0:
            return tuples.new_zeros((0, 3)), tuples.new_zeros((0,), dtype=torch.float32)
        # 2. Aggregate logits per local head index
        # scores_rp has shape (num_samples, R_total); inverse_entity_indices maps each row to its head index
        scores_rp_grouped = torch_scatter.scatter_mean(scores_rp, inverse_entity_indices, dim=0)
        if scores_rp_grouped.shape != scores_rp_grouped.shape:
            logger.warning(f"Needed to aggregated logit, this might be a bad sign as averaging over raw logits (before: {scores_rp.shape}, after: {scores_rp_grouped.shape})!")
            # print(f"before: {scores_rp.shape}, after: {scores_rp_grouped.shape}")
        return unique_heads, scores_rp_grouped


    # -------------------------
    # Analysis print functions
    # -------------------------

    def print_candidate_statistics(
        self,
        candidates: torch.Tensor,
        candidates_group_ids: torch.Tensor,
        gold_triples: torch.Tensor,
        gold_group_ids: torch.Tensor,
        relation_maps: RelationMaps,
        name: str = "Set",
    ) -> None:
        """
        Print basic candidate statistics: total count, unique count, avg per input tuple.

        Args:
            candidates: (M,3) candidate triples tensor
            candidates_group_ids: (M,) long tensor, group id for each candidate triple (aligned with candidates)
            gold_triples: (N,3) gold triples tensor
            gold_group_ids: (N,) long tensor, group id for each gold triple (aligned with gold_triples)
            name: label for print messages
        """
        BaseCandidateGenerator.analyze_total_coverage(candidates, gold_triples, name=name, print_results=True)
        self.analyze_coverage_per_group(candidates, candidates_group_ids, gold_triples, gold_group_ids, relation_maps, name=name)
        return

    @staticmethod
    @torch.no_grad()
    def analyze_total_coverage(
        candidates: torch.Tensor,
        gold_triples: torch.Tensor,
        name: str = "Set",
        print_results: bool = True,
    ) -> tuple[float, float]:
        """
        Compute total (micro) coverage and density of positives in candidate set.

        Args:
            candidates: (M, 3) candidate triples tensor
            gold_triples: (N, 3) gold triples tensor
            name: label for print messages
        Returns:
            total_cov: float in [0,1], fraction of unique gold triples covered by candidates
            pos_frac: float in [0,1], density of candidate triples that are positives
        """
        assert candidates.dim() == 2 and candidates.size(1) == 3, "candidates must be (M,3)"

        total_gold = int(gold_triples.size(0))
        if total_gold == 0:
            return 1.0, 0.0 # full coverage & low density
        elif candidates.numel() == 0:
            return 0.0, 0.0 # no coverage & low density
        
        # deduplicate candidates if needed
        candidates = candidates.unique(dim=0)  

        # Build sets for unique-based coverage
        gold_set = {tuple(row.tolist()) for row in gold_triples.cpu()}
        candidate_set = {tuple(row.tolist()) for row in candidates.cpu()}

        # Coverage over unique gold triples
        positives_in_candidates = sum(1 for g in gold_set if g in candidate_set)
        total_cov = positives_in_candidates / total_gold

        # Candidate positives density over all rows (counts duplicates if present)
        pos_density = positives_in_candidates / int(candidates.size(0))

        if print_results:
            print(f"[Coverage::{name}] Total coverage (micro): {positives_in_candidates} / {total_gold} = {total_cov:.4f}")
            print(f"[Density::{name}] Candidate positives density: {positives_in_candidates} / {int(candidates.size(0))} = {pos_density:.4f}")
        return total_cov, pos_density


    @staticmethod
    def _process_group_for_analyze_coverage_per_group(gids, gold_triples, gold_group_ids, candidates, candidates_group_ids, name):
        torch.set_num_threads(1)  # Prevent thread oversubscription and deadlocks in worker processes
        per_group_cov = {}
        per_group_density = {}
        per_group_count = {}
        covered_total = 0.0
        total_total = 0
        for gid in tqdm(gids, position=2, leave=False, desc="Processing groups"):
            gid = int(gid)
            gold_idx = (gold_group_ids == gid).nonzero(as_tuple=False).flatten()
            if gold_idx.numel() == 0:
                continue
            gold_subset = gold_triples[gold_idx]

            cand_idx = (candidates_group_ids == gid).nonzero(as_tuple=False).flatten()
            cand_subset = candidates[cand_idx] if cand_idx.numel() > 0 else candidates.new_zeros((0, 3), dtype=candidates.dtype)

            cov, dens = BaseCandidateGenerator.analyze_total_coverage(
                candidates=cand_subset,
                gold_triples=gold_subset,
                name=f"{name}|gid={gid}",
                print_results=False,
            )
            count = cand_subset.size(0)
            group_size = int(gold_subset.size(0))
            per_group_cov[gid] = float(cov)
            per_group_density[gid] = float(dens)
            per_group_count[gid] = count
            covered_total += cov * group_size
            total_total += group_size
        return per_group_cov, per_group_density, per_group_count, covered_total, total_total



    @torch.no_grad()
    def analyze_coverage_per_group(
        self,
        candidates: torch.Tensor,
        candidates_group_ids: torch.Tensor,
        gold_triples: torch.Tensor,
        gold_group_ids: torch.Tensor,
        relation_maps: RelationMaps,
        name: str = "Set",
        print_results: bool = True,
    ) -> tuple[float, float, float] | None:
        """
        Print per-group (macro) coverage stats by calling analyze_total_coverage per group,
        plus overall micro coverage. Does not require candidate labels.

        Args:
            candidates: (M,3) candidate triples tensor
            candidates_group_ids: (M,) long tensor, group id for each candidate triple (aligned with candidates)
            gold_triples: (N,3) gold triples tensor
            gold_group_ids: (N,) long tensor, group id for each gold triple (aligned with gold_triples)
            name: label for print messages
        Returns:
            avg_group_cov: float in [0,1], average per-group coverage (macro)
            avg_group_density: float in [0,1], average per-group candidate positive density (macro)
            avg_group_count: float, average number of candidates per group
            or None if no groups with gold triples exist
        """
        if gold_triples.numel() == 0 or gold_group_ids.numel() == 0:
            print(f"[Coverage::{name}] No gold triples or group ids available. Per-group coverage unavailable.")
            return
        if candidates.numel() == 0 or candidates_group_ids.numel() == 0:
            print(f"[Coverage::{name}] No candidates or candidate group ids available. Per-group coverage unavailable.")
            return

        # # Filter gold to original relations and align group ids
        # orig = relation_maps.original_relations_tensor.to(gold_triples.device)
        # mask_gold = torch.isin(gold_triples[:, 1], orig)
        # gold = gold_triples[mask_gold]
        # gold_gids = gold_group_ids[mask_gold].to(torch.long)

        if gold_triples.numel() == 0:
            print(f"[Coverage::{name}] No gold triples with original relations. Per-group coverage unavailable.")
            return

        # Per-group coverage using group-specific subsets of candidates and gold
        unique_gids = torch.unique(gold_group_ids)
        per_group_cov = {}
        per_group_density = {}
        per_group_count = {}
        covered_total = 0.0
        total_total = 0

        # Share tensors for multiprocessing
        gold_triples.share_memory_()
        gold_group_ids.share_memory_()
        candidates.share_memory_()
        candidates_group_ids.share_memory_()

        # Parallel processing: divide unique_gids into num_workers chunks
        num_workers = min(self.max_num_workers, len(unique_gids))
        chunks = np.array_split(unique_gids.tolist(), num_workers)
        self.pool = self._get_or_create_pool(num_workers)
        worker_function = partial(BaseCandidateGenerator._process_group_for_analyze_coverage_per_group, 
                                  gold_triples=gold_triples, 
                                  gold_group_ids=gold_group_ids, 
                                  candidates=candidates, 
                                  candidates_group_ids=candidates_group_ids, 
                                  name=name)
        for result in tqdm(self.pool.imap_unordered(worker_function, chunks), 
                           desc=f"Coverage per group [{name}]", 
                           leave=False, 
                           total=len(chunks)):
            per_group_cov_chunk, per_group_density_chunk, per_group_count_chunk, covered_total_chunk, total_total_chunk = result
            per_group_cov.update(per_group_cov_chunk)
            per_group_density.update(per_group_density_chunk)
            per_group_count.update(per_group_count_chunk)
            covered_total += covered_total_chunk
            total_total += total_total_chunk

        if not per_group_cov:
            print(f"[Coverage::{name}] No groups with gold triples. Per-group coverage unavailable.")
            return
        
        if print_results:
            # Print statistics
            print(f"{f'[Group count::{name}].':<50}{len(per_group_cov)}")
            print(f"{f'[Coverage per group::{name}] Macro. ':<50}"
                f"Avg: {statistics.mean(per_group_cov.values()):.4f} | "
                f"Min: {min(per_group_cov.values()):.4f} | "
                f"Max: {max(per_group_cov.values()):.4f} | "
                f"Deciles: {[round(q, 2) for q in statistics.quantiles(per_group_cov.values(), n=10)]}")
            print(f"{f'[Density per group::{name}] Density. ':<50}"
                f"Avg: {statistics.mean(per_group_density.values()):.4f} | "
                f"Min: {min(per_group_density.values()):.4f} | "
                f"Max: {max(per_group_density.values()):.4f} | "
                f"Deciles: {[round(q, 2) for q in statistics.quantiles(per_group_density.values(), n=10)]}")
            print(f"{f'[Candidate Distribution over groups::{name}]. ':<50}"
                f"Avg: {statistics.mean(per_group_count.values()):.2f} | "
                f"Min: {min(per_group_count.values())} | "
                f"Max: {max(per_group_count.values())} | "
                f"Deciles: {[int(q) for q in statistics.quantiles(per_group_count.values(), n=10)]}")
        return statistics.mean(per_group_cov.values()), statistics.mean(per_group_density.values()), statistics.mean(per_group_count.values())

    def _compute_adaptive_head_block_size(self, E: int, k_total: int) -> int:
        """
        Compute adaptive head_block_size based on entity count E and total candidates k_total.
        Returns the head_block_size to use, setting self.head_block_size if it was None.
        """
        if self.head_block_size is not None:
            return self.head_block_size

        # Adaptive head_block_size based on number of unique entities
        memory_budget_bytes = int(2.5e8)  # 250MB per worker (adjust based on your RAM/GPU)
        bytes_per_float = 4
        bytes_per_long = 8

        # Memory for top-k buffers per worker, based on k_total
        buffer_memory_bytes = k_total * (bytes_per_float + 3 * bytes_per_long)
        
        # Remaining memory for score calculation blocks
        score_calc_memory_budget = memory_budget_bytes - buffer_memory_bytes
        # assert score_calc_memory_budget > 0, "Not enough memory budget for score calculation blocks."
        if score_calc_memory_budget <= 0:
            head_block_size = input(f"Warning: Not enough memory budget (missing {abs(score_calc_memory_budget)} bytes) for score calculation blocks. Please enter a fixed head_block_size (int) to use (e.g., 1, 10, 100): ")
            head_block_size = int(head_block_size)
        else: 
            max_head_block_size = 1024**10  # Cap to avoid too large batches
            # The dominant memory usage in a block is multiple (B, E) tensors (e.g., V, v_top, vals_flat).
            # Estimate peak as 4 * (B, E) for safety.
            head_block_size_calc = int(score_calc_memory_budget // (4 * E * bytes_per_float)) if E > 0 else 1
            head_block_size = min(max_head_block_size, max(100, head_block_size_calc))

        estimated_block_memory = 4 * head_block_size * E * bytes_per_float
        total_estimated_memory_per_worker = buffer_memory_bytes + estimated_block_memory
        logger.info(f"Using adaptive head_block_size: {head_block_size} | Estimated total memory per worker: {total_estimated_memory_per_worker / 1e6:.2f} MB (buffer: {buffer_memory_bytes / 1e6:.2f} MB, block: {estimated_block_memory / 1e6:.2f} MB)")
        # print(f"Using adaptive head_block_size: {head_block_size} | Estimated total memory per worker: {total_estimated_memory_per_worker / 1e6:.2f} MB (buffer: {buffer_memory_bytes / 1e6:.2f} MB, block: {estimated_block_memory / 1e6:.2f} MB)")
        # input("Press Enter to continue...")

        return head_block_size

class CandidateGeneratorGlobal(BaseCandidateGenerator):
    def __init__(self, p: float, q: float, temperature: float, alpha: float, per_group_cap: int, normalize_mode: str = "per_head",
                 rel_block_size: int | None = None,
                 head_block_size: int | None = None,
                 max_num_workers: int | None = None,
                 phase1_loss_fn: str = "bce",
                 ):
        super().__init__(max_num_workers=max_num_workers)
        self.p = p                              # keep candidates with P >= p (global threshold)
        self.q = q                              # use as fraction-cap: cap = ceil((1-q) * |E|*|R|*|E|)
        self.temperature = temperature          # temperature for softmax calibration
        self.alpha = alpha                      # weight for tail vs head log-probs
        self.per_group_cap = per_group_cap      # per-group cap, used to compute total cap
        self.normalize_mode = normalize_mode    # "per_head" | "global_joint" | "none"
        self.phase1_loss_fn = phase1_loss_fn    # "bce" or "poisson"

        # new knobs for memory and parallelism
        self.rel_block_size = max(1, int(rel_block_size)) if rel_block_size is not None else None
        self.head_block_size = max(1, int(head_block_size)) if head_block_size is not None else None

        # sanity checks
        assert self.temperature > 0, "temperature must be > 0"
        assert 0.0 <= self.alpha <= 1.0, "alpha must be in [0,1]"
        assert self.normalize_mode in ("per_head", "global_joint", "per_relation", "none"), "normalize_mode invalid"
        if self.p is not None:
            assert 0.0 <= self.p <= 1.0, "p must be in [0,1]"
        if self.q is not None:
            assert 0.0 <= self.q < 1.0, "q must be in [0,1)"

    @staticmethod
    def _process_chunk(chunk_indices: torch.Tensor, log_p_head_2d: torch.Tensor, log_p_tail_2d: torch.Tensor, alpha: float, k: int, E: int, R: int, head_block_size: int):
        """Worker function for parallel chunk processing with head blocking to control memory."""
        torch.set_num_threads(1)  # 1 thread per worker as we parallelize over processes, otherwise it breaks
        C = len(chunk_indices)
        a_h = float(1 - alpha)
        a_t = float(alpha)
        B = max(1, int(head_block_size))

        # Global buffers for this chunk
        chunk_vals = torch.full((k,), float("-inf"), dtype=torch.float32)
        chunk_r    = torch.full((k,), -1, dtype=torch.long)
        chunk_h    = torch.full((k,), -1, dtype=torch.long)
        chunk_t    = torch.full((k,), -1, dtype=torch.long)
        chunk_filled = 0

        # Preload per-relation tail inverse
        t_chunk = a_t * log_p_tail_2d[chunk_indices, :]  # (C, E)

        for idx in tqdm(range(C), desc="Processing relations in chunk", position=2, leave=False):
            r_global = chunk_indices[idx]
            t_inv = t_chunk[idx, :]  # (E,)

            for h0 in tqdm(range(0, E, B), desc="Processing heads in blocks", position=3, leave=False):
                h1 = min(E, h0 + B)
                head_term = a_h * log_p_head_2d[h0:h1, r_global]  # (B,)
                V = head_term.unsqueeze(1) + t_inv.unsqueeze(0)  # (B, E)

                vals_flat = V.reshape(-1)
                k_sub = min(k, vals_flat.numel())
                vals_sub, idx_flat = torch.topk(vals_flat, k=k_sub, largest=True)

                h_idx_sub = torch.arange(h0, h1, dtype=torch.long).unsqueeze(1).repeat(1, E).reshape(-1)[idx_flat]
                t_idx_sub = torch.arange(E, dtype=torch.long).repeat(B)[idx_flat]
                r_idx_sub = torch.full_like(h_idx_sub, r_global)

                # Merge into chunk top-k
                if chunk_filled == 0:
                    take = min(k, vals_sub.numel())
                    chunk_vals[:take] = vals_sub[:take]
                    chunk_r[:take]    = r_idx_sub[:take]
                    chunk_h[:take]    = h_idx_sub[:take]
                    chunk_t[:take]    = t_idx_sub[:take]
                    chunk_filled = take
                else:
                    cand_vals = torch.cat([chunk_vals[:chunk_filled], vals_sub], dim=0)
                    cand_r    = torch.cat([chunk_r[:chunk_filled],    r_idx_sub], dim=0)
                    cand_h    = torch.cat([chunk_h[:chunk_filled],    h_idx_sub], dim=0)
                    cand_t    = torch.cat([chunk_t[:chunk_filled],    t_idx_sub], dim=0)
                    if cand_vals.numel() > k:
                        vtop, order = torch.topk(cand_vals, k=k, largest=True)
                        chunk_vals[:k] = vtop
                        chunk_r[:k]    = cand_r[order]
                        chunk_h[:k]    = cand_h[order]
                        chunk_t[:k]    = cand_t[order]
                        chunk_filled = k
                    else:
                        chunk_vals[:cand_vals.numel()] = cand_vals
                        chunk_r[:cand_vals.numel()]    = cand_r
                        chunk_h[:cand_vals.numel()]    = cand_h
                        chunk_t[:cand_vals.numel()]    = cand_t
                        chunk_filled = cand_vals.numel()

        return chunk_vals[:chunk_filled], chunk_r[:chunk_filled], chunk_h[:chunk_filled], chunk_t[:chunk_filled]

    def _global_topk_joint_streaming(
        self,
        log_p_head_2d: torch.Tensor,
        log_p_tail_2d: torch.Tensor,
        k_total: int,
    ):
        """
        Parallel global top-k over all (h, r, t), processing relation chunks in parallel.
        Builds full (C, E, E) per chunk but distributes across processes.
        """
        E, R = log_p_head_2d.shape
        assert log_p_tail_2d.shape == (R, E)

        head_block_size = self._compute_adaptive_head_block_size(E, k_total)

        # Share tensors for multiprocessing
        log_p_head_2d.share_memory_()
        log_p_tail_2d.share_memory_()

        # Global buffers
        top_vals = torch.full((k_total,), float("-inf"), dtype=torch.float32)
        top_r    = torch.full((k_total,), -1, dtype=torch.long)
        top_h    = torch.full((k_total,), -1, dtype=torch.long)
        top_t    = torch.full((k_total,), -1, dtype=torch.long)
        filled = 0

        # Multiprocessing: distribute relation chunks across processes to leverage multi-core CPUs
        # Each process computes top-k for its chunk and returns partial results, which are merged globally
        num_workers = min(self.max_num_workers, math.ceil(R / self.rel_block_size if self.rel_block_size is not None else R))
        chunks = np.array_split(np.arange(R), num_workers)
        chunks = [torch.tensor(chunk, dtype=torch.long) for chunk in chunks]
        self.pool = self._get_or_create_pool(num_workers)
        # Submit jobs
        worker_function = partial(CandidateGeneratorGlobal._process_chunk, log_p_head_2d=log_p_head_2d, log_p_tail_2d=log_p_tail_2d, alpha=self.alpha, k=k_total, E=E, R=R, head_block_size=head_block_size)
        results_iterator = self.pool.imap_unordered(worker_function, chunks)
        # Process results in batches to control memory usage
        batch_size = max(16, len(chunks) // 2)  # Concatenate when half of chunks are collected efficient for cpu with high core count
        collected_chunks = 0
        batch_vals = []
        batch_r = []
        batch_h = []
        batch_t = []
        
        for vals_chunk, r_chunk, h_chunk, t_chunk in tqdm(results_iterator, desc="Processing parallel chunks in batches", unit="chunk", leave=False, total=len(chunks)):
            batch_vals.append(vals_chunk)
            batch_r.append(r_chunk)
            batch_h.append(h_chunk)
            batch_t.append(t_chunk)
            collected_chunks += 1
            
            if len(batch_vals) >= batch_size or collected_chunks == len(chunks):
                # Concatenate the current batch
                cand_vals = torch.cat(batch_vals, dim=0)
                cand_r = torch.cat(batch_r, dim=0)
                cand_h = torch.cat(batch_h, dim=0)
                cand_t = torch.cat(batch_t, dim=0)
                
                # Merge batch into global top-k
                if filled == 0:
                    take = min(k_total, cand_vals.numel())
                    top_vals[:take] = cand_vals[:take]
                    top_r[:take] = cand_r[:take]
                    top_h[:take] = cand_h[:take]
                    top_t[:take] = cand_t[:take]
                    filled = take
                else:
                    global_cand_vals = torch.cat([top_vals[:filled], cand_vals], dim=0)
                    global_cand_r = torch.cat([top_r[:filled], cand_r], dim=0)
                    global_cand_h = torch.cat([top_h[:filled], cand_h], dim=0)
                    global_cand_t = torch.cat([top_t[:filled], cand_t], dim=0)
                    if global_cand_vals.numel() > k_total:
                        vtop, order = torch.topk(global_cand_vals, k=k_total, largest=True)
                        top_vals[:k_total] = vtop
                        top_r[:k_total] = global_cand_r[order]
                        top_h[:k_total] = global_cand_h[order]
                        top_t[:k_total] = global_cand_t[order]
                        filled = k_total
                    else:
                        top_vals[:global_cand_vals.numel()] = global_cand_vals
                        top_r[:global_cand_vals.numel()] = global_cand_r
                        top_h[:global_cand_vals.numel()] = global_cand_h
                        top_t[:global_cand_vals.numel()] = global_cand_t
                        filled = global_cand_vals.numel()
                
                # Clear the batch
                batch_vals.clear()
                batch_r.clear()
                batch_h.clear()
                batch_t.clear()
                batch_size = max(batch_size // 2, 16)
        
        # All chunks should be processed in batches; raise error if any remaining
        if batch_vals:
            raise ValueError(f"Some chunks ({len(batch_vals)}) were not processed in batches. Batch size was {batch_size}, total chunks {len(chunks)}. This should not happen.")

        return top_vals[:filled], top_r[:filled], top_h[:filled], top_t[:filled]

    def generate_candidates(self,
        tuples: torch.Tensor,
        scores_rp: torch.Tensor,
        relation_maps: RelationMaps,
        num_groups: int,
        scores_tp: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Efficiently generate candidate (head, relation, tail) triples and their joint probabilities
        for two-phase PathE training, using global top-k streaming to avoid OOM on large graphs.

        Candidate selection logic:
          1. Compute an effective cap from quantile q (cap_q = ceil((1-q) * E*R*E)), and/or cap_candidates.
             The final cap is min(cap_candidates, cap_q) if both are set.
          2. Compute joint log-probabilities for all (h, r, t) triples using:
                joint(h, r, t) = (1-alpha) * log P(r|h) + alpha * log P(r^{-1}|t)
             without materializing the full (E, R, E) tensor.
          3. Use a streaming top-k algorithm to keep only the highest-probability candidates globally.
          4. Stack all candidate triples and their scores.
          5. If a global probability threshold p is set, filter candidates by score >= p.
             Always keep at least one candidate to avoid empty sets downstream.

        Args:
            tuples: (num_samples, 2) tensor, entity in col 0.
            relation_maps: RelationMaps object mapping original to inverse relations.
            scores_rp: (num_samples, num_relations) tensor of per-sample relation logits.
            num_groups: int, number of groups for candidate cap.
            p: Optional[float], global probability threshold for candidate filtering.
            q: Optional[float] in [0,1), quantile threshold for global top-k (keeps top (1-q) fraction).
            temperature: float, softmax temperature for calibration.
            alpha: float in [0,1], weight for tail vs head log-probabilities.
            cap_candidates: Optional[int], hard cap on number of candidates.

        Returns:
            candidates: (N, 3) tensor of (head_id, relation_id, tail_id) triples.
            scores: (N,) tensor of joint probabilities for each candidate.
        """
        assert scores_rp is not None, "scores_rp required."

        # Aggregate logits per unique head entity
        entities, scores_rp_grouped = self._aggregate_logits_per_head(tuples, scores_rp)
        E = entities.size(0)
        device = scores_rp_grouped.device

        # 3. Resolve original & inverse relation ids
        original_relations = relation_maps.original_relations_tensor.to(device)  # (R,)
        inverse_relations  = relation_maps.inverse_relations_tensor.to(device)   # (R,)
        R = original_relations.size(0)

        # 4. Slice logits for original and inverse relation columns
        head_logits_subset = scores_rp_grouped[:, original_relations]   # (E, R)
        tail_logits_subset = scores_rp_grouped[:, inverse_relations]    # (E, R)

        # # Apply softplus for Poisson loss
        # if self.phase1_loss_fn == "poisson":
        #     head_logits_subset = torch.exp(head_logits_subset)
        #     tail_logits_subset = torch.exp(tail_logits_subset)

        # 5. Calibrated log-probabilities or raw logits based on normalize_mode
        if self.normalize_mode == "per_head":
            log_p_head_2d = torch.log_softmax(head_logits_subset / self.temperature, dim=1).to(torch.float32).cpu()                 # (E, R)
            log_p_tail_2d = torch.log_softmax(tail_logits_subset / self.temperature, dim=1).to(torch.float32).cpu().transpose(0,1) # (R, E)
        elif self.normalize_mode == "global_joint":
            z_hr = (head_logits_subset / self.temperature).to(torch.float32).cpu()             # (E, R)
            log_p_head_2d = torch.log_softmax(z_hr.reshape(-1), dim=0).reshape(E, R)          # joint over all (h,r)
            z_tr = (tail_logits_subset / self.temperature).to(torch.float32).cpu().transpose(0,1)  # (R, E)
            log_p_tail_2d = torch.log_softmax(z_tr.reshape(-1), dim=0).reshape(R, E)          # joint over all (r,t)
        elif self.normalize_mode == "per_relation":
            log_p_head_2d = torch.log_softmax(head_logits_subset / self.temperature, dim=0).to(torch.float32).cpu()                 # (E, R)
            log_p_tail_2d = torch.log_softmax(tail_logits_subset / self.temperature, dim=0).to(torch.float32).cpu().transpose(0,1) # (R, E)
        else:  # "none" -> use temperature-scaled logits as log-scores
            log_p_head_2d = (head_logits_subset / self.temperature).to(torch.float32).cpu()                 # (E, R)
            log_p_tail_2d = (tail_logits_subset / self.temperature).to(torch.float32).cpu().transpose(0,1) # (R, E)

        # Derive effective cap from q first (before any threshold). This bounds the search space.
        total = int(E) * int(R) * int(E)
        # Compute base cap from group strategy
        effective_cap = num_groups * self.per_group_cap
        if self.q is not None:
            cap_q = max(1, int(math.ceil((1.0 - self.q) * total)))
            if effective_cap is not None and effective_cap < cap_q:
                logger.warning(f"effective_cap < cap_q from q-quantile. Using smaller effective_cap {effective_cap} instead of {cap_q}.")
            effective_cap = cap_q if effective_cap is None else min(effective_cap, cap_q)
        if effective_cap is None:
            raise ValueError("Candidate generation requires a cap (q or per_group_cap). Threshold-only (p) is unsafe for large graphs.")

        # Compute global top-k in a streaming fashion without O(E*R*E) memory
        # Peak RAM ~ rel_block_size * E * E * 4 bytes
        # memory_limit_gb = 1.0  # target RAM limit in GB
        # bytes_per_float = 4
        # max_bytes = int(memory_limit_gb * (1024**3))
        # rel_block_size = max(1, max_bytes // max(1, (E * E * bytes_per_float)))

        # torch.set_num_threads(int(self.num_threads))

        top_log_vals, r_idx, h_idx, t_idx = self._global_topk_joint_streaming(
            log_p_head_2d=log_p_head_2d,
            log_p_tail_2d=log_p_tail_2d,
            k_total=int(effective_cap),
        )

        # Build candidate triples (global entity indexing)
        heads_tensor = entities[h_idx]
        rels_tensor  = original_relations[r_idx].cpu()
        tails_tensor = entities[t_idx]
        candidates = torch.stack([heads_tensor, rels_tensor, tails_tensor], dim=1)

        # Convert to probabilities for thresholding
        scores = torch.exp(top_log_vals)
        # 6. Apply global threshold p if provided
        if self.p is not None:
            keep_mask = scores >= self.p
            if keep_mask.any():
                candidates = candidates[keep_mask]
                scores = scores[keep_mask]
            else:
                # Keep at least the best entry to avoid empty sets downstream
                best = torch.argmax(scores)
                candidates = candidates[best:best+1]
                scores = scores[best:best+1]

        return candidates, scores

class CandidateGeneratorPerHead(BaseCandidateGenerator):
    def __init__(self, per_group_cap: int, alpha: float, phase1_loss_fn: str = "bce"):
        super().__init__()
        self.per_group_cap = per_group_cap    # number of (r,t) pairs to keep per head entity
        self.alpha = alpha  # weight for tail vs head logits
        self.phase1_loss_fn = phase1_loss_fn  # "bce" or "poisson"
        assert self.per_group_cap and self.per_group_cap > 0, "per_group_cap must be > 0"
        assert 0.0 <= self.alpha <= 1.0, "alpha must be in [0,1]"

    def _aggregate_logits_per_entity(tuples_2col: torch.Tensor,
                                     scores_rp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Aggregate (mean) relation logits for each unique head entity using torch_scatter.
        Returns:
            unique_heads: (H,) long tensor of entity ids
            agg_logits:   (H, R_total) mean logits per head
        """
        heads = tuples_2col[:, 0]
        unique_heads, inverse = heads.unique(return_inverse=True, sorted=False)
        # scores_rp has shape (num_samples, R_total); inverse maps each row to its head index
        agg_logits = torch_scatter.scatter_mean(scores_rp, inverse, dim=0)
        return unique_heads, agg_logits

    def generate_candidates(self, 
            tuples_all: torch.Tensor,
            scores_rp_all: torch.Tensor,
            relation_maps: RelationMaps,
            num_groups: int,
            scores_tp: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        For each head entity h present in tuples_all:
           Score (h, r, t) = P(r|h) * P(r^{-1}|t)
           Keep top-k (r, t) pairs.
        Only original relations are considered (relation_maps.original_relation_to_inverse_relation keys).
        """
        # 1. Aggregate logits per entity (treat any entity that appears as head)
        entity_ids, scores_rp_grouped = self._aggregate_logits_per_head(tuples_all, scores_rp_all)  # (E',), (E', R_total)
        device = scores_rp_grouped.device
        
        # 2. Prepare relation mappings
        orig2inv = relation_maps.original_relation_to_inverse_relation
        if len(orig2inv) == 0:
            return {}
        
        # 3. Resolve original & inverse relation ids
        original_relation_ids = relation_maps.original_relations_tensor.to(device)  # (R,)
        inverse_relation_ids  = relation_maps.inverse_relations_tensor.to(device)   # (R,)


        # 4. Slice logits for originals & inverses then softmax separately
        logits_orig = scores_rp_grouped[:, original_relation_ids]          # (E', R)
        logits_inv  = scores_rp_grouped[:, inverse_relation_ids]           # (E', R)

        # # Apply softplus for Poisson loss
        # if self.phase1_loss_fn == "poisson":
        #     logits_orig = torch.exp(logits_orig)
        #     logits_inv = torch.exp(logits_inv)

        prob_r_given_h = torch.softmax(logits_orig, dim=1)      # (E', R)
        prob_rinv_given_t = torch.softmax(logits_inv, dim=1)    # (E', R)
        prob_rinv_T = prob_rinv_given_t.transpose(0, 1).contiguous()  # (R, E')

        # Eprime, R = prob_r_given_h.shape
        # k_eff = min(k, R * Eprime)
        # head_to_topk: dict[int, torch.Tensor] = {}

        # for h_idx in tqdm(range(Eprime), desc="Generating Top-K per head", leave=False):
        #     p_r_h = prob_r_given_h[h_idx]              # (R,)
        #     # (R,E') matrix of scores
        #     scores = p_r_h.unsqueeze(1) * prob_rinv_T  # (R,E')
        #     flat = scores.view(-1)                     # (R*E',)
        #     topk_vals, topk_idx = torch.topk(flat, k=k_eff, largest=True, sorted=True)
        #     rel_local = topk_idx // Eprime             # (k_eff,)
        #     tail_local = topk_idx % Eprime             # (k_eff,)
        #     rel_global = original_relation_ids[rel_local]   # map back to global relation ids
        #     tail_global = entity_ids[tail_local]       # map local entity index to global entity id
        #     head_to_topk[int(entity_ids[h_idx].item())] = torch.stack([rel_global, tail_global], dim=1)

        # return head_to_topk
        E, R = prob_r_given_h.shape
        k_eff = min(self.per_group_cap, R * E)
        
        all_candidates = []
        all_scores = []
        
        for h_idx in tqdm(range(E), desc="Generating Top-K per head", leave=False):
            p_r_h = prob_r_given_h[h_idx]
            scores = (p_r_h ** (1 - self.alpha)).unsqueeze(1) * (prob_rinv_T ** self.alpha)  # (R, E)
            flat = scores.view(-1)
            
            topk_vals, topk_idx = torch.topk(flat, k=k_eff, largest=True, sorted=True)
            rel_local = topk_idx // E
            tail_local = topk_idx % E
            
            # Map to global IDs
            rel_global = original_relation_ids[rel_local]
            tail_global = entity_ids[tail_local]
            head_global = entity_ids[h_idx].expand_as(rel_global)
            
            candidates_h = torch.stack([head_global, rel_global, tail_global], dim=1)
            all_candidates.append(candidates_h)
            all_scores.append(topk_vals)
        
        # Concatenate all candidates and scores
        candidates = torch.cat(all_candidates, dim=0) if all_candidates else tuples_all.new_zeros((0, 3))
        scores = torch.cat(all_scores, dim=0) if all_scores else tuples_all.new_zeros((0,), dtype=torch.float32)
        
        return candidates, scores


class CandidateGeneratorGlobalWithTail(BaseCandidateGenerator):
    def __init__(self, p: float, q: float, temperature: float, alpha: float, beta: float, per_group_cap: int, normalize_mode: str = "per_head",
                 rel_block_size: int = None,
                 head_block_size: int | None = None,
                 max_num_workers: int | None = None,
                 phase1_loss_fn: str = "bce",
                #  num_threads: int | None = None
                 ):
        super().__init__(max_num_workers=max_num_workers)
        self.p = p                              # keep candidates with P >= p (global threshold)
        self.q = q                              # use as fraction-cap: cap = ceil((1-q) * |E|*|R|*|E|)
        self.temperature = temperature          # temperature for softmax calibration
        self.alpha = alpha                      # weight for tail vs head
        self.beta = beta                        # weight for tail prediction prob
        self.per_group_cap = per_group_cap      # per-group cap, used to compute total cap
        self.normalize_mode = normalize_mode    # "per_head" | "global_joint" | "none"
        self.phase1_loss_fn = phase1_loss_fn    # "bce" or "poisson"

        # new knobs for memory and parallelism
        self.rel_block_size = max(1, int(rel_block_size)) if rel_block_size is not None else None
        self.head_block_size = max(1, int(head_block_size)) if head_block_size is not None else None

        # sanity checks
        assert self.temperature > 0, "temperature must be > 0"
        assert 0.0 <= self.alpha <= 1.0, "alpha must be in [0,1]"
        assert 0.0 <= self.beta <= 1.0, "beta must be in [0,1]"
        assert self.alpha + self.beta <= 1.0, "alpha + beta must be <= 1.0 (gamma = 1 - alpha - beta)"
        assert self.normalize_mode in ("per_head", "global_joint", "per_relation", "none"), "normalize_mode invalid"
        if self.p is not None:
            assert 0.0 <= self.p <= 1.0, "p must be in [0,1]"
        if self.q is not None:
            assert 0.0 <= self.q < 1.0, "q must be in [0,1)"

    @staticmethod
    def _process_chunk(chunk_indices: torch.Tensor, log_p_head_2d: torch.Tensor, log_p_tail_2d: torch.Tensor, log_p_t_given_h_2d: torch.Tensor, alpha: float, beta: float, k: int, E: int, R: int, head_block_size: int):
        """Worker function for parallel chunk processing with tail prediction."""
        torch.set_num_threads(1)  # 1 thread per worker as we parallelize over processes, otherwise it breaks
        C = len(chunk_indices)
        a_h = (1 - beta) * (1 - alpha)
        a_t_pred = beta
        a_t_inv = (1 - beta) * alpha
        B = max(1, int(head_block_size))

        # Global buffers for this chunk
        chunk_vals = torch.full((k,), float("-inf"), dtype=torch.float32)
        chunk_r    = torch.full((k,), -1, dtype=torch.long)
        chunk_h    = torch.full((k,), -1, dtype=torch.long)
        chunk_t    = torch.full((k,), -1, dtype=torch.long)
        chunk_filled = 0

        # Preload per-relation tail inverse
        t_inv_chunk = (a_t_inv * log_p_tail_2d[chunk_indices, :])  # (C, E)

        for idx in tqdm(range(C), desc="Processing relations in chunk", position=2, leave=False):
            r_global = chunk_indices[idx]
            t_inv = t_inv_chunk[idx, :]  # (E,)

            for h0 in tqdm(range(0, E, B), desc="Processing heads in blocks", position=3, leave=False):
                h1 = min(E, h0 + B)
                V = (a_t_pred * log_p_t_given_h_2d[h0:h1, :]) + t_inv.unsqueeze(0)  # (B, E)
                v_top = V + (a_h * log_p_head_2d[h0:h1, r_global]).unsqueeze(1)     # (B, E)

                vals_flat = v_top.reshape(-1)
                k_sub = min(k, vals_flat.numel())
                vals_sub, idx_flat = torch.topk(vals_flat, k=k_sub, largest=True)

                h_idx_sub = torch.arange(h0, h1, dtype=torch.long).unsqueeze(1).repeat(1, E).reshape(-1)[idx_flat]
                t_idx_sub = torch.arange(E, dtype=torch.long).repeat(B)[idx_flat]
                r_idx_sub = torch.full_like(h_idx_sub, r_global)

                # Merge into chunk top-k
                if chunk_filled == 0:
                    take = min(k, vals_sub.numel())
                    chunk_vals[:take] = vals_sub[:take]
                    chunk_r[:take]    = r_idx_sub[:take]
                    chunk_h[:take]    = h_idx_sub[:take]
                    chunk_t[:take]    = t_idx_sub[:take]
                    chunk_filled = take
                else:
                    cand_vals = torch.cat([chunk_vals[:chunk_filled], vals_sub], dim=0)
                    cand_r    = torch.cat([chunk_r[:chunk_filled],    r_idx_sub], dim=0)
                    cand_h    = torch.cat([chunk_h[:chunk_filled],    h_idx_sub], dim=0)
                    cand_t    = torch.cat([chunk_t[:chunk_filled],    t_idx_sub], dim=0)
                    if cand_vals.numel() > k:
                        vtop, order = torch.topk(cand_vals, k=k, largest=True)
                        chunk_vals[:k] = vtop
                        chunk_r[:k]    = cand_r[order]
                        chunk_h[:k]    = cand_h[order]
                        chunk_t[:k]    = cand_t[order]
                        chunk_filled = k
                    else:
                        chunk_vals[:cand_vals.numel()] = cand_vals
                        chunk_r[:cand_vals.numel()]    = cand_r
                        chunk_h[:cand_vals.numel()]    = cand_h
                        chunk_t[:cand_vals.numel()]    = cand_t
                        chunk_filled = cand_vals.numel()

        return chunk_vals[:chunk_filled], chunk_r[:chunk_filled], chunk_h[:chunk_filled], chunk_t[:chunk_filled]

    def _global_topk_joint_streaming(
        self,
        log_p_head_2d: torch.Tensor,
        log_p_tail_2d: torch.Tensor,
        log_p_t_given_h_2d: torch.Tensor,
        k_total: int,
    ):
        """
        Parallel global top-k with tail prediction, processing relation chunks in parallel.
        Builds full scores per chunk but avoids E×E materialization via batched head processing.
        """
        E, R = log_p_head_2d.shape
        assert log_p_tail_2d.shape == (R, E)
        assert log_p_t_given_h_2d.shape == (E, E)

        head_block_size = self._compute_adaptive_head_block_size(E, k_total)

        # Share tensors
        log_p_head_2d.share_memory_()
        log_p_tail_2d.share_memory_()
        log_p_t_given_h_2d.share_memory_()

        # Global buffers
        top_vals = torch.full((k_total,), float("-inf"), dtype=torch.float32)
        top_r    = torch.full((k_total,), -1, dtype=torch.long)
        top_h    = torch.full((k_total,), -1, dtype=torch.long)
        top_t    = torch.full((k_total,), -1, dtype=torch.long)
        filled = 0

        # Multiprocessing: distribute relation chunks across processes to leverage multi-core CPUs
        # Each process computes top-k for its chunk (with batched head processing to control memory) and returns partial results, which are merged globally
        num_workers = min(self.max_num_workers, math.ceil(R / self.rel_block_size if self.rel_block_size is not None else R))
        chunks = np.array_split(np.arange(R), num_workers)
        chunks = [torch.tensor(chunk, dtype=torch.long) for chunk in chunks]
        self.pool = self._get_or_create_pool(num_workers)

        worker_function = partial(CandidateGeneratorGlobalWithTail._process_chunk, log_p_head_2d=log_p_head_2d, log_p_tail_2d=log_p_tail_2d, log_p_t_given_h_2d=log_p_t_given_h_2d, alpha=self.alpha, beta=self.beta, k=k_total, E=E, R=R, head_block_size=head_block_size)
        results_iterator = self.pool.imap_unordered(worker_function, chunks)
        # Process results in batches to control memory usage
        batch_size = max(16, len(chunks) // 2)  # Concatenate when half of chunks are collected efficient for cpu with high core count
        collected_chunks = 0
        batch_vals = []
        batch_r = []
        batch_h = []
        batch_t = []
        
        for vals_chunk, r_idx, h_idx, t_idx in tqdm(results_iterator, desc="Processing parallel chunks in batches", unit="chunk", leave=False, total=len(chunks)):
            batch_vals.append(vals_chunk)
            batch_r.append(r_idx)
            batch_h.append(h_idx)
            batch_t.append(t_idx)
            collected_chunks += 1
            
            if len(batch_vals) >= batch_size or collected_chunks == len(chunks):
                # Concatenate the current batch
                cand_vals = torch.cat(batch_vals, dim=0)
                cand_r = torch.cat(batch_r, dim=0)
                cand_h = torch.cat(batch_h, dim=0)
                cand_t = torch.cat(batch_t, dim=0)
                
                # Merge batch into global top-k
                if filled == 0:
                    take = min(k_total, cand_vals.numel())
                    top_vals[:take] = cand_vals[:take]
                    top_r[:take] = cand_r[:take]
                    top_h[:take] = cand_h[:take]
                    top_t[:take] = cand_t[:take]
                    filled = take
                else:
                    global_cand_vals = torch.cat([top_vals[:filled], cand_vals], dim=0)
                    global_cand_r = torch.cat([top_r[:filled], cand_r], dim=0)
                    global_cand_h = torch.cat([top_h[:filled], cand_h], dim=0)
                    global_cand_t = torch.cat([top_t[:filled], cand_t], dim=0)
                    if global_cand_vals.numel() > k_total:
                        vtop, order = torch.topk(global_cand_vals, k=k_total, largest=True)
                        top_vals[:k_total] = vtop
                        top_r[:k_total] = global_cand_r[order]
                        top_h[:k_total] = global_cand_h[order]
                        top_t[:k_total] = global_cand_t[order]
                        filled = k_total
                    else:
                        top_vals[:global_cand_vals.numel()] = global_cand_vals
                        top_r[:global_cand_vals.numel()] = global_cand_r
                        top_h[:global_cand_vals.numel()] = global_cand_h
                        top_t[:global_cand_vals.numel()] = global_cand_t
                        filled = global_cand_vals.numel()
                
                # Clear the batch
                batch_vals.clear()
                batch_r.clear()
                batch_h.clear()
                batch_t.clear()
                batch_size = max(batch_size // 2, 16)
        
        # All chunks should be processed in batches; raise error if any remaining
        if batch_vals:
            raise ValueError(f"Some chunks ({len(batch_vals)}) were not processed in batches. Batch size was {batch_size}, total chunks {len(chunks)}. This should not happen.")

        return top_vals[:filled], top_r[:filled], top_h[:filled], top_t[:filled]

    def generate_candidates(self,
        tuples: torch.Tensor,
        scores_rp: torch.Tensor,
        relation_maps: RelationMaps,
        num_groups: int,
        scores_tp: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Efficiently generate candidate (head, relation, tail) triples and their joint probabilities
        for two-phase PathE training, using global top-k streaming to avoid OOM on large graphs.
        Now includes tail prediction P(t|h) in the scoring.

        Candidate selection logic:
          1. Compute an effective cap from quantile q (cap_q = ceil((1-q) * E*R*E)), and/or cap_candidates.
             The final cap is min(cap_candidates, cap_q) if both are set.
          2. Compute joint log-probabilities for all (h, r, t) triples using:
                joint(h, r, t) = a_h * log P(r|h) + a_t_inv * log P(r^{-1}|t) + a_t_pred * log P(t|h)
                where a_h = (1-beta)*(1-alpha), a_t_inv = (1-beta)*alpha, a_t_pred = beta
             without materializing the full (E, R, E) tensor.
          3. Use a streaming top-k algorithm to keep only the highest-probability candidates globally.
          4. Stack all candidate triples and their scores.
          5. If a global probability threshold p is set, filter candidates by score >= p.
             Always keep at least one candidate to avoid empty sets downstream.

        Args:
            tuples: (num_samples, 2) tensor, entity in col 0.
            relation_maps: RelationMaps object mapping original to inverse relations.
            scores_rp: (num_samples, num_relations) tensor of per-sample relation logits.
            scores_tp: (num_samples, num_entities) tensor of per-sample tail logits (required for tail prediction).
            num_groups: int, number of groups for candidate cap.
            p: Optional[float], global probability threshold for candidate filtering.
            q: Optional[float] in [0,1), quantile threshold for global top-k (keeps top (1-q) fraction).
            temperature: float, softmax temperature for calibration.
            alpha: float in [0,1], weight for tail vs head.
            beta: float, weight for tail prediction probability.
            cap_candidates: Optional[int], hard cap on number of candidates.

        Returns:
            candidates: (N, 3) tensor of (head_id, relation_id, tail_id) triples.
            scores: (N,) tensor of joint probabilities for each candidate.
        """
        assert scores_rp is not None, "scores_rp required."
        assert scores_tp is not None, "scores_tp required for tail prediction."

        # Aggregate logits per unique head entity
        entities, scores_rp_grouped = self._aggregate_logits_per_head(tuples, scores_rp)
        _, scores_tp_grouped = self._aggregate_logits_per_head(tuples, scores_tp)  # Aggregate tail logits
        E = entities.size(0)

        # Restrict to known entities
        scores_tp_grouped = scores_tp_grouped[:, entities]

        device = scores_rp_grouped.device

        # 3. Resolve original & inverse relation ids
        original_relations = relation_maps.original_relations_tensor.to(device)  # (R,)
        inverse_relations  = relation_maps.inverse_relations_tensor.to(device)   # (R,)
        R = original_relations.size(0)

        # 4. Slice logits for original and inverse relation columns
        head_logits_subset = scores_rp_grouped[:, original_relations]   # (E, R)
        tail_logits_subset = scores_rp_grouped[:, inverse_relations]    # (E, R)

        # if self.phase1_loss_fn == "poisson":
        #     head_logits_subset = torch.exp(head_logits_subset)
        #     tail_logits_subset = torch.exp(tail_logits_subset)
        #     scores_tp_grouped = torch.exp(scores_tp_grouped)

        # 5. Calibrated log-probabilities or raw logits based on normalize_mode
        if self.normalize_mode == "per_head":
            log_p_head_2d = torch.log_softmax(head_logits_subset / self.temperature, dim=1).to(torch.float32).cpu()                 # (E, R)
            log_p_tail_2d = torch.log_softmax(tail_logits_subset / self.temperature, dim=1).to(torch.float32).cpu().transpose(0,1) # (R, E)
            log_p_t_given_h_2d = torch.log_softmax(scores_tp_grouped / self.temperature, dim=1).to(torch.float32).cpu()          # (E, E)
        elif self.normalize_mode == "global_joint":
            z_hr = (head_logits_subset / self.temperature).to(torch.float32).cpu()             # (E, R)
            log_p_head_2d = torch.log_softmax(z_hr.reshape(-1), dim=0).reshape(E, R)          # joint over (h,r)
            z_tr = (tail_logits_subset / self.temperature).to(torch.float32).cpu().transpose(0,1)  # (R, E)
            log_p_tail_2d = torch.log_softmax(z_tr.reshape(-1), dim=0).reshape(R, E)          # joint over (r,t)
            z_ht = (scores_tp_grouped / self.temperature).to(torch.float32).cpu()             # (E, E)
            log_p_t_given_h_2d = torch.log_softmax(z_ht.reshape(-1), dim=0).reshape(E, E)     # joint over (h,t)
        elif self.normalize_mode == "per_relation":
            log_p_head_2d = torch.log_softmax(head_logits_subset / self.temperature, dim=0).to(torch.float32).cpu()                 # (E, R)
            log_p_tail_2d = torch.log_softmax(tail_logits_subset / self.temperature, dim=0).to(torch.float32).cpu().transpose(0,1) # (R, E)
            log_p_t_given_h_2d = torch.log_softmax(scores_tp_grouped / self.temperature, dim=1).to(torch.float32).cpu()          # (E, E)
        else:  # "none"
            log_p_head_2d = (head_logits_subset / self.temperature).to(torch.float32).cpu()                 # (E, R)
            log_p_tail_2d = (tail_logits_subset / self.temperature).to(torch.float32).cpu().transpose(0,1) # (R, E)
            log_p_t_given_h_2d = (scores_tp_grouped / self.temperature).to(torch.float32).cpu()             # (E, E)

        # Derive effective cap from q first (before any threshold). This bounds the search space.
        total = int(E) * int(R) * int(E)
        # Compute base cap from group strategy
        effective_cap = num_groups * self.per_group_cap
        if self.q is not None:
            cap_q = max(1, int(math.ceil((1.0 - self.q) * total)))
            if effective_cap is not None and effective_cap < cap_q:
                logger.warning(f"effective_cap < cap_q from q-quantile. Using smaller effective_cap {effective_cap} instead of {cap_q}.")
            effective_cap = cap_q if effective_cap is None else min(effective_cap, cap_q)
        if effective_cap is None:
            raise ValueError("Candidate generation requires a cap (q or cap_candidates). Threshold-only (p) is unsafe for large graphs.")

        # Compute global top-k in a streaming fashion without O(E*R*E) memory
        # torch.set_num_threads(int(self.num_threads))

        top_log_vals, r_idx, h_idx, t_idx = self._global_topk_joint_streaming(
            log_p_head_2d=log_p_head_2d,
            log_p_tail_2d=log_p_tail_2d,
            log_p_t_given_h_2d=log_p_t_given_h_2d,
            k_total=int(effective_cap),
        )

        # Build candidate triples (global entity indexing)
        heads_tensor = entities[h_idx]
        rels_tensor  = original_relations[r_idx].cpu()
        tails_tensor = entities[t_idx]
        candidates = torch.stack([heads_tensor, rels_tensor, tails_tensor], dim=1)

        # Convert to probabilities for thresholding
        scores = torch.exp(top_log_vals)
        # 6. Apply global threshold p if provided
        if self.p is not None:
            keep_mask = scores >= self.p
            if keep_mask.any():
                candidates = candidates[keep_mask]
                scores = scores[keep_mask]
            else:
                # Keep at least the best entry to avoid empty sets downstream
                best = torch.argmax(scores)
                candidates = candidates[best:best+1]
                scores = scores[best:best+1]

        return candidates, scores

from . import triple_lib
import itertools
from .figures import create_coverage_vs_size_plot, create_heatmaps

def grid_search_candidates(candidate_generator: BaseCandidateGenerator, args, tr_tuples_all, tr_logits_all, tr_scores_tp_all, va_tuples_all, va_logits_all, va_scores_tp_all, te_tuples_all, te_logits_all, te_scores_tp_all, train_triples, val_triples, test_triples, train_set_t, valid_set_t, test_set_t):
    """
    Perform grid search over alpha, beta, temperature for CandidateGeneratorGlobalWithTail
    to maximize total coverage and average recall per group on the test set.
    Initializes the candidate generator once and manually changes alpha, beta, temperature.
    Assumes CandidateGeneratorGlobalWithTail is used.
    """
    # Define grid ranges (adjust as needed)
    steps = 11
    if steps == 1:
        alpha_values = [0.5]
        beta_values = [0.5]
    else:
        # Generate values from 0 to 1 inclusive with 'steps' number of points
        alpha_values = np.round(np.linspace(0, 1, steps), 10)
        beta_values = np.round(np.linspace(0, 1, steps), 10)
    temperature_values = [1.0]
    # temperature_values = [0.5, 1.0, 2.0]

    
    param_combinations = list(itertools.product(temperature_values, beta_values, alpha_values))
    print(f"Grid search over {len(param_combinations)} parameter combinations: {param_combinations[:5]}...{param_combinations[-5:]}.")
    
    best_total_cov = -float('inf')
    best_params_total = None
    best_per_group = -float('inf')
    best_params_per_group = None
    results = []
    
    # Compute num_groups for test (only needed for test)
    num_groups_test = len(torch.unique(test_triples[:, args.group_strategy], dim=0))
    
    # Initialize multiprocessing pool and other resources once
    # if hasattr(candidate_generator, 'rel_block_size'):
    #     candidate_generator._get_or_create_pool(min(args.num_workers, math.ceil(test_set_t.relation_maps.original_relations_tensor.size(0) / (candidate_generator.rel_block_size if candidate_generator.rel_block_size else 1))))
    candidate_generator._get_or_create_pool(args.num_workers)
    test_triples_group_ids = triple_lib.generate_group_id_function(test_triples, args.group_strategy)(test_triples)
    
    for temp, beta, alpha in tqdm(param_combinations, desc="Grid Search", unit="config", leave=False):
        # Manually change the parameters
        candidate_generator.alpha = alpha
        candidate_generator.beta = beta
        candidate_generator.temperature = temp
        
        # Generate candidates only for test set
        candidates_test, _ = candidate_generator.generate_candidates(te_tuples_all, te_logits_all, test_set_t.relation_maps, num_groups_test, scores_tp=te_scores_tp_all)
        
        # Compute total coverage on test
        total_cov, _ = BaseCandidateGenerator.analyze_total_coverage(candidates_test, test_triples, name=f"alpha={alpha}_beta={beta}_temp={temp}", print_results=False)
        
        # Compute per-group coverage on test (average recall per group)
        # Assuming analyze_coverage_per_group is modified to return the average recall
        avg_cov_per_group, avg_group_density, avg_group_count = candidate_generator.analyze_coverage_per_group(
            candidates_test, 
            triple_lib.generate_group_id_function(torch.cat([test_triples, candidates_test], dim=0), args.group_strategy)(candidates_test), 
            test_triples, 
            test_triples_group_ids, 
            test_set_t.relation_maps,
            name=f"alpha={alpha}_beta={beta}_temp={temp}",
            print_results=False
        )
        
        results.append({'candidate_size': candidates_test.size(0), 'avg_group_count': avg_group_count, 'total_cov': total_cov, 'avg_cov_per_group': avg_cov_per_group, 'alpha': alpha, 'beta': beta, 'temp': temp, 'avg_group_density': avg_group_density})
        
        if total_cov > best_total_cov:
            best_total_cov = total_cov
            best_params_total = (alpha, beta, temp)
        
        if avg_cov_per_group > best_per_group:
            best_per_group = avg_cov_per_group
            best_params_per_group = (alpha, beta, temp)
        # tqdm.write(f"Params: alpha={alpha}\tbeta={beta}\ttemp={temp}\t=> total_cov={total_cov:<4.4f}\tavg_recall_per_group={avg_recall_per_group:<4.4f}")

    print(f"Best params for total coverage: alpha={best_params_total[0]}, beta={best_params_total[1]}, temperature={best_params_total[2]}, total_cov={best_total_cov:<.4f}")
    print(f"Best params for per-group coverage: alpha={best_params_per_group[0]}, beta={best_params_per_group[1]}, temperature={best_params_per_group[2]}, avg_recall_per_group={best_per_group:<.4f}")
    print()
    
    # Create heatmaps
    create_heatmaps(results, save_dir=args.figure_dir)
    
    # Set best params to the generator
    candidate_generator.alpha = best_params_total[0]
    candidate_generator.beta = best_params_total[1]
    candidate_generator.temperature = best_params_total[2]
    print(f"Set candidate generator to best params for total coverage: alpha={candidate_generator.alpha}, beta={candidate_generator.beta}, temperature={candidate_generator.temperature}")
    
    return best_params_total, best_params_per_group

def grid_search_candidate_sizes(candidate_generator: BaseCandidateGenerator, args, tr_tuples_all, tr_logits_all, tr_scores_tp_all, va_tuples_all, va_logits_all, va_scores_tp_all, te_tuples_all, te_logits_all, te_scores_tp_all, train_triples, val_triples, test_triples, train_set_t, valid_set_t, test_set_t):
    """
    Perform grid search over per_group_cap (candidate sizes) for CandidateGeneratorGlobalWithTail
    to analyze total coverage and average recall per group on the test set.
    For global generators, generates candidates for the largest size and iteratively slices
    top-k for smaller sizes, using the previous smaller set for efficiency.
    """
    # Define candidate sizes (log-distributed)
    min_val = 1
    max_val = 1000
    total_count = 15  # Adjust as needed; matches approx. length of original list
    candidate_sizes = np.unique(np.logspace(np.log10(min_val), np.log10(max_val), num=total_count, dtype=int)).tolist()
    p = 5  # Power for stretching (higher p = more emphasis on small sizes)
    candidate_sizes = np.unique(np.round(np.linspace(min_val**(1/p), max_val**(1/p), num=total_count)**p).astype(int)).tolist()

    print(f"Grid search over {len(candidate_sizes)} candidate sizes: {candidate_sizes[:5]}...{candidate_sizes[-5:]}.")
    
    results = []
    
    # Compute num_groups for test
    num_groups_test = len(torch.unique(test_triples[:, args.group_strategy], dim=0))
    
    # Initialize multiprocessing pool and other resources once
    # if hasattr(candidate_generator, 'rel_block_size'):
    #     candidate_generator._get_or_create_pool(min(args.num_workers, math.ceil(test_set_t.relation_maps.original_relations_tensor.size(0) / (candidate_generator.rel_block_size if candidate_generator.rel_block_size else 1))))
    candidate_generator._get_or_create_pool(args.num_workers)
    test_triples_group_ids = triple_lib.generate_group_id_function(test_triples, args.group_strategy)(test_triples)
    
    # Optimization for global generators: generate once for max size, then iteratively slice for smaller sizes
    if isinstance(candidate_generator, (CandidateGeneratorGlobal, CandidateGeneratorGlobalWithTail)) and args.candidates_threshold_p is None and args.candidates_quantile_q is None:
        # Sort in descending order for iterative slicing from largest to smallest
        candidate_sizes = sorted(candidate_sizes, reverse=True)
        largest_size = max(candidate_sizes) if candidate_sizes else 0

        # Generate initial candidate set for the largest size
        tqdm.write(f"Generating initial candidate set for largest size: {largest_size}")
        # Manually set per_group_cap
        candidate_generator.per_group_cap = largest_size
        # The returned candidates from global generators are sorted by score.
        all_candidates, all_scores = candidate_generator.generate_candidates(te_tuples_all, te_logits_all, test_set_t.relation_maps, num_groups_test, te_scores_tp_all)
        
        # Ensure candidates are sorted by scores (descending) to guarantee order
        sorted_indices = torch.argsort(all_scores, descending=True)
        all_candidates = all_candidates[sorted_indices]
        all_scores = all_scores[sorted_indices]
        
        # Start with the full set for the largest size
        current_candidates = all_candidates.clone()
        current_scores = all_scores.clone()

        for size in tqdm(candidate_sizes, desc="Grid Search Sizes (iterative slicing)", unit="size", leave=False):
            # Compute effective cap for this size
            effective_cap = num_groups_test * size
            assert effective_cap <= current_candidates.size(0), "Effective cap should be less than or equal to available candidates."
            current_k = effective_cap
            
            # Slice the top-k from current candidates and scores (already sorted)
            candidates = current_candidates[:current_k]
            scores = current_scores[:current_k]

            # Update current_candidates and current_scores for next iteration (smaller size)
            current_candidates = candidates
            current_scores = scores
            
            # Compute group IDs for candidates
            candidates_group_ids = triple_lib.generate_group_id_function(candidates, args.group_strategy)(candidates)
            
            # Analyze total coverage
            total_cov, pos_density = BaseCandidateGenerator.analyze_total_coverage(candidates, test_triples, name=f"size={size}", print_results=False)
            
            # Analyze coverage per group
            avg_cov_per_group, avg_group_density, avg_group_count = candidate_generator.analyze_coverage_per_group(candidates, candidates_group_ids, test_triples, test_triples_group_ids, test_set_t.relation_maps, name=f"size={size}", print_results=False)
            
            results.append({'candidate_size': candidates.size(0), 'avg_group_count': avg_group_count, 'total_cov': total_cov, 'avg_cov_per_group': avg_cov_per_group, 'avg_group_density': avg_group_density, 'pos_density': pos_density})
    else:  # Original iterative approach for other generators like CandidateGeneratorPerHead
        for size in tqdm(candidate_sizes, desc="Grid Search Sizes", unit="size", leave=False):
            # Manually set per_group_cap
            candidate_generator.per_group_cap = size
            
            # Generate candidates for test set
            candidates, scores = candidate_generator.generate_candidates(te_tuples_all, te_logits_all, test_set_t.relation_maps, num_groups_test, te_scores_tp_all)
            
            # Ensure candidates are sorted by scores (descending) to guarantee order
            sorted_indices = torch.argsort(scores, descending=True)
            candidates = candidates[sorted_indices]
            scores = scores[sorted_indices]
            
            # Compute group IDs for candidates (assuming group by head for simplicity; adapt if needed)
            candidates_group_ids = triple_lib.generate_group_id_function(candidates, args.group_strategy)(candidates)
            
            # Analyze total coverage
            total_cov, pos_density = BaseCandidateGenerator.analyze_total_coverage(candidates, test_triples, name=f"size={size}", print_results=False)
            
            # Analyze coverage per group
            avg_cov_per_group, avg_group_density, avg_group_count = candidate_generator.analyze_coverage_per_group(candidates, candidates_group_ids, test_triples, test_triples_group_ids, test_set_t.relation_maps, name=f"size={size}", print_results=False)
            
            results.append({'candidate_size': candidates.size(0), 'avg_group_count': avg_group_count, 'total_cov': total_cov, 'avg_cov_per_group': avg_cov_per_group, 'avg_group_density': avg_group_density, 'pos_density': pos_density})
            # tqdm.write(f"Size {size}: total_cov={total_cov:.4f}, avg_coverage_per_group={per_group_cov:.4f}")
            
            if total_cov >= 0.95:
                tqdm.write(f"Reached total coverage {total_cov} with candidate size {size}. Stopping early.")
                break
    
    candidate_generator.per_group_cap = args.candidates_cap  # reset to original
    # Create coverage vs size plot
    create_coverage_vs_size_plot(results, save_dir=args.figure_dir)
    
    return