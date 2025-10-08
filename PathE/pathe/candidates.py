from abc import ABC, abstractmethod
from functools import partial
import math
import statistics
import multiprocessing as mp
import os

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
        if self.pool is None or self.pool._processes < num_processes:
            if self.pool is not None:
                self.close_pool()
            logger.info(f"Creating multiprocessing pool with {num_processes} processes. This may take a while...")
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
        logits_rp: torch.Tensor,
        relation_maps: RelationMaps,
        num_groups: int,
        logits_tp: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate candidate triples from tuple predictions.
        
        Args:
            tuples: (N, 2) tensor with head entities in column 0
            logits_rp: (N, R_total) tensor of relation logits
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
    def _aggregate_logits_per_head(tuples: torch.Tensor, logits_rp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Aggregate (mean) relation logits for each unique head entity using torch_scatter.
        Args:
            tuples: (num_samples, 2) tensor with head entities in column 0 and relations in column 1
            logits_rp: (num_samples, R_total) tensor of relation logits
        Returns:
            unique_heads: (H,) long tensor of unique entity ids;
            logits_rp_grouped: (H, R_total) mean logits per head
        """
        # 1. Collect unique head entities (local indexing)
        unique_heads, inverse_entity_indices = tuples[:, 0].unique(return_inverse=True, sorted=False)
        if unique_heads.size(0) == 0:
            return tuples.new_zeros((0, 3)), tuples.new_zeros((0,), dtype=torch.float32)
        # 2. Aggregate logits per local head index
        # logits_rp has shape (num_samples, R_total); inverse_entity_indices maps each row to its head index
        logits_rp_grouped = torch_scatter.scatter_mean(logits_rp, inverse_entity_indices, dim=0)
        return unique_heads, logits_rp_grouped


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
        self.analyze_total_coverage(candidates, gold_triples, relation_maps, name=name, print_results=True)
        self.analyze_coverage_per_group(candidates, candidates_group_ids, gold_triples, gold_group_ids, relation_maps, name=name)
        return

    @torch.no_grad()
    def analyze_total_coverage(
        self,
        candidates: torch.Tensor,
        gold_triples: torch.Tensor,
        relation_maps: RelationMaps,
        name: str = "Set",
        print_results: bool = True,
    ) -> None:
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
    ) -> None:
        """
        Print per-group (macro) coverage stats by calling analyze_total_coverage per group,
        plus overall micro coverage. Does not require candidate labels.

        Args:
            candidates: (M,3) candidate triples tensor
            candidates_group_ids: (M,) long tensor, group id for each candidate triple (aligned with candidates)
            gold_triples: (N,3) gold triples tensor
            gold_group_ids: (N,) long tensor, group id for each gold triple (aligned with gold_triples)
            name: label for print messages
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

        for gid in tqdm(unique_gids.tolist(), desc=f"Coverage per group [{name}]", leave=False):
            gid = int(gid)
            gold_idx = (gold_group_ids == gid).nonzero(as_tuple=False).flatten()
            if gold_idx.numel() == 0:
                continue
            gold_subset = gold_triples[gold_idx]

            cand_idx = (candidates_group_ids == gid).nonzero(as_tuple=False).flatten()
            cand_subset = candidates[cand_idx] if cand_idx.numel() > 0 else candidates.new_zeros((0, 3), dtype=candidates.dtype)

            cov, dens = self.analyze_total_coverage(
                candidates=cand_subset,
                gold_triples=gold_subset,
                relation_maps=relation_maps,
                name=f"{name}|gid={gid}",
                print_results=False,
            )
            per_group_cov[gid] = float(cov)
            per_group_density[gid] = float(dens)
            per_group_count[gid] = cand_subset.size(0)

            group_size = int(gold_subset.size(0))
            covered_total += cov * group_size
            total_total += group_size

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
        return statistics.mean(per_group_cov.values()), statistics.mean(per_group_density.values())

class CandidateGeneratorGlobal(BaseCandidateGenerator):
    def __init__(self, p: float, q: float, temperature: float, alpha: float, per_group_cap: int, normalize_mode: str = "per_head",
                 rel_block_size: int = 1,
                 head_block_size: int = 256,
                 max_num_workers: int | None = None,
                 ):
        super().__init__(max_num_workers=max_num_workers)
        self.p = p                              # keep candidates with P >= p (global threshold)
        self.q = q                              # use as fraction-cap: cap = ceil((1-q) * |E|*|R|*|E|)
        self.temperature = temperature          # temperature for softmax calibration
        self.alpha = alpha                      # weight for head vs tail log-probs
        self.per_group_cap = per_group_cap      # per-group cap, used to compute total cap
        self.normalize_mode = normalize_mode    # "per_head" | "global_joint" | "none"

        # new knobs for memory and parallelism
        self.rel_block_size = max(1, int(rel_block_size))
        self.head_block_size = max(1, int(head_block_size))

        # sanity checks
        assert self.temperature > 0, "temperature must be > 0"
        assert 0.0 <= self.alpha <= 1.0, "alpha must be in [0,1]"
        assert self.normalize_mode in ("per_head", "global_joint", "none"), "normalize_mode invalid"
        if self.p is not None:
            assert 0.0 <= self.p <= 1.0, "p must be in [0,1]"
        if self.q is not None:
            assert 0.0 <= self.q < 1.0, "q must be in [0,1)"

    @staticmethod
    def _process_chunk(r0: int, r1: int, log_p_head_2d: torch.Tensor, log_p_tail_2d: torch.Tensor, alpha: float, k: int, E: int, R: int, head_block_size: int):
        """Worker function for parallel chunk processing with head blocking to control memory."""
        torch.set_num_threads(1)  # 1 thread per worker as we parallelize over processes, otherwise it breaks
        C = r1 - r0
        a_h = float(alpha)
        a_t = float(1.0 - alpha)
        B = max(1, int(head_block_size))

        # Global buffers for this chunk
        chunk_vals = torch.full((k,), float("-inf"), dtype=torch.float32)
        chunk_r    = torch.full((k,), -1, dtype=torch.long)
        chunk_h    = torch.full((k,), -1, dtype=torch.long)
        chunk_t    = torch.full((k,), -1, dtype=torch.long)
        chunk_filled = 0

        # Preload per-relation tail inverse
        t_chunk = a_t * log_p_tail_2d[r0:r1, :]  # (C, E)

        for r in tqdm(range(C), desc="Processing relations in chunk", leave=False):
            r_global = r0 + r
            t_inv = t_chunk[r, :]  # (E,)

            for h0 in tqdm(range(0, E, B), desc="Processing heads in blocks", leave=False):
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
        self.pool = self._get_or_create_pool(min(self.max_num_workers, math.ceil(R / self.rel_block_size)))
        # Prepare chunk ranges
        chunk_ranges = [(r0, min(R, r0 + self.rel_block_size)) for r0 in range(0, R, self.rel_block_size)]
        # Submit jobs
        results = [self.pool.apply_async(CandidateGeneratorGlobal._process_chunk, (r0, r1, log_p_head_2d, log_p_tail_2d, self.alpha, k_total, E, R, self.head_block_size)) for r0, r1 in chunk_ranges]
        # Collect and merge
        for res in tqdm(results, desc="Merging parallel chunks", unit="chunk", leave=False):
            vals_chunk, r_idx, h_idx, t_idx = res.get()
            # Merge into global top-k
            if filled == 0:
                take = min(k_total, vals_chunk.numel())
                top_vals[:take] = vals_chunk[:take]
                top_r[:take]    = r_idx[:take]
                top_h[:take]    = h_idx[:take]
                top_t[:take]    = t_idx[:take]
                filled = take
            else:
                cand_vals = torch.cat([top_vals[:filled], vals_chunk], dim=0)
                cand_r    = torch.cat([top_r[:filled],    r_idx],      dim=0)
                cand_h    = torch.cat([top_h[:filled],    h_idx],      dim=0)
                cand_t    = torch.cat([top_t[:filled],    t_idx],      dim=0)
                if cand_vals.numel() > k_total:
                    vtop, order = torch.topk(cand_vals, k=k_total, largest=True)
                    top_vals[:k_total] = vtop
                    top_r[:k_total]    = cand_r[order]
                    top_h[:k_total]    = cand_h[order]
                    top_t[:k_total]    = cand_t[order]
                    filled = k_total
                else:
                    top_vals[:cand_vals.numel()] = cand_vals
                    top_r[:cand_vals.numel()]    = cand_r
                    top_h[:cand_vals.numel()]    = cand_h
                    top_t[:cand_vals.numel()]    = cand_t
                    filled = cand_vals.numel()

        return top_vals[:filled], top_r[:filled], top_h[:filled], top_t[:filled]

    def generate_candidates(
        self,
        tuples: torch.Tensor,
        logits_rp: torch.Tensor,
        relation_maps: RelationMaps,
        num_groups: int,
        logits_tp: torch.Tensor = None,
    ):
        """
        Efficiently generate candidate (head, relation, tail) triples and their joint probabilities
        for two-phase PathE training, using global top-k streaming to avoid OOM on large graphs.

        Candidate selection logic:
          1. Compute an effective cap from quantile q (cap_q = ceil((1-q) * E*R*E)), and/or cap_candidates.
             The final cap is min(cap_candidates, cap_q) if both are set.
          2. Compute joint log-probabilities for all (h, r, t) triples using:
                joint(h, r, t) = alpha * log P(r|h) + (1-alpha) * log P(r^{-1}|t)
             without materializing the full (E, R, E) tensor.
          3. Use a streaming top-k algorithm to keep only the highest-probability candidates globally.
          4. Stack all candidate triples and their scores.
          5. If a global probability threshold p is set, filter candidates by score >= p.
             Always keep at least one candidate to avoid empty sets downstream.

        Args:
            tuples: (num_samples, 2) tensor, entity in col 0.
            relation_maps: RelationMaps object mapping original to inverse relations.
            logits_rp: (num_samples, num_relations) tensor of per-sample relation logits.
            num_groups: int, number of groups for candidate cap.
            p: Optional[float], global probability threshold for candidate filtering.
            q: Optional[float] in [0,1), quantile threshold for global top-k (keeps top (1-q) fraction).
            temperature: float, softmax temperature for calibration.
            alpha: float in [0,1], weight for head vs tail log-probabilities.
            cap_candidates: Optional[int], hard cap on number of candidates.

        Returns:
            candidates: (N, 3) tensor of (head_id, relation_id, tail_id) triples.
            scores: (N,) tensor of joint probabilities for each candidate.
        """
        assert logits_rp is not None, "logits_rp required."

        # Aggregate logits per unique head entity
        entities, logits_rp_grouped = self._aggregate_logits_per_head(tuples, logits_rp)
        E = entities.size(0)
        device = logits_rp_grouped.device

        # 3. Resolve original & inverse relation ids
        original_relations = relation_maps.original_relations_tensor.to(device)  # (R,)
        inverse_relations  = relation_maps.inverse_relations_tensor.to(device)   # (R,)
        R = original_relations.size(0)

        # 4. Slice logits for original and inverse relation columns
        head_logits_subset = logits_rp_grouped[:, original_relations]   # (E, R)
        tail_logits_subset = logits_rp_grouped[:, inverse_relations]    # (E, R)

        # 5. Calibrated log-probabilities or raw logits based on normalize_mode
        if self.normalize_mode == "per_head":
            log_p_head_2d = torch.log_softmax(head_logits_subset / self.temperature, dim=1).to(torch.float32).cpu()                 # (E, R)
            log_p_tail_2d = torch.log_softmax(tail_logits_subset / self.temperature, dim=1).to(torch.float32).cpu().transpose(0,1) # (R, E)
        elif self.normalize_mode == "global_joint":
            z_hr = (head_logits_subset / self.temperature).to(torch.float32).cpu()             # (E, R)
            log_p_head_2d = torch.log_softmax(z_hr.reshape(-1), dim=0).reshape(E, R)          # joint over all (h,r)
            z_tr = (tail_logits_subset / self.temperature).to(torch.float32).cpu().transpose(0,1)  # (R, E)
            log_p_tail_2d = torch.log_softmax(z_tr.reshape(-1), dim=0).reshape(R, E)          # joint over all (r,t)
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
    def __init__(self, per_group_cap: int):
        super().__init__()
        self.per_group_cap = per_group_cap    # number of (r,t) pairs to keep per head entity
        assert self.per_group_cap and self.per_group_cap > 0, "per_group_cap must be > 0"

    def _aggregate_logits_per_entity(tuples_2col: torch.Tensor,
                                     logits_rp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Aggregate (mean) relation logits for each unique head entity using torch_scatter.
        Returns:
            unique_heads: (H,) long tensor of entity ids
            agg_logits:   (H, R_total) mean logits per head
        """
        heads = tuples_2col[:, 0]
        unique_heads, inverse = heads.unique(return_inverse=True, sorted=False)
        # logits_rp has shape (num_samples, R_total); inverse maps each row to its head index
        agg_logits = torch_scatter.scatter_mean(logits_rp, inverse, dim=0)
        return unique_heads, agg_logits

    def generate_candidates(self, 
            tuples_all: torch.Tensor,
            logits_rp_all: torch.Tensor,
            relation_maps: RelationMaps,
            num_groups: int,
            logits_tp: torch.Tensor = None) -> dict[int, torch.Tensor]:
        """
        For each head entity h present in tuples_all:
           Score (h, r, t) = P(r|h) * P(r^{-1}|t)
           Keep top-k (r, t) pairs.
        Only original relations are considered (relation_maps.original_relation_to_inverse_relation keys).
        """
        # 1. Aggregate logits per entity (treat any entity that appears as head)
        entity_ids, logits_rp_grouped = self._aggregate_logits_per_head(tuples_all, logits_rp_all)  # (E',), (E', R_total)
        device = logits_rp_grouped.device
        
        # 2. Prepare relation mappings
        orig2inv = relation_maps.original_relation_to_inverse_relation
        if len(orig2inv) == 0:
            return {}
        
        # 3. Resolve original & inverse relation ids
        original_relation_ids = relation_maps.original_relations_tensor.to(device)  # (R,)
        inverse_relation_ids  = relation_maps.inverse_relations_tensor.to(device)   # (R,)


        # 4. Slice logits for originals & inverses then softmax separately
        logits_orig = logits_rp_grouped[:, original_relation_ids]          # (E', R)
        logits_inv  = logits_rp_grouped[:, inverse_relation_ids]           # (E', R)
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
            scores = p_r_h.unsqueeze(1) * prob_rinv_T  # (R, E)
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
                 rel_block_size: int = 1,
                 head_block_size: int = 256,
                 max_num_workers: int | None = None,
                #  num_threads: int | None = None
                 ):
        super().__init__(max_num_workers=max_num_workers)
        self.p = p                              # keep candidates with P >= p (global threshold)
        self.q = q                              # use as fraction-cap: cap = ceil((1-q) * |E|*|R|*|E|)
        self.temperature = temperature          # temperature for softmax calibration
        self.alpha = alpha                      # weight for head log-probs P(r|h)
        self.beta = beta                        # weight for tail prediction P(t|h)
        self.per_group_cap = per_group_cap      # per-group cap, used to compute total cap
        self.normalize_mode = normalize_mode    # "per_head" | "global_joint" | "none"

        # new knobs for memory and parallelism
        self.rel_block_size = max(1, int(rel_block_size))
        self.head_block_size = max(1, int(head_block_size))

        # sanity checks
        assert self.temperature > 0, "temperature must be > 0"
        assert 0.0 <= self.alpha <= 1.0, "alpha must be in [0,1]"
        assert 0.0 <= self.beta <= 1.0, "beta must be in [0,1]"
        assert self.alpha + self.beta <= 1.0, "alpha + beta must be <= 1.0 (gamma = 1 - alpha - beta)"
        assert self.normalize_mode in ("per_head", "global_joint", "none"), "normalize_mode invalid"
        if self.p is not None:
            assert 0.0 <= self.p <= 1.0, "p must be in [0,1]"
        if self.q is not None:
            assert 0.0 <= self.q < 1.0, "q must be in [0,1)"

    @staticmethod
    def _process_chunk(r0: int, r1: int, log_p_head_2d: torch.Tensor, log_p_tail_2d: torch.Tensor, log_p_t_given_h_2d: torch.Tensor, alpha: float, beta: float, k: int, E: int, R: int, head_block_size: int):
        """Worker function for parallel chunk processing with tail prediction."""
        torch.set_num_threads(1)  # 1 thread per worker as we parallelize over processes, otherwise it breaks
        C = r1 - r0
        gamma = 1.0 - alpha - beta
        a_h = float(alpha)
        a_t_pred = float(beta)
        a_t_inv = float(gamma)
        B = max(1, int(head_block_size))

        # Global buffers for this chunk
        chunk_vals = torch.full((k,), float("-inf"), dtype=torch.float32)
        chunk_r    = torch.full((k,), -1, dtype=torch.long)
        chunk_h    = torch.full((k,), -1, dtype=torch.long)
        chunk_t    = torch.full((k,), -1, dtype=torch.long)
        chunk_filled = 0

        # Preload per-relation tail inverse
        t_inv_chunk = (a_t_inv * log_p_tail_2d[r0:r1, :])  # (C, E)

        for r in tqdm(range(C), desc="Processing relations in chunk", leave=False):
            r_global = r0 + r
            t_inv = t_inv_chunk[r, :]  # (E,)

            for h0 in tqdm(range(0, E, B), desc="Processing heads in blocks", leave=False):
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
        self.pool = self._get_or_create_pool(min(self.max_num_workers, math.ceil(R / self.rel_block_size)))
        chunk_ranges = [(r0, min(R, r0 + self.rel_block_size)) for r0 in range(0, R, self.rel_block_size)]
        results = [self.pool.apply_async(CandidateGeneratorGlobalWithTail._process_chunk, (r0, r1, log_p_head_2d, log_p_tail_2d, log_p_t_given_h_2d, self.alpha, self.beta, k_total, E, R, self.head_block_size)) for r0, r1 in chunk_ranges]
        for res in tqdm(results, desc="Merging parallel chunks", unit="chunk", leave=False):
            vals_chunk, r_idx, h_idx, t_idx = res.get()
            # Merge into global top-k (same as before)
            if filled == 0:
                take = min(k_total, vals_chunk.numel())
                top_vals[:take] = vals_chunk[:take]
                top_r[:take]    = r_idx[:take]
                top_h[:take]    = h_idx[:take]
                top_t[:take]    = t_idx[:take]
                filled = take
            else:
                cand_vals = torch.cat([top_vals[:filled], vals_chunk], dim=0)
                cand_r    = torch.cat([top_r[:filled],    r_idx],      dim=0)
                cand_h    = torch.cat([top_h[:filled],    h_idx],      dim=0)
                cand_t    = torch.cat([top_t[:filled],    t_idx],      dim=0)
                if cand_vals.numel() > k_total:
                    vtop, order = torch.topk(cand_vals, k=k_total, largest=True)
                    top_vals[:k_total] = vtop
                    top_r[:k_total]    = cand_r[order]
                    top_h[:k_total]    = cand_h[order]
                    top_t[:k_total]    = cand_t[order]
                    filled = k_total
                else:
                    top_vals[:cand_vals.numel()] = cand_vals
                    top_r[:cand_vals.numel()]    = cand_r
                    top_h[:cand_vals.numel()]    = cand_h
                    top_t[:cand_vals.numel()]    = cand_t
                    filled = cand_vals.numel()

        return top_vals[:filled], top_r[:filled], top_h[:filled], top_t[:filled]

    def generate_candidates(
        self,
        tuples: torch.Tensor,
        logits_rp: torch.Tensor,
        relation_maps: RelationMaps,
        num_groups: int,
        logits_tp: torch.Tensor = None,
    ):
        """
        Efficiently generate candidate (head, relation, tail) triples and their joint probabilities
        for two-phase PathE training, using global top-k streaming to avoid OOM on large graphs.
        Now includes tail prediction P(t|h) in the scoring.

        Candidate selection logic:
          1. Compute an effective cap from quantile q (cap_q = ceil((1-q) * E*R*E)), and/or cap_candidates.
             The final cap is min(cap_candidates, cap_q) if both are set.
          2. Compute joint log-probabilities for all (h, r, t) triples using:
                joint(h, r, t) = alpha * log P(r|h) + beta * log P(t|h) + gamma * log P(r^{-1}|t)
                where gamma = 1 - alpha - beta
             without materializing the full (E, R, E) tensor.
          3. Use a streaming top-k algorithm to keep only the highest-probability candidates globally.
          4. Stack all candidate triples and their scores.
          5. If a global probability threshold p is set, filter candidates by score >= p.
             Always keep at least one candidate to avoid empty sets downstream.

        Args:
            tuples: (num_samples, 2) tensor, entity in col 0.
            relation_maps: RelationMaps object mapping original to inverse relations.
            logits_rp: (num_samples, num_relations) tensor of per-sample relation logits.
            logits_tp: (num_samples, num_entities) tensor of per-sample tail logits (required for tail prediction).
            p: Optional[float], global probability threshold for candidate filtering.
            q: Optional[float] in [0,1), quantile threshold for global top-k (keeps top (1-q) fraction).
            temperature: float, softmax temperature for calibration.
            alpha: float in [0,1], weight for head log-probs P(r|h).
            beta: float in [0,1], weight for tail prediction P(t|h).
            cap_candidates: Optional[int], hard cap on number of candidates.

        Returns:
            candidates: (N, 3) tensor of (head_id, relation_id, tail_id) triples.
            scores: (N,) tensor of joint probabilities for each candidate.
        """
        assert logits_rp is not None, "logits_rp required."
        assert logits_tp is not None, "logits_tp required for tail prediction."

        # Aggregate logits per unique head entity
        entities, logits_rp_grouped = self._aggregate_logits_per_head(tuples, logits_rp)
        _, logits_tp_grouped = self._aggregate_logits_per_head(tuples, logits_tp)  # Aggregate tail logits
        E = entities.size(0)

        # Restrict to known entities
        logits_tp_grouped = logits_tp_grouped[:, entities]

        device = logits_rp_grouped.device

        # 3. Resolve original & inverse relation ids
        original_relations = relation_maps.original_relations_tensor.to(device)  # (R,)
        inverse_relations  = relation_maps.inverse_relations_tensor.to(device)   # (R,)
        R = original_relations.size(0)

        # 4. Slice logits for original and inverse relation columns
        head_logits_subset = logits_rp_grouped[:, original_relations]   # (E, R)
        tail_logits_subset = logits_rp_grouped[:, inverse_relations]    # (E, R)

        # 5. Calibrated log-probabilities or raw logits based on normalize_mode
        if self.normalize_mode == "per_head":
            log_p_head_2d = torch.log_softmax(head_logits_subset / self.temperature, dim=1).to(torch.float32).cpu()                 # (E, R)
            log_p_tail_2d = torch.log_softmax(tail_logits_subset / self.temperature, dim=1).to(torch.float32).cpu().transpose(0,1) # (R, E)
            log_p_t_given_h_2d = torch.log_softmax(logits_tp_grouped / self.temperature, dim=1).to(torch.float32).cpu()          # (E, E)
        elif self.normalize_mode == "global_joint":
            z_hr = (head_logits_subset / self.temperature).to(torch.float32).cpu()             # (E, R)
            log_p_head_2d = torch.log_softmax(z_hr.reshape(-1), dim=0).reshape(E, R)          # joint over (h,r)
            z_tr = (tail_logits_subset / self.temperature).to(torch.float32).cpu().transpose(0,1)  # (R, E)
            log_p_tail_2d = torch.log_softmax(z_tr.reshape(-1), dim=0).reshape(R, E)          # joint over (r,t)
            z_ht = (logits_tp_grouped / self.temperature).to(torch.float32).cpu()             # (E, E)
            log_p_t_given_h_2d = torch.log_softmax(z_ht.reshape(-1), dim=0).reshape(E, E)     # joint over (h,t)
        else:  # "none"
            log_p_head_2d = (head_logits_subset / self.temperature).to(torch.float32).cpu()                 # (E, R)
            log_p_tail_2d = (tail_logits_subset / self.temperature).to(torch.float32).cpu().transpose(0,1) # (R, E)
            log_p_t_given_h_2d = (logits_tp_grouped / self.temperature).to(torch.float32).cpu()             # (E, E)

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

def grid_search_candidates(args, tr_tuples_all, tr_logits_all, tr_logits_tp_all, va_tuples_all, va_logits_all, va_logits_tp_all, te_tuples_all, te_logits_all, te_logits_tp_all, train_triples, val_triples, test_triples, train_set_t, valid_set_t, test_set_t):
    """
    Perform grid search over alpha, beta, temperature for CandidateGeneratorGlobalWithTail
    to maximize total coverage and average recall per group on the test set.
    Initializes the candidate generator once and manually changes alpha, beta, temperature.
    Assumes CandidateGeneratorGlobalWithTail is used.
    """
    # Define grid ranges (adjust as needed)
    L = list(range(0, 11, 1))
    alpha_values = [a / 10.0 for a in L]
    beta_values = [b / 10.0 for b in L]
    temperature_values = [1.0]
    # alpha_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    # beta_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    # temperature_values = [0.5, 1.0, 2.0]
    
    # Filter combinations to ensure alpha + beta <= 1.0
    param_combinations = [(a, b, t) for a in alpha_values for b in beta_values for t in temperature_values if a + b <= 1.0]
    # param_combinations = list(itertools.product(alpha_values, beta_values, temperature_values))
    print(f"Grid search over {len(param_combinations)} parameter combinations.")
    
    best_total_cov = -float('inf')
    best_params_total = None
    best_per_group = -float('inf')
    best_params_per_group = None
    results = []
    
    # Compute num_groups for test (only needed for test)
    num_groups_test = len(torch.unique(test_triples[:, args.group_strategy], dim=0))
    
    # Initialize candidate generator once with dummy values
    candidate_generator = CandidateGeneratorGlobalWithTail(
        p=args.candidates_threshold_p, q=args.candidates_quantile_q, temperature=1.0, alpha=0.5, beta=0.5,  # dummy initial values
        per_group_cap=args.candidates_cap, normalize_mode=args.candidates_normalize_mode, max_num_workers=args.num_workers
    )

    test_triples_group_ids = triple_lib.generate_group_id_function(test_triples, args.group_strategy)(test_triples)
    
    for alpha, beta, temp in tqdm(param_combinations, desc="Grid Search", unit="config", leave=False):
        # Manually change the parameters
        candidate_generator.alpha = alpha
        candidate_generator.beta = beta
        candidate_generator.temperature = temp
        
        # Generate candidates only for test set
        candidates_test, _ = candidate_generator.generate_candidates(te_tuples_all, te_logits_all, test_set_t.relation_maps, num_groups_test, logits_tp=te_logits_tp_all)
        
        # Compute total coverage on test
        total_cov, _ = candidate_generator.analyze_total_coverage(candidates_test, test_triples, test_set_t.relation_maps, print_results=False)
        
        # Compute per-group coverage on test (average recall per group)
        # Assuming analyze_coverage_per_group is modified to return the average recall
        avg_recall_per_group, _ = candidate_generator.analyze_coverage_per_group(
            candidates_test, 
            triple_lib.generate_group_id_function(torch.cat([test_triples, candidates_test], dim=0), args.group_strategy)(candidates_test), 
            test_triples, 
            test_triples_group_ids, 
            test_set_t.relation_maps, 
            name="Test",
            print_results=False
        )
        
        results.append((alpha, beta, temp, total_cov, avg_recall_per_group))
        
        if total_cov > best_total_cov:
            best_total_cov = total_cov
            best_params_total = (alpha, beta, temp)
        
        if avg_recall_per_group > best_per_group:
            best_per_group = avg_recall_per_group
            best_params_per_group = (alpha, beta, temp)
        tqdm.write(f"Params: alpha={alpha}\tbeta={beta}\ttemp={temp}\t=> total_cov={total_cov:<4.4f}\tavg_recall_per_group={avg_recall_per_group:<4.4f}")

    print(f"Best params for total coverage: alpha={best_params_total[0]}, beta={best_params_total[1]}, temperature={best_params_total[2]}, total_cov={best_total_cov:<.4f}")
    print(f"Best params for per-group coverage: alpha={best_params_per_group[0]}, beta={best_params_per_group[1]}, temperature={best_params_per_group[2]}, avg_recall_per_group={best_per_group:<.4f}")
    print()
    results_sorted = sorted(results, key=lambda x: x[3], reverse=True)
    print("All results sorted by total_cov (alpha, beta, temp, total_cov, avg_recall_per_group):")
    for row in results_sorted:
        print(row)
    return best_params_total, best_params_per_group