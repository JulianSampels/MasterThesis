


from abc import ABC, abstractmethod
import math
import statistics

import torch
import torch_scatter
from tqdm import tqdm
from .pathdata import RelationMaps

import logging
logger = logging.getLogger(__name__)

class BaseCandidateGenerator(ABC):
    """Abstract base class for all candidate generation strategies."""
    
    def __init__(self):
        return

    @abstractmethod
    def generate_candidates(self,
        tuples: torch.Tensor,
        logits_rp: torch.Tensor,
        relation_maps: RelationMaps,
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
        return

class CandidateGeneratorGlobal(BaseCandidateGenerator):
    def __init__(self, p: float, q: float, temperature: float, alpha: float, cap_candidates: int):
        super().__init__()
        self.p = p                              # keep candidates with P >= p (global threshold)
        self.q = q                              # use as fraction-cap: cap = ceil((1-q) * |E|*|R|*|E|)
        self.temperature = temperature          # temperature for softmax calibration
        self.alpha = alpha                      # weight for head vs tail log-probs
        self.cap_candidates = cap_candidates    # final cap after thresholding only keep top-k candidates

        # sanity checks
        assert self.temperature > 0, "temperature must be > 0"
        assert 0.0 <= self.alpha <= 1.0, "alpha must be in [0,1]"
        if self.p is not None:
            assert 0.0 <= self.p <= 1.0, "p must be in [0,1]"
        if self.q is not None:
            assert 0.0 <= self.q < 1.0, "q must be in [0,1)"

    @staticmethod
    def _global_topk_joint_streaming(
        log_p_head_2d: torch.Tensor,
        log_p_tail_2d: torch.Tensor,
        alpha: float,
        k: int,
        rel_block_size: int = 1,
    ):
        """
        Memory-efficient global top-k over all (head, relation, tail) triples.

        Vectorized over relation chunks:
        - Process at most `rel_block_size` relations at a time (no tail-splitting).
        - For a chunk r in [r0:r1), build a joint score tensor S with shape (C, E, E),
          where C = r1 - r0 and:
              S[c, h, t] = alpha * log_p_head_2d[h, r0+c] + (1 - alpha) * log_p_tail_2d[r0+c, t]
        - Run a single topk over the whole chunk (flattened), then decode to (r, h, t).
        - Merge the chunk top-k into a running global top-k buffer of size k.

        Args:
            log_p_head_2d: Tensor (E, R) with log P(r | h)
            log_p_tail_2d: Tensor (R, E) with log P(r^{-1} | t), aligned by relation index
            alpha: Weight in [0,1] for head vs tail terms
            k: Number of top entries to keep globally
            rel_block_size: Max number of relations to process per chunk

        Returns:
            top_vals: (k',) tensor of joint log-probs (k' <= k)
            top_r:    (k',) tensor of relation indices
            top_h:    (k',) tensor of head indices
            top_t:    (k',) tensor of tail indices
        """
        assert 0.0 <= alpha <= 1.0, "alpha must be in [0,1]"
        E, R = log_p_head_2d.shape
        assert log_p_tail_2d.shape == (R, E), "log_p_tail_2d must be (R, E)"

        # Work on CPU float32 to minimize memory pressure
        log_p_head_2d = log_p_head_2d.to(dtype=torch.float32, device="cpu", copy=False)  # (E, R)
        log_p_tail_2d = log_p_tail_2d.to(dtype=torch.float32, device="cpu", copy=False)  # (R, E)

        # Running global top-k buffers (pre-allocated, updated incrementally)
        top_vals = torch.full((k,), float("-inf"), dtype=torch.float32)
        top_r    = torch.full((k,), -1, dtype=torch.long)
        top_h    = torch.full((k,), -1, dtype=torch.long)
        top_t    = torch.full((k,), -1, dtype=torch.long)
        filled = 0  # number of valid entries currently stored in the buffers

        a_h = float(alpha)
        a_t = float(1.0 - alpha)

        # Progress bar disappears after loop (leave=False)
        for r0 in tqdm(range(0, R, max(1, int(rel_block_size))),
                       desc=f"Computing global top-k candidates.", unit=f"{rel_block_size} relations", leave=False):
            r1 = min(R, r0 + max(1, int(rel_block_size)))
            C = r1 - r0  # number of relations in this chunk

            # Vectorized joint scores for the relation chunk
            # h_chunk: (E, C), t_chunk: (C, E)
            h_chunk = (a_h * log_p_head_2d[:, r0:r1])         # (E, C)
            t_chunk = (a_t * log_p_tail_2d[r0:r1, :])         # (C, E)
            # S: (C, E, E) with broadcasting: per relation c, S[c] = h_chunk[:, c][:, None] + t_chunk[c][None, :]
            S = h_chunk.t().unsqueeze(2) + t_chunk.unsqueeze(1)  # (C, E, E)
            # S = h_chunk.t().unsqueeze(2) * t_chunk.unsqueeze(1)  # (C, E, E)

            # Single top-k for the whole chunk
            numel_chunk = S.numel()
            if numel_chunk == 0:
                continue
            k_chunk = min(k, numel_chunk)
            vals_chunk, idx_chunk_flat = torch.topk(S.reshape(-1), k=k_chunk, largest=True)

            # Decode flat indices to (c, h, t) and map to global r = r0 + c
            per_rel = E * E
            c_idx = idx_chunk_flat // per_rel
            rem   = idx_chunk_flat % per_rel
            h_idx = rem // E
            t_idx = rem %  E
            r_idx = (c_idx + r0).to(torch.long)

            # Merge with running global top-k heap
            # General idea:
            # - Maintain the best 'k' triples seen so far across processed relation chunks.
            # - Concatenate current global list with this chunk's list, then take top-k again.
            # - We never allocate or sort the full (E*R*E) array; memory stays bounded by
            #   O(C*E*E) for the current chunk plus O(k) for the heap.
            if filled == 0:
                take = min(k, vals_chunk.numel())
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

                if cand_vals.numel() > k:
                    vtop, order = torch.topk(cand_vals, k=k, largest=True)
                    top_vals[:k] = vtop
                    top_r[:k]    = cand_r[order]
                    top_h[:k]    = cand_h[order]
                    top_t[:k]    = cand_t[order]
                    filled = k
                else:
                    top_vals[:cand_vals.numel()] = cand_vals
                    top_r[:cand_vals.numel()]    = cand_r
                    top_h[:cand_vals.numel()]    = cand_h
                    top_t[:cand_vals.numel()]    = cand_t
                    filled = cand_vals.numel()

        # Trim to filled size
        return top_vals[:filled], top_r[:filled], top_h[:filled], top_t[:filled]

    def generate_candidates(
        self,
        tuples: torch.Tensor,
        logits_rp: torch.Tensor,
        relation_maps: RelationMaps,
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

        # 5. Calibrated log-probabilities (avoid tiny exp, use log_softmax); keep as 2D on CPU
        log_p_head_2d = torch.log_softmax(head_logits_subset / self.temperature, dim=1).to(torch.float32).cpu()  # (E, R)
        # transpose to (R, E) to index by relation first on tail-side
        log_p_tail_2d = torch.log_softmax(tail_logits_subset / self.temperature, dim=1).to(torch.float32).cpu().transpose(0, 1)  # (R, E)
        # log_p_head_2d = (head_logits_subset / self.temperature).cpu()  # (E, R)
        # log_p_tail_2d = (tail_logits_subset / self.temperature).cpu().transpose(0, 1)  # (R, E)

        # Derive effective cap from q first (before any threshold). This bounds the search space.
        total = int(E) * int(R) * int(E)
        effective_cap = self.cap_candidates
        if self.q is not None:
            cap_q = max(1, int(math.ceil((1.0 - self.q) * total)))
            if effective_cap is not None and effective_cap < cap_q:
                logger.warning(f"cap_candidates < cap_q  from q-quantile. Using smaller cap_candidates {self.cap_candidates} instead of {cap_q}.")
            effective_cap = cap_q if effective_cap is None else min(effective_cap, cap_q)
        if effective_cap is None:
            raise ValueError("Candidate generation requires a cap (q or cap_candidates). Threshold-only (p) is unsafe for large graphs.")

        # Compute global top-k in a streaming fashion without O(E*R*E) memory
        # Peak RAM ~ rel_block_size * E * E * 4 bytes
        # memory_limit_gb = 1.0  # target RAM limit in GB
        # bytes_per_float = 4
        # max_bytes = int(memory_limit_gb * (1024**3))
        # rel_block_size = max(1, max_bytes // max(1, (E * E * bytes_per_float)))

        top_log_vals, r_idx, h_idx, t_idx = self._global_topk_joint_streaming(
            log_p_head_2d=log_p_head_2d,
            log_p_tail_2d=log_p_tail_2d,
            alpha=self.alpha,
            k=int(effective_cap),
            rel_block_size=10  # tune based on memory constraints,
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
    def __init__(self, topk: int):
        super().__init__()
        self.topk = topk    # number of (r,t) pairs to keep per head entity
        assert self.topk and self.topk > 0, "topk must be > 0"

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
        k_eff = min(self.topk, R * E)
        
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
    def __init__(self, p: float, q: float, temperature: float, alpha: float, beta: float, cap_candidates: int):
        super().__init__()
        self.p = p                              # keep candidates with P >= p (global threshold)
        self.q = q                              # use as fraction-cap: cap = ceil((1-q) * |E|*|R|*|E|)
        self.temperature = temperature          # temperature for softmax calibration
        self.alpha = alpha                      # weight for head log-probs P(r|h)
        self.beta = beta                        # weight for tail prediction P(t|h)
        self.cap_candidates = cap_candidates    # final cap after thresholding only keep top-k candidates

        # sanity checks
        assert self.temperature > 0, "temperature must be > 0"
        assert 0.0 <= self.alpha <= 1.0, "alpha must be in [0,1]"
        assert 0.0 <= self.beta <= 1.0, "beta must be in [0,1]"
        assert self.alpha + self.beta <= 1.0, "alpha + beta must be <= 1.0 (gamma = 1 - alpha - beta)"
        if self.p is not None:
            assert 0.0 <= self.p <= 1.0, "p must be in [0,1]"
        if self.q is not None:
            assert 0.0 <= self.q < 1.0, "q must be in [0,1)"

    @staticmethod
    def _global_topk_joint_streaming(
        log_p_head_2d: torch.Tensor,
        log_p_tail_2d: torch.Tensor,
        log_p_t_given_h_2d: torch.Tensor,
        alpha: float,
        beta: float,
        k: int,
        rel_block_size: int = 1,
    ):
        """
        Memory-efficient global top-k over all (head, relation, tail) triples, now including P(t|h).

        Vectorized over relation chunks:
        - Process at most `rel_block_size` relations at a time.
        - For a chunk r in [r0:r1), build a joint score tensor S with shape (C, E, E),
          where C = r1 - r0 and:
              S[c, h, t] = alpha * log_p_head_2d[h, r0+c] + beta * log_p_t_given_h_2d[h, t] + gamma * log_p_tail_2d[r0+c, t]
              (gamma = 1 - alpha - beta)
        - Run a single topk over the whole chunk (flattened), then decode to (r, h, t).
        - Merge the chunk top-k into a running global top-k buffer of size k.

        Args:
            log_p_head_2d: Tensor (E, R) with log P(r | h)
            log_p_tail_2d: Tensor (R, E) with log P(r^{-1} | t), aligned by relation index
            log_p_t_given_h_2d: Tensor (E, E) with log P(t | h)
            alpha: Weight in [0,1] for head log-probs P(r|h)
            beta: Weight in [0,1] for tail prediction P(t|h)
            k: Number of top entries to keep globally
            rel_block_size: Max number of relations to process per chunk

        Returns:
            top_vals: (k',) tensor of joint log-probs (k' <= k)
            top_r:    (k',) tensor of relation indices
            top_h:    (k',) tensor of head indices
            top_t:    (k',) tensor of tail indices
        """
        assert 0.0 <= alpha <= 1.0, "alpha must be in [0,1]"
        assert 0.0 <= beta <= 1.0, "beta must be in [0,1]"
        assert alpha + beta <= 1.0, "alpha + beta must be <= 1.0"
        gamma = 1.0 - alpha - beta
        E, R = log_p_head_2d.shape
        assert log_p_tail_2d.shape == (R, E), "log_p_tail_2d must be (R, E)"
        assert log_p_t_given_h_2d.shape == (E, E), "log_p_t_given_h_2d must be (E, E)"

        # Work on CPU float32 to minimize memory pressure
        log_p_head_2d = log_p_head_2d.to(dtype=torch.float32, device="cpu", copy=False)  # (E, R)
        log_p_tail_2d = log_p_tail_2d.to(dtype=torch.float32, device="cpu", copy=False)  # (R, E)
        log_p_t_given_h_2d = log_p_t_given_h_2d.to(dtype=torch.float32, device="cpu", copy=False)  # (E, E)

        # Running global top-k buffers (pre-allocated, updated incrementally)
        top_vals = torch.full((k,), float("-inf"), dtype=torch.float32)
        top_r    = torch.full((k,), -1, dtype=torch.long)
        top_h    = torch.full((k,), -1, dtype=torch.long)
        top_t    = torch.full((k,), -1, dtype=torch.long)
        filled = 0  # number of valid entries currently stored in the buffers

        a_h = float(alpha)
        a_t_pred = float(beta)
        a_t_inv = float(gamma)

        # Progress bar disappears after loop (leave=False)
        for r0 in tqdm(range(0, R, max(1, int(rel_block_size))),
                       desc=f"Computing global top-k candidates.", unit=f"{rel_block_size} relations", leave=False):
            r1 = min(R, r0 + max(1, int(rel_block_size)))
            C = r1 - r0  # number of relations in this chunk

            # Vectorized joint scores for the relation chunk
            # h_chunk: (E, C), t_inv_chunk: (C, E), t_pred_chunk: (E, E)
            h_chunk = (a_h * log_p_head_2d[:, r0:r1])         # (E, C)
            t_inv_chunk = (a_t_inv * log_p_tail_2d[r0:r1, :])  # (C, E)
            # S: (C, E, E) with broadcasting:
            # S[c, h, t] = h_chunk[h, c] + t_inv_chunk[c, t] + a_t_pred * log_p_t_given_h_2d[h, t]
            S = h_chunk.t().unsqueeze(2) + t_inv_chunk.unsqueeze(1) + (a_t_pred * log_p_t_given_h_2d).unsqueeze(0)  # (C, E, E)

            # Single top-k for the whole chunk
            numel_chunk = S.numel()
            if numel_chunk == 0:
                continue
            k_chunk = min(k, numel_chunk)
            vals_chunk, idx_chunk_flat = torch.topk(S.reshape(-1), k=k_chunk, largest=True)

            # Decode flat indices to (c, h, t) and map to global r = r0 + c
            per_rel = E * E
            c_idx = idx_chunk_flat // per_rel
            rem   = idx_chunk_flat % per_rel
            h_idx = rem // E
            t_idx = rem %  E
            r_idx = (c_idx + r0).to(torch.long)

            # Merge with running global top-k heap
            if filled == 0:
                take = min(k, vals_chunk.numel())
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

                if cand_vals.numel() > k:
                    vtop, order = torch.topk(cand_vals, k=k, largest=True)
                    top_vals[:k] = vtop
                    top_r[:k]    = cand_r[order]
                    top_h[:k]    = cand_h[order]
                    top_t[:k]    = cand_t[order]
                    filled = k
                else:
                    top_vals[:cand_vals.numel()] = cand_vals
                    top_r[:cand_vals.numel()]    = cand_r
                    top_h[:cand_vals.numel()]    = cand_h
                    top_t[:cand_vals.numel()]    = cand_t
                    filled = cand_vals.numel()

        # Trim to filled size
        return top_vals[:filled], top_r[:filled], top_h[:filled], top_t[:filled]

    def generate_candidates(
        self,
        tuples: torch.Tensor,
        logits_rp: torch.Tensor,
        relation_maps: RelationMaps,
        logits_tp: torch.Tensor = None,  # NEW: Tail logits (batch_size, num_entities)
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

        # 5. Calibrated log-probabilities (avoid tiny exp, use log_softmax); keep as 2D on CPU
        log_p_head_2d = torch.log_softmax(head_logits_subset / self.temperature, dim=1).to(torch.float32).cpu()  # (E, R)
        # transpose to (R, E) to index by relation first on tail-side
        log_p_tail_2d = torch.log_softmax(tail_logits_subset / self.temperature, dim=1).to(torch.float32).cpu().transpose(0, 1)  # (R, E)
        # For tail prediction P(t|h)
        log_p_t_given_h_2d = torch.log_softmax(logits_tp_grouped / self.temperature, dim=1).to(torch.float32).cpu()  # (E, E)

        # Derive effective cap from q first (before any threshold). This bounds the search space.
        total = int(E) * int(R) * int(E)
        effective_cap = self.cap_candidates
        if self.q is not None:
            cap_q = max(1, int(math.ceil((1.0 - self.q) * total)))
            if effective_cap is not None and effective_cap < cap_q:
                logger.warning(f"cap_candidates < cap_q from q-quantile. Using smaller cap_candidates {self.cap_candidates} instead of {cap_q}.")
            effective_cap = cap_q if effective_cap is None else min(effective_cap, cap_q)
        if effective_cap is None:
            raise ValueError("Candidate generation requires a cap (q or cap_candidates). Threshold-only (p) is unsafe for large graphs.")

        # Compute global top-k in a streaming fashion without O(E*R*E) memory
        top_log_vals, r_idx, h_idx, t_idx = self._global_topk_joint_streaming(
            log_p_head_2d=log_p_head_2d,
            log_p_tail_2d=log_p_tail_2d,
            log_p_t_given_h_2d=log_p_t_given_h_2d,
            alpha=self.alpha,
            beta=self.beta,
            k=int(effective_cap),
            rel_block_size=10  # tune based on memory constraints
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
