"""
RETA-Filter: Schema-Aware Candidate Filter for Instance Completion
==================================================================
Reproduced from:
  Rosso, Yang, Ostapuk, Cudré-Mauroux.
  "RETA: A Schema-Aware, End-to-End Solution for Instance Completion in Knowledge Graphs."
  WWW'21. https://doi.org/10.1145/3442381.3449883

This module implements RETA-Filter, which generates a set of candidate (relation, tail)
pairs for a given head entity h by extracting and exploiting the implicit schema of a KG
through entity-typed triplets.

Two variants are implemented:
  - WITH entity types   (uses explicit h_type, r, t_type annotations)
  - WITHOUT entity types (uses heuristic domain/range types derived from training triples)

The no-type variant matches the setting of our paper, where entity types are unavailable.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class KGSchema:
    """
    Encodes the implicit schema of a KG extracted from training triples.

    For the no-type variant, heuristic types are used:
      - h_type(h)  = frozenset of outgoing relation types from h in training
      - t_type(t)  = frozenset of incoming relation types to   t in training

    These are analogous to r_domain / r_range used by RETA-Filter when true
    entity types are unavailable (see RETA paper, Section 3.1, para 5).
    """
    # entity -> frozenset of outgoing relation ids
    entity_domain: Dict[int, frozenset] = field(default_factory=dict)
    # entity -> frozenset of incoming relation ids
    entity_range: Dict[int, frozenset] = field(default_factory=dict)
    # (h_type_key, r, t_type_key) -> frequency
    typed_triple_freq: Dict[Tuple, int] = field(default_factory=dict)
    # relation -> set of (h_domain_key, t_range_key) pairs observed in training
    relation_schema: Dict[int, Set[Tuple]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Schema extraction
# ---------------------------------------------------------------------------

def build_schema_from_triples(
    train_triples: torch.Tensor,
    use_entity_types: bool = False,
    entity_types: Optional[Dict[int, List[int]]] = None,
    alpha: float = 0.0,
) -> KGSchema:
    """
    Build the KG schema from training triples.

    Args:
        train_triples: (N, 3) tensor of (head, relation, tail) ids.
        use_entity_types: If True, use explicit entity type annotations.
        entity_types: Dict[entity_id -> list of type_ids]. Required if use_entity_types=True.
        alpha: Frequency threshold. Entity-typed triplets with frequency < alpha are discarded.
               Set to 0 to keep all (equivalent to using it as a Boolean tensor).

    Returns:
        KGSchema with extracted type information.
    """
    schema = KGSchema()

    triples_np = train_triples.numpy() if isinstance(train_triples, torch.Tensor) else np.array(train_triples)

    # Build entity domain/range maps
    entity_outgoing: Dict[int, Set[int]] = defaultdict(set)  # h -> {r}
    entity_incoming: Dict[int, Set[int]] = defaultdict(set)  # t -> {r}

    for h, r, t in triples_np:
        entity_outgoing[int(h)].add(int(r))
        entity_incoming[int(t)].add(int(r))

    schema.entity_domain = {e: frozenset(rs) for e, rs in entity_outgoing.items()}
    schema.entity_range  = {e: frozenset(rs) for e, rs in entity_incoming.items()}

    # Build entity-typed triplet frequency tensor
    # typed_triple_freq[(h_type_key, r, t_type_key)] = count
    typed_freq: Dict[Tuple, int] = defaultdict(int)

    for h, r, t in triples_np:
        h, r, t = int(h), int(r), int(t)
        if use_entity_types and entity_types is not None:
            h_types = entity_types.get(h, [])
            t_types = entity_types.get(t, [])
        else:
            # Heuristic types: use the domain/range frozensets as type keys
            h_types = [schema.entity_domain.get(h, frozenset())]
            t_types = [schema.entity_range.get(t, frozenset())]

        for h_type in h_types:
            for t_type in t_types:
                typed_freq[(h_type, r, t_type)] += 1

    # Apply frequency threshold alpha
    schema.typed_triple_freq = {
        key: freq for key, freq in typed_freq.items() if freq > alpha
    }

    # Build relation_schema: r -> set of (h_type, t_type) pairs
    rel_schema: Dict[int, Set[Tuple]] = defaultdict(set)
    for (h_type, r, t_type), freq in schema.typed_triple_freq.items():
        rel_schema[r].add((h_type, t_type))
    schema.relation_schema = dict(rel_schema)

    logger.info(
        f"Built schema: {len(schema.typed_triple_freq)} entity-typed triplets, "
        f"{len(rel_schema)} relations with schema entries."
    )
    return schema


# ---------------------------------------------------------------------------
# RETA-Filter
# ---------------------------------------------------------------------------

class RETAFilter:
    """
    RETA-Filter generates candidate (r, t) pairs for a head entity h by
    exploiting the KG schema.

    Algorithm (no-type variant):
      1. For head entity h, compute its heuristic type: h_type = domain(h).
      2. For each candidate relation r, check if (h_type, r, *) appears in
         the schema tensor with count >= beta.
      3. For each valid (h_type, r, t_type) entry, collect all tail entities t
         in training whose range matches t_type.
      4. Return the union of all (r, t) candidates, ranked by schema match count.

    Parameters:
        alpha (float): Minimum frequency of entity-typed triplet to include in schema (≥0).
        beta  (int):   Minimum schema match count for a candidate (r, t) to be included (≥1).

    Reference: RETA paper, Section 3.1.
    """

    def __init__(
        self,
        schema: KGSchema,
        train_triples: torch.Tensor,
        beta: int = 1,
    ):
        self.schema = schema
        self.beta = beta

        # Precompute: for each (r, t_type_key) -> list of tail entities with that type
        self._rt_index: Dict[Tuple[int, frozenset], List[int]] = defaultdict(list)
        triples_np = train_triples.numpy() if isinstance(train_triples, torch.Tensor) else np.array(train_triples)
        for h, r, t in triples_np:
            h, r, t = int(h), int(r), int(t)
            t_type = schema.entity_range.get(t, frozenset())
            self._rt_index[(r, t_type)].append(t)

        # Deduplicate
        self._rt_index = {k: list(set(v)) for k, v in self._rt_index.items()}

        logger.info(
            f"RETAFilter initialised: beta={beta}, "
            f"{len(self._rt_index)} (r, t_type) index entries."
        )

    def generate_candidates(
        self,
        head: int,
        max_candidates: Optional[int] = None,
    ) -> List[Tuple[int, int, int]]:
        """
        Generate candidate (head, relation, tail) triples for a given head entity.

        Args:
            head: The query head entity id.
            max_candidates: If set, return at most this many candidates (ranked by score).

        Returns:
            List of (head, relation, tail) tuples, sorted by descending schema match count.
        """
        h_type = self.schema.entity_domain.get(head, frozenset())

        # Collect candidates and their schema match scores
        candidate_scores: Dict[Tuple[int, int], float] = defaultdict(float)

        for (ht, r, tt), freq in self.schema.typed_triple_freq.items():
            # Check if h_type is compatible with this schema entry's h_type
            if not self._type_compatible(h_type, ht):
                continue

            # Retrieve all tails matching t_type = tt
            tails = self._rt_index.get((r, tt), [])
            for t in tails:
                score = candidate_scores[(r, t)]
                candidate_scores[(r, t)] = score + freq

        # Filter by beta threshold
        filtered = [
            (r, t, score)
            for (r, t), score in candidate_scores.items()
            if score >= self.beta
        ]

        # Sort by descending score
        filtered.sort(key=lambda x: -x[2])

        if max_candidates is not None:
            filtered = filtered[:max_candidates]

        return [(head, r, t) for r, t, _ in filtered]

    def generate_candidates_batch(
        self,
        heads: List[int],
        max_candidates: Optional[int] = None,
    ) -> Dict[int, List[Tuple[int, int, int]]]:
        """Generate candidates for a batch of head entities."""
        return {
            h: self.generate_candidates(h, max_candidates=max_candidates)
            for h in heads
        }

    @staticmethod
    def _type_compatible(entity_type: frozenset, schema_type) -> bool:
        """
        Check if an entity's heuristic type is compatible with a schema type entry.

        For the no-type variant, both are frozensets of relation ids.
        Compatibility = non-empty intersection (entity has at least one relation
        in common with the schema entry's domain).
        """
        if isinstance(schema_type, frozenset):
            return len(entity_type & schema_type) > 0
        # For explicit type ids (integer), check direct membership
        return schema_type in entity_type


# ---------------------------------------------------------------------------
# Coverage evaluation utility
# ---------------------------------------------------------------------------

def evaluate_filter_coverage(
    candidates: Dict[int, List[Tuple[int, int, int]]],
    gold_triples: torch.Tensor,
    test_heads: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Evaluate candidate filter quality: coverage and candidate set size.

    Coverage = fraction of gold (h, r, t) test triples covered by the candidate set.
    Size     = average number of candidates per head entity.

    Args:
        candidates: Dict mapping head entity id -> list of (h, r, t) candidates.
        gold_triples: (N, 3) tensor of ground-truth test triples.
        test_heads: If given, restrict evaluation to these heads.

    Returns:
        Dict with keys: 'coverage', 'avg_size', 'total_candidates', 'total_gold'.
    """
    gold_np = gold_triples.numpy() if isinstance(gold_triples, torch.Tensor) else np.array(gold_triples)

    # Build gold set per head
    gold_per_head: Dict[int, Set[Tuple[int, int]]] = defaultdict(set)
    for h, r, t in gold_np:
        gold_per_head[int(h)].add((int(r), int(t)))

    if test_heads is None:
        test_heads = list(gold_per_head.keys())

    total_gold = 0
    covered = 0
    total_cand = 0

    for h in test_heads:
        gold = gold_per_head.get(h, set())
        cand = {(r, t) for (hh, r, t) in candidates.get(h, [])}
        total_gold += len(gold)
        covered    += len(gold & cand)
        total_cand += len(cand)

    coverage  = covered / total_gold if total_gold > 0 else 0.0
    avg_size  = total_cand / len(test_heads) if test_heads else 0.0

    return {
        "coverage":          coverage,
        "avg_size":          avg_size,
        "total_candidates":  total_cand,
        "total_gold":        total_gold,
        "covered":           covered,
    }


# ---------------------------------------------------------------------------
# End-to-end helper
# ---------------------------------------------------------------------------

def build_reta_filter(
    train_triples: torch.Tensor,
    alpha: float = 0.0,
    beta: int = 1,
    use_entity_types: bool = False,
    entity_types: Optional[Dict[int, List[int]]] = None,
) -> RETAFilter:
    """
    Convenience function: build schema and RETA-Filter from training triples.

    Args:
        train_triples: (N, 3) int tensor.
        alpha: Minimum entity-typed triplet frequency to include in schema.
        beta:  Minimum schema match count for candidate inclusion.
        use_entity_types: Use explicit entity type annotations.
        entity_types: Dict[entity_id -> list of type_ids].

    Returns:
        Fitted RETAFilter instance.
    """
    schema = build_schema_from_triples(
        train_triples,
        use_entity_types=use_entity_types,
        entity_types=entity_types,
        alpha=alpha,
    )
    return RETAFilter(schema=schema, train_triples=train_triples, beta=beta)
