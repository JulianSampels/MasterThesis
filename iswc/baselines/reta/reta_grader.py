"""
RETA-Grader: CNN-Based Relation-Tail Pair Ranking Model
=======================================================
Reproduced from:
  Rosso, Yang, Ostapuk, Cudré-Mauroux.
  "RETA: A Schema-Aware, End-to-End Solution for Instance Completion in Knowledge Graphs."
  WWW'21. https://doi.org/10.1145/3442381.3449883

RETA-Grader takes the candidate (r, t) set produced by RETA-Filter and ranks them by
estimating the plausibility of each triple (h, r, t), considering both:
  1. Triplet relatedness (from the triple itself via a CNN over embeddings).
  2. Schema relatedness (from entity-typed triplets (h_type, r, t_type) via a second CNN).

This reproduction implements a simplified version suitable for the no-type setting used
in our paper, where schema learning falls back to heuristic (domain/range) type information.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Triplet Relatedness Module (CNN over entity/relation embeddings)
# ---------------------------------------------------------------------------

class TripletCNN(nn.Module):
    """
    Captures triple-level relatedness by convolving over the concatenated
    embeddings of (h, r, t).

    Architecture (RETA paper, Section 3.2.1):
      - Concatenate: [h_emb; r_emb; t_emb] → matrix I ∈ R^{3 × K}
      - 2D-Conv with n_f filters of size 1×3 → n_f feature maps of size 1×(K-2)
      - Flatten → triplet relatedness vector φ ∈ R^{n_f * (K-2)}
    """

    def __init__(self, embed_dim: int, n_filters: int = 50):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_filters = n_filters
        # 2D conv: in_channels=1, out_channels=n_f, kernel=(3,3)
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=n_filters,
            kernel_size=(3, 3),
            padding=0,
        )
        self.out_dim = n_filters * (embed_dim - 2)

    def forward(self, h: Tensor, r: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            h, r, t: (B, D) entity/relation embeddings.
        Returns:
            phi: (B, n_f * (D-2)) triplet relatedness vector.
        """
        # Stack to (B, 3, D), then unsqueeze channel dim → (B, 1, 3, D)
        I = torch.stack([h, r, t], dim=1).unsqueeze(1)
        out = F.relu(self.conv(I))      # (B, n_f, 1, D-2)
        return out.view(out.size(0), -1)  # (B, n_f*(D-2))


# ---------------------------------------------------------------------------
# Schema Relatedness Module (CNN over entity-typed triplet embeddings)
# ---------------------------------------------------------------------------

class SchemaCNN(nn.Module):
    """
    Captures schema relatedness by convolving over entity-typed triplets.

    For each candidate (h, r, t), this module considers the entity-typed triplets
    (h_type_i, r, t_type_j) associated with h and t, computes one relatedness
    vector per typed triplet, then aggregates using a min operation across all
    m×n typed triplets.

    In the no-type variant, m = n = 1 (one heuristic type per entity), so the
    schema CNN is applied once and no min aggregation is needed.

    Architecture: same CNN as TripletCNN but applied to type embeddings.
    """

    def __init__(self, embed_dim: int, n_filters: int = 50, n_type_embed: int = 32):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_filters = n_filters
        # Type embedding table (maps frozenset-hash or type_id → embedding)
        self.type_embed_dim = n_type_embed
        self.type_proj = nn.Linear(embed_dim, n_type_embed)
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=n_filters,
            kernel_size=(3, 3),
            padding=0,
        )
        self.out_dim = n_filters * (n_type_embed - 2)

    def forward(self, h_type_emb: Tensor, r: Tensor, t_type_emb: Tensor) -> Tensor:
        """
        Args:
            h_type_emb: (B, D) embedding of the head entity's heuristic type.
            r:          (B, D) relation embedding.
            t_type_emb: (B, D) embedding of the tail entity's heuristic type.
        Returns:
            psi: (B, n_f * (n_type_embed - 2)) schema relatedness vector.
        """
        h_proj = F.relu(self.type_proj(h_type_emb))   # (B, n_type_embed)
        r_proj = F.relu(self.type_proj(r))
        t_proj = F.relu(self.type_proj(t_type_emb))
        I = torch.stack([h_proj, r_proj, t_proj], dim=1).unsqueeze(1)
        out = F.relu(self.conv(I))
        return out.view(out.size(0), -1)


# ---------------------------------------------------------------------------
# RETA-Grader
# ---------------------------------------------------------------------------

class RETAGrader(nn.Module):
    """
    Full RETA-Grader model that combines triplet and schema relatedness.

    Architecture:
      φ = TripletCNN(h_emb, r_emb, t_emb)         [triplet relatedness]
      ψ = SchemaCNN(h_type_emb, r_emb, t_type_emb) [schema relatedness]
      overall = concat(φ, ψ)
      σ = FC(ReLU(BN(overall))) → scalar score

    Training objective: softplus loss (negative log-likelihood of logistic model),
    as defined in RETA paper Eq. (2):
        L = Σ_{ω} [log(1 + exp(-σ(ω))) + log(1 + exp(-σ(ω')))]
    where ω is a positive triple and ω' is the corrupted negative.
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embed_dim: int = 64,
        n_filters: int = 50,
        n_type_embed: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_entities = num_entities
        self.num_relations = num_relations

        # Learnable embeddings
        self.entity_emb  = nn.Embedding(num_entities,  embed_dim)
        self.relation_emb = nn.Embedding(num_relations, embed_dim)

        # L2-normalise embeddings (as in RETA paper)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

        # Sub-modules
        self.triplet_cnn = TripletCNN(embed_dim, n_filters)
        self.schema_cnn  = SchemaCNN(embed_dim, n_filters, n_type_embed)

        combined_dim = self.triplet_cnn.out_dim + self.schema_cnn.out_dim

        # Prediction head
        self.bn1     = nn.BatchNorm1d(combined_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(combined_dim, 1)

    def _normalise_embeddings(self):
        """Enforce L2-norm constraint on entity and relation embeddings."""
        with torch.no_grad():
            self.entity_emb.weight.data  = F.normalize(self.entity_emb.weight.data,  p=2, dim=1)
            self.relation_emb.weight.data = F.normalize(self.relation_emb.weight.data, p=2, dim=1)

    def forward(
        self,
        heads: Tensor,       # (B,) int
        relations: Tensor,   # (B,) int
        tails: Tensor,       # (B,) int
        h_type_heads: Optional[Tensor] = None,  # (B,) int or None
        t_type_tails: Optional[Tensor] = None,  # (B,) int or None
    ) -> Tensor:
        """
        Compute plausibility score for each (h, r, t) triple.

        When type tensors are None (no-type setting), we use the entity's own
        embedding as a proxy for its type representation.

        Returns:
            scores: (B,) scalar plausibility scores (logits).
        """
        self._normalise_embeddings()

        h_emb = self.entity_emb(heads)
        r_emb = self.relation_emb(relations)
        t_emb = self.entity_emb(tails)

        # Triplet relatedness
        phi = self.triplet_cnn(h_emb, r_emb, t_emb)

        # Schema relatedness — fall back to entity embedding if no types
        h_type_emb = self.entity_emb(h_type_heads) if h_type_heads is not None else h_emb
        t_type_emb = self.entity_emb(t_type_tails) if t_type_tails is not None else t_emb
        psi = self.schema_cnn(h_type_emb, r_emb, t_type_emb)

        # Combine and project
        combined = torch.cat([phi, psi], dim=-1)
        combined = self.dropout(F.relu(self.bn1(combined)))
        return self.fc(combined).squeeze(-1)  # (B,)

    def score(self, heads: Tensor, relations: Tensor, tails: Tensor) -> Tensor:
        """Convenience inference-only scoring (sigmoid-normalised)."""
        with torch.no_grad():
            logits = self.forward(heads, relations, tails)
            return torch.sigmoid(logits)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

class RETAGraderTrainer:
    """
    Simple trainer for RETA-Grader using the softplus loss from the RETA paper.

    Loss per triple ω = (h, r, t) with negative sample ω' = (h, r, t'):
        L(ω) = log(1 + exp(-σ(ω))) + log(1 + exp(-σ(ω')))
             = softplus(-σ(ω)) + softplus(-σ(ω'))

    This is equivalent to binary cross-entropy with labels (1, 0).
    """

    def __init__(
        self,
        model: RETAGrader,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        self.model = model
        self.optimiser = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

    def train_step(
        self,
        pos_heads: Tensor,
        pos_rels:  Tensor,
        pos_tails: Tensor,
        neg_heads: Tensor,
        neg_rels:  Tensor,
        neg_tails: Tensor,
    ) -> float:
        """Single gradient step. Returns scalar loss value."""
        self.model.train()
        self.optimiser.zero_grad()

        pos_scores = self.model(pos_heads, pos_rels, pos_tails)
        neg_scores = self.model(neg_heads, neg_rels, neg_tails)

        # softplus(-σ) = log(1 + exp(-σ))
        loss = F.softplus(-pos_scores).mean() + F.softplus(neg_scores).mean()
        loss.backward()
        self.optimiser.step()
        return loss.item()

    def rank_candidates(
        self,
        head: int,
        candidates: List[Tuple[int, int]],
        device: torch.device = torch.device("cpu"),
    ) -> List[Tuple[int, int, float]]:
        """
        Rank a list of (relation, tail) candidates for a given head entity.

        Returns:
            List of (relation, tail, score) sorted by descending score.
        """
        if not candidates:
            return []

        self.model.eval()
        rs = torch.tensor([r for r, t in candidates], dtype=torch.long, device=device)
        ts = torch.tensor([t for r, t in candidates], dtype=torch.long, device=device)
        hs = torch.full_like(rs, head)

        scores = self.model.score(hs, rs, ts).cpu().numpy()
        ranked = sorted(
            zip([r for r, t in candidates], [t for r, t in candidates], scores.tolist()),
            key=lambda x: -x[2],
        )
        return [(r, t, s) for r, t, s in ranked]


# ---------------------------------------------------------------------------
# Negative sampling
# ---------------------------------------------------------------------------

def corrupt_triple(
    heads: Tensor,
    relations: Tensor,
    tails: Tensor,
    num_entities: int,
    corrupt_head_prob: float = 0.5,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Generate negative samples by randomly corrupting either the head or tail entity.

    Args:
        heads, relations, tails: (B,) positive triple tensors.
        num_entities: Total number of entities.
        corrupt_head_prob: Probability of corrupting the head (vs tail).

    Returns:
        Corrupted (heads, relations, tails) tensors of the same shape.
    """
    B = heads.size(0)
    random_entities = torch.randint(0, num_entities, (B,), device=heads.device)
    corrupt_head = torch.rand(B, device=heads.device) < corrupt_head_prob

    neg_heads = torch.where(corrupt_head, random_entities, heads)
    neg_tails = torch.where(~corrupt_head, random_entities, tails)
    return neg_heads, relations.clone(), neg_tails
