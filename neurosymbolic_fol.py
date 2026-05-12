"""A tiny neuro-symbolic model that uses First-Order Logic (FOL).

This module combines:
1) A neural scorer over entity/relation embeddings.
2) Symbolic FOL rules used for differentiable regularization and hard checks.

The implementation is intentionally compact and educational.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


Atom = Tuple[str, str, str]  # (predicate, subject, object)
Rule = Tuple[Atom, Atom]  # implication body -> head


@dataclass
class KnowledgeBase:
    """Simple triple store over named entities and binary predicates."""

    entities: Sequence[str]
    predicates: Sequence[str]
    facts: Sequence[Atom]

    def entity_index(self) -> Dict[str, int]:
        return {name: i for i, name in enumerate(self.entities)}

    def predicate_index(self) -> Dict[str, int]:
        return {name: i for i, name in enumerate(self.predicates)}


class NeuroSymbolicFOLModel(nn.Module):
    """TransE-style neural link predictor with FOL rule regularization.

    Neural score: s(p, h, t) = -||e_h + r_p - e_t||
    Truth value: sigma(s)

    For each rule body(x, y) -> head(x, y), we push:
      truth(head) >= truth(body)
    by minimizing relu(truth(body) - truth(head)).
    """

    def __init__(
        self,
        kb: KnowledgeBase,
        rules: Sequence[Rule],
        embedding_dim: int = 32,
        rule_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.kb = kb
        self.rules = list(rules)
        self.rule_weight = rule_weight

        self.e2i = kb.entity_index()
        self.p2i = kb.predicate_index()

        self.entity_emb = nn.Embedding(len(kb.entities), embedding_dim)
        self.relation_emb = nn.Embedding(len(kb.predicates), embedding_dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def score_triples(self, triples: Sequence[Atom]) -> torch.Tensor:
        p = torch.tensor([self.p2i[t[0]] for t in triples], dtype=torch.long)
        h = torch.tensor([self.e2i[t[1]] for t in triples], dtype=torch.long)
        t = torch.tensor([self.e2i[t[2]] for t in triples], dtype=torch.long)

        ph = self.entity_emb(h)
        pp = self.relation_emb(p)
        pt = self.entity_emb(t)
        return -torch.norm(ph + pp - pt, dim=-1)

    def truth(self, triples: Sequence[Atom]) -> torch.Tensor:
        return torch.sigmoid(self.score_triples(triples))

    def data_loss(self, positive: Sequence[Atom], negative: Sequence[Atom]) -> torch.Tensor:
        pos_scores = self.score_triples(positive)
        neg_scores = self.score_triples(negative)
        logits = torch.cat([pos_scores, neg_scores], dim=0)
        labels = torch.cat(
            [
                torch.ones(len(positive), dtype=torch.float32),
                torch.zeros(len(negative), dtype=torch.float32),
            ],
            dim=0,
        )
        return F.binary_cross_entropy_with_logits(logits, labels)

    def rule_loss(self) -> torch.Tensor:
        if not self.rules:
            return torch.tensor(0.0)

        violations: List[torch.Tensor] = []
        for body, head in self.rules:
            body_truth = self.truth([body])[0]
            head_truth = self.truth([head])[0]
            violations.append(F.relu(body_truth - head_truth))

        return torch.stack(violations).mean()

    def loss(self, positive: Sequence[Atom], negative: Sequence[Atom]) -> torch.Tensor:
        return self.data_loss(positive, negative) + self.rule_weight * self.rule_loss()


def sample_negative(facts: Sequence[Atom], entities: Sequence[str]) -> List[Atom]:
    """Corrupt object entity for each fact to build negative samples."""

    out: List[Atom] = []
    n = len(entities)
    for i, (p, s, o) in enumerate(facts):
        replacement = entities[(i + 1) % n]
        if replacement == o:
            replacement = entities[(i + 2) % n]
        out.append((p, s, replacement))
    return out


def build_demo_model() -> Tuple[NeuroSymbolicFOLModel, List[Atom], List[Atom]]:
    """Create a toy neuro-symbolic FOL setup.

    Rule encoded:
      parent(x, y) -> ancestor(x, y)
    """

    kb = KnowledgeBase(
        entities=["alice", "bob", "carol", "dave"],
        predicates=["parent", "ancestor", "likes"],
        facts=[
            ("parent", "alice", "bob"),
            ("parent", "bob", "carol"),
            ("ancestor", "alice", "bob"),
            ("likes", "carol", "dave"),
        ],
    )

    rules: List[Rule] = [
        (("parent", "alice", "bob"), ("ancestor", "alice", "bob")),
        (("parent", "bob", "carol"), ("ancestor", "bob", "carol")),
    ]

    model = NeuroSymbolicFOLModel(kb=kb, rules=rules, embedding_dim=16, rule_weight=0.8)
    negative = sample_negative(kb.facts, kb.entities)
    return model, list(kb.facts), negative


def train_demo(steps: int = 200, lr: float = 1e-2) -> NeuroSymbolicFOLModel:
    model, positive, negative = build_demo_model()
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(steps):
        optim.zero_grad()
        loss = model.loss(positive, negative)
        loss.backward()
        optim.step()

    return model


if __name__ == "__main__":
    model, pos, neg = build_demo_model()
    start = model.loss(pos, neg).item()
    trained = train_demo(steps=100)
    end = trained.loss(pos, neg).item()
    print(f"initial loss: {start:.4f}")
    print(f"final loss:   {end:.4f}")
