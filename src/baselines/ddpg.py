from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.baselines.base import BaselineOutput
from src.baselines.dispatching import (
    dispatching_release_count,
    order_dispatching_candidates,
)
from src.baselines.learned_rule_selector import extract_rule_selector_features
from src.solver.base import RepairDecision
from src.train.checkpoint import load_checkpoint


DEFAULT_RULE_CANDIDATES = ["SPT", "MWKR", "EDD", "CR", "ATC"]


def extract_ddpg_state_features(graph: dict[str, Any]) -> torch.Tensor:
    return extract_rule_selector_features(graph)


def build_rule_priority_matrix(
    instance,
    snapshot,
    op_ids: list[int],
    rule_candidates: list[str],
) -> torch.Tensor:
    score_matrix = torch.zeros((len(op_ids), len(rule_candidates)), dtype=torch.float32)
    if not op_ids or not rule_candidates:
        return score_matrix

    op_index = {op_id: idx for idx, op_id in enumerate(op_ids)}
    for rule_idx, rule in enumerate(rule_candidates):
        ordered = order_dispatching_candidates(instance, snapshot, rule=rule)
        if not ordered:
            continue
        denom = max(1, len(ordered) - 1)
        for rank, op_id in enumerate(ordered):
            idx = op_index.get(op_id)
            if idx is None:
                continue
            score = 1.0 if len(ordered) == 1 else 1.0 - (rank / float(denom))
            score_matrix[idx, rule_idx] = float(score)
    return score_matrix


class DDPGRulePolicy(nn.Module):
    """
    Lightweight repo-consistent approximation of Gui et al. (2023).

    The model predicts continuous weights over a pool of dispatching rules,
    and the weighted rule scores form the constructive release priority.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1, num_rules: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_rules),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return torch.softmax(logits, dim=-1)


@dataclass(slots=True)
class DDPGBundle:
    model: DDPGRulePolicy
    rule_candidates: list[str]
    release_fraction: float


_DDPG_CACHE: dict[tuple[str, str], DDPGBundle] = {}


def load_ddpg_bundle(cfg: dict[str, Any], device: str = "cpu") -> DDPGBundle:
    ddpg_cfg = cfg["ddpg_baseline"]
    checkpoint_path = str(Path(ddpg_cfg["checkpoint_path"]))
    cache_key = (checkpoint_path, device)
    if cache_key in _DDPG_CACHE:
        return _DDPG_CACHE[cache_key]

    rule_candidates = list(ddpg_cfg.get("rule_candidates", DEFAULT_RULE_CANDIDATES))
    model = DDPGRulePolicy(
        input_dim=int(ddpg_cfg.get("input_dim", 65)),
        hidden_dim=int(ddpg_cfg.get("hidden_dim", 64)),
        dropout=float(ddpg_cfg.get("dropout", 0.1)),
        num_rules=len(rule_candidates),
    )
    extra = load_checkpoint(checkpoint_path, model)
    model.to(device)
    model.eval()
    bundle = DDPGBundle(
        model=model,
        rule_candidates=list(extra.get("rule_candidates", rule_candidates)),
        release_fraction=float(ddpg_cfg.get("release_fraction", 1.0 / 3.0)),
    )
    _DDPG_CACHE[cache_key] = bundle
    return bundle


def ddpg_decision(
    instance,
    incumbent,
    snapshot,
    graph: dict[str, Any],
    bundle: DDPGBundle,
) -> BaselineOutput:
    op_ids = list(graph["op_ids"])
    features = extract_ddpg_state_features(graph).unsqueeze(0)
    score_matrix = build_rule_priority_matrix(instance, snapshot, op_ids, bundle.rule_candidates)
    with torch.no_grad():
        rule_weights = bundle.model(features).squeeze(0).cpu()
    composite_scores = score_matrix @ rule_weights

    ordered_with_scores = sorted(
        (
            (op_id, float(composite_scores[idx].item()))
            for idx, op_id in enumerate(op_ids)
            if score_matrix[idx].sum().item() > 0.0
        ),
        key=lambda item: (-item[1], item[0]),
    )
    ordered = [op_id for op_id, _ in ordered_with_scores]
    release_count = dispatching_release_count(len(ordered))
    if ordered:
        release_count = min(
            len(ordered),
            max(release_count, int(round(len(ordered) * bundle.release_fraction))),
        )
    release = ordered[:release_count]
    keep = [op_id for op_id in snapshot.window_op_ids if op_id not in release]
    immutable = sorted(set(snapshot.completed_op_ids + snapshot.active_op_ids))
    metadata = {
        "rule_candidates": bundle.rule_candidates,
        "rule_weights": {rule: float(weight) for rule, weight in zip(bundle.rule_candidates, rule_weights.tolist())},
        "selector_type": "ddpg",
    }
    return BaselineOutput(
        decision=RepairDecision(
            immutable_op_ids=immutable,
            kept_op_ids=keep,
            released_op_ids=release,
            metadata=metadata,
        ),
        name="ddpg",
    )
