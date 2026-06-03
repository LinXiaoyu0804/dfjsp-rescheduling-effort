from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.baselines.base import BaselineOutput
from src.baselines.dispatching import dispatching_release_decision
from src.train.checkpoint import load_checkpoint


RULE_NAME_TO_INDEX = {"SPT": 0, "MWKR": 1, "EDD": 2, "CR": 3}
RULE_INDEX_TO_NAME = {value: key for key, value in RULE_NAME_TO_INDEX.items()}
DEFAULT_RULE_CANDIDATES = ["SPT", "MWKR", "EDD", "CR"]


def _safe_stats(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim == 0:
        x = x.reshape(1, 1)
    feature_dim = int(x.shape[-1]) if x.ndim >= 2 else 1
    if x.numel() == 0:
        zeros = torch.zeros(feature_dim, dtype=torch.float32)
        return zeros, zeros
    if x.ndim == 1:
        x = x.unsqueeze(0)
    return x.mean(dim=0), x.std(dim=0, unbiased=False)


def extract_rule_selector_features(graph: dict[str, Any]) -> torch.Tensor:
    op_mean, op_std = _safe_stats(graph["op_x"].float())
    machine_mean, machine_std = _safe_stats(graph["machine_x"].float())
    event_x = graph["event_x"].float().reshape(-1)

    snapshot = graph["snapshot"]
    releasable_count = len(
        [
            op_id
            for op_id in snapshot.window_op_ids
            if op_id not in snapshot.completed_op_ids and op_id not in snapshot.active_op_ids
        ]
    )
    window_count = max(1, len(snapshot.window_op_ids))
    unfinished_count = max(1, len(snapshot.unfinished_op_ids))

    scalars = torch.tensor(
        [
            torch.log1p(torch.tensor(float(len(graph["op_ids"])))).item(),
            torch.log1p(torch.tensor(float(len(graph["machine_ids"])))).item(),
            torch.log1p(torch.tensor(float(len(snapshot.completed_op_ids)))).item(),
            torch.log1p(torch.tensor(float(len(snapshot.active_op_ids)))).item(),
            torch.log1p(torch.tensor(float(window_count))).item(),
            torch.log1p(torch.tensor(float(len(snapshot.directly_impacted_op_ids)))).item(),
            torch.log1p(torch.tensor(float(releasable_count))).item(),
            float(len(snapshot.directly_impacted_op_ids)) / float(window_count),
            float(releasable_count) / float(window_count),
            float(window_count) / float(unfinished_count),
            torch.log1p(torch.tensor(float(snapshot.current_time))).item(),
        ],
        dtype=torch.float32,
    )

    return torch.cat([op_mean, op_std, machine_mean, machine_std, event_x, scalars], dim=0)


class LearnedRuleSelector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1, num_rules: int = 4):
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
        return self.net(x)


@dataclass(slots=True)
class SelectorBundle:
    model: LearnedRuleSelector
    rule_candidates: list[str]


_SELECTOR_CACHE: dict[tuple[str, str], SelectorBundle] = {}


def load_selector_bundle(cfg: dict[str, Any], device: str = "cpu") -> SelectorBundle:
    selector_cfg = cfg["rule_selector_baseline"]
    checkpoint_path = str(Path(selector_cfg["checkpoint_path"]))
    cache_key = (checkpoint_path, device)
    if cache_key in _SELECTOR_CACHE:
        return _SELECTOR_CACHE[cache_key]

    input_dim = int(selector_cfg.get("input_dim", 65))
    hidden_dim = int(selector_cfg.get("hidden_dim", 64))
    dropout = float(selector_cfg.get("dropout", 0.1))
    model = LearnedRuleSelector(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
    extra = load_checkpoint(checkpoint_path, model)
    model.to(device)
    model.eval()
    rule_candidates = list(extra.get("rule_candidates", selector_cfg.get("rule_candidates", DEFAULT_RULE_CANDIDATES)))
    bundle = SelectorBundle(model=model, rule_candidates=rule_candidates)
    _SELECTOR_CACHE[cache_key] = bundle
    return bundle


def learned_rule_selector_decision(
    instance,
    incumbent,
    snapshot,
    graph: dict[str, Any],
    selector: SelectorBundle,
) -> BaselineOutput:
    features = extract_rule_selector_features(graph).unsqueeze(0)
    with torch.no_grad():
        logits = selector.model(features)
        chosen_index = int(torch.argmax(logits, dim=1).item())
    chosen_rule = selector.rule_candidates[chosen_index]
    baseline = dispatching_release_decision(instance, incumbent, snapshot, rule=chosen_rule)
    baseline.decision.metadata["selector_rule"] = chosen_rule
    baseline.decision.metadata["selector_type"] = "learned_rule_selector"
    baseline.name = "learned_rule_selector"
    return baseline
