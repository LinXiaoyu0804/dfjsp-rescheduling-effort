from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from src.baselines.base import BaselineOutput
from src.baselines.daniel_local import daniel_local_decision
from src.baselines.ddpg import ddpg_decision, load_ddpg_bundle
from src.baselines.dispatching import dispatching_release_decision
from src.baselines.full_reopt import full_reoptimization_decision
from src.baselines.heuristic_rh import heuristic_rh_decision
from src.baselines.learned_rule_selector import learned_rule_selector_decision, load_selector_bundle
from src.baselines.no_learning_repair import no_learning_repair_decision


def _normalize_baseline_key(name: str) -> str:
    return str(name).strip().lower()


def _resolve_acceptance_reference_policy(
    baseline_name: str,
    controller_cfg: dict[str, Any],
) -> dict[str, Any]:
    policy: dict[str, Any] = {}
    global_gate = controller_cfg.get("acceptance_reference_gate", {})
    if isinstance(global_gate, dict):
        policy.update(global_gate)

    baseline_key = _normalize_baseline_key(baseline_name)
    per_baseline_gates = controller_cfg.get("acceptance_reference_gates", {})
    if isinstance(per_baseline_gates, dict):
        matched_gate = per_baseline_gates.get(baseline_name)
        if matched_gate is None:
            for key, value in per_baseline_gates.items():
                if _normalize_baseline_key(str(key)) == baseline_key:
                    matched_gate = value
                    break
        if isinstance(matched_gate, dict):
            policy.update(matched_gate)

    margin_abs = controller_cfg.get("acceptance_reference_margin_abs", 0.0)
    try:
        policy["margin_abs"] = float(policy.get("margin_abs", margin_abs))
    except (TypeError, ValueError):
        policy["margin_abs"] = 0.0
    return policy


def _acceptance_event_type_bucket(snapshot_row: dict[str, Any]) -> str:
    event_context = snapshot_row.get("event_context", {})
    event_type = str(event_context.get("type", "")).strip().lower()
    if "arrival" in event_type:
        return "arrival"
    if "breakdown" in event_type:
        return "breakdown"
    event_id = str(snapshot_row.get("event_id", "")).strip().lower()
    if event_id.startswith("arr_"):
        return "arrival"
    if event_id.startswith("bd_"):
        return "breakdown"
    return "other"


def acceptance_reference_gate_passes(
    baseline_name: str,
    snapshot_row: dict[str, Any],
    controller_cfg: dict[str, Any],
    *,
    learned_selection_active: bool,
) -> bool:
    policy = _resolve_acceptance_reference_policy(baseline_name, controller_cfg)
    if bool(policy.get("require_no_learned_selection", False)) and learned_selection_active:
        return False
    if bool(policy.get("require_learned_selection", False)) and not learned_selection_active:
        return False

    window_size = len(snapshot_row.get("window_ops", []))
    forced_release_count = len(snapshot_row.get("forced_release_ops", []))
    motif_count = int(snapshot_row.get("motif_count", 0))

    if window_size < int(policy.get("min_window_size", 0)):
        return False
    if forced_release_count < int(policy.get("min_forced_release_count", 0)):
        return False
    if motif_count < int(policy.get("min_motif_count", 0)):
        return False

    allowed_event_types = policy.get("allowed_event_types", [])
    if isinstance(allowed_event_types, (list, tuple)) and allowed_event_types:
        normalized_allowed = {_normalize_baseline_key(str(item)) for item in allowed_event_types}
        if _acceptance_event_type_bucket(snapshot_row) not in normalized_allowed:
            return False
    return True


def acceptance_reference_margin_abs(
    baseline_name: str,
    controller_cfg: dict[str, Any],
) -> float:
    policy = _resolve_acceptance_reference_policy(baseline_name, controller_cfg)
    return max(0.0, float(policy.get("margin_abs", 0.0)))


def merge_teacher_trace_shards(shard_paths: list[str | Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    merged_rows: list[dict[str, Any]] = []
    grouped_counts: dict[tuple[str, int], dict[str, int | str]] = {}
    for shard_path_like in sorted(Path(path_like) for path_like in shard_paths):
        shard_path = Path(shard_path_like)
        shard_rows = torch.load(shard_path, map_location="cpu", weights_only=False)
        for row in shard_rows:
            merged_rows.append(row)
            metadata = dict(row.get("metadata", {}))
            key = (str(metadata.get("instance_id", "")), int(metadata.get("seed", 0)))
            summary = grouped_counts.setdefault(
                key,
                {
                    "instance_id": str(metadata.get("instance_id", "")),
                    "seed": int(metadata.get("seed", 0)),
                    "num_events": 0,
                    "num_feasible_events": 0,
                    "teacher_release_total": 0,
                },
            )
            summary["num_events"] = int(summary["num_events"]) + 1
            summary["num_feasible_events"] = int(summary["num_feasible_events"]) + int(
                bool(metadata.get("teacher_feasible", False))
            )
            summary["teacher_release_total"] = int(summary["teacher_release_total"]) + int(
                metadata.get("teacher_release_count", 0)
            )
    summary_rows: list[dict[str, Any]] = []
    for _, summary in sorted(grouped_counts.items(), key=lambda item: (item[0][0], item[0][1])):
        num_events = max(1, int(summary["num_events"]))
        summary_rows.append(
            {
                "instance_id": str(summary["instance_id"]),
                "seed": int(summary["seed"]),
                "num_events": int(summary["num_events"]),
                "num_feasible_events": int(summary["num_feasible_events"]),
                "feasible_rate": float(summary["num_feasible_events"]) / float(num_events),
                "mean_teacher_release_count": float(summary["teacher_release_total"]) / float(num_events),
            }
        )
    return merged_rows, summary_rows


def compute_baseline_proxy_selection(
    released_op_ids: list[int],
    forced_release_ops: list[int],
) -> tuple[int, int]:
    forced_release = {int(op_id) for op_id in forced_release_ops}
    released = {int(op_id) for op_id in released_op_ids}
    extra_released = released - forced_release
    return (1 if extra_released else 0), len(extra_released)


def build_external_baseline_output(
    *,
    baseline_name: str,
    instance,
    incumbent,
    snapshot,
    graph: dict[str, Any] | None,
    cfg: dict[str, Any],
    device: str = "cpu",
    cache: dict[str, Any] | None = None,
) -> BaselineOutput:
    cache_dict = cache if cache is not None else {}
    baseline_key = str(baseline_name).strip().lower()
    if baseline_key == "dispatching_spt":
        return dispatching_release_decision(instance, incumbent, snapshot, rule="SPT")
    if baseline_key == "dispatching_mwkr":
        return dispatching_release_decision(instance, incumbent, snapshot, rule="MWKR")
    if baseline_key == "dispatching_edd":
        return dispatching_release_decision(instance, incumbent, snapshot, rule="EDD")
    if baseline_key == "dispatching_cr":
        return dispatching_release_decision(instance, incumbent, snapshot, rule="CR")
    if baseline_key == "dispatching_atc":
        return dispatching_release_decision(instance, incumbent, snapshot, rule="ATC")
    if baseline_key == "heuristic_rh":
        return heuristic_rh_decision(snapshot)
    if baseline_key == "full_reoptimization":
        return full_reoptimization_decision(snapshot)
    if baseline_key == "no_learning_repair":
        return no_learning_repair_decision(snapshot)
    if baseline_key == "learned_rule_selector":
        selector_bundle = cache_dict.get("learned_rule_selector")
        if selector_bundle is None:
            selector_bundle = load_selector_bundle(cfg)
            cache_dict["learned_rule_selector"] = selector_bundle
        if graph is None:
            raise RuntimeError("learned_rule_selector baseline requires a graph state.")
        return learned_rule_selector_decision(instance, incumbent, snapshot, graph, selector_bundle)
    if baseline_key == "ddpg":
        ddpg_bundle = cache_dict.get("ddpg")
        if ddpg_bundle is None:
            ddpg_bundle = load_ddpg_bundle(cfg, device=device)
            cache_dict["ddpg"] = ddpg_bundle
        if graph is None:
            raise RuntimeError("ddpg baseline requires a graph state.")
        return ddpg_decision(instance, incumbent, snapshot, graph, ddpg_bundle)
    if baseline_key == "daniel_local":
        return daniel_local_decision(instance, incumbent, snapshot, cfg, device=device)
    raise ValueError(f"Unknown baseline: {baseline_name}")
