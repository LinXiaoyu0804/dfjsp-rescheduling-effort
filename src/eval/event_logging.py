from __future__ import annotations

from typing import Any


def build_event_log_row(
    *,
    method: str,
    instance_id: str,
    seed: int,
    episode_id: str,
    event_id: str,
    tau: float,
    budget_sec: float,
    window_size: int,
    forced_release_count: int,
    motif_count: int,
    selected_motif_count: int,
    released_op_count: int,
    pred_gain_sum: float | None,
    inference_runtime_sec: float,
    selector_runtime_sec: float,
    solver_runtime_sec: float,
    makespan_after: float,
    tardiness_after: float,
    instability_after: float,
    weighted_objective_after: float,
    changed_op_ratio: float,
    changed_machine_ratio: float,
    mean_abs_start_time_deviation: float,
    status: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "method": method,
        "instance_id": instance_id,
        "seed": int(seed),
        "episode_id": episode_id,
        "event_id": event_id,
        "tau": float(tau),
        "budget_sec": float(budget_sec),
        "window_size": int(window_size),
        "forced_release_count": int(forced_release_count),
        "motif_count": int(motif_count),
        "selected_motif_count": int(selected_motif_count),
        "released_op_count": int(released_op_count),
        "pred_gain_sum": None if pred_gain_sum is None else float(pred_gain_sum),
        "inference_runtime_sec": float(inference_runtime_sec),
        "selector_runtime_sec": float(selector_runtime_sec),
        "solver_runtime_sec": float(solver_runtime_sec),
        "makespan_after": float(makespan_after),
        "tardiness_after": float(tardiness_after),
        "instability_after": float(instability_after),
        "weighted_objective_after": float(weighted_objective_after),
        "changed_op_ratio": float(changed_op_ratio),
        "changed_machine_ratio": float(changed_machine_ratio),
        "mean_abs_start_time_deviation": float(mean_abs_start_time_deviation),
        "status": status,
    }
    if extra:
        row.update(extra)
    return row
