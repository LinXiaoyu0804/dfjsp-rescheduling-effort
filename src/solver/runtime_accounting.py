from __future__ import annotations

import math
from typing import Any


def _coerce_positive_float(value: Any) -> float | None:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(resolved) or resolved <= 0.0:
        return None
    return resolved


def resolve_solver_runtime_accounting(
    *,
    raw_wall_time_sec: float,
    solver_metadata: dict[str, Any] | None,
    decision_metadata: dict[str, Any] | None,
    guard_multiplier: float = 1.1,
    guard_slack_sec: float = 0.5,
    anomaly_ratio_threshold: float = 8.0,
    anomaly_abs_slack_sec: float = 30.0,
) -> dict[str, Any]:
    """
    Convert raw runtime observations into a budget-aware runtime record.

    Why this exists:
    - OR-Tools exposes its own wall-time accounting.
    - host-side timers can occasionally be polluted by long sleeps / pauses,
      which can create pathological outliers unrelated to actual solver effort.
    - strict online-budget reporting should therefore prefer solver-reported
      time when host timing is clearly anomalous, and should flag genuine
      overtime as a budget violation.
    """

    solver_metadata = dict(solver_metadata or {})
    decision_metadata = dict(decision_metadata or {})

    raw_runtime_sec = max(0.0, float(raw_wall_time_sec))
    requested_time_limit_sec = _coerce_positive_float(
        decision_metadata.get("solver_time_limit_sec", solver_metadata.get("time_limit_sec"))
    )
    solver_wall_time_sec = _coerce_positive_float(solver_metadata.get("solver_wall_time_sec"))
    solver_user_time_sec = _coerce_positive_float(solver_metadata.get("solver_user_time_sec"))

    raw_timing_anomaly = False
    if solver_wall_time_sec is not None:
        raw_timing_anomaly = raw_runtime_sec > max(
            solver_wall_time_sec * max(1.0, float(anomaly_ratio_threshold)),
            solver_wall_time_sec + max(0.0, float(anomaly_abs_slack_sec)),
        )

    if solver_wall_time_sec is not None and raw_timing_anomaly:
        accounted_runtime_sec = solver_wall_time_sec
        accounting_source = "solver_wall_time"
    elif solver_wall_time_sec is not None:
        accounted_runtime_sec = max(raw_runtime_sec, solver_wall_time_sec)
        accounting_source = "max_raw_vs_solver_wall_time"
    else:
        accounted_runtime_sec = raw_runtime_sec
        accounting_source = "raw_wall_time"

    runtime_budget_cap_sec = None
    if requested_time_limit_sec is not None:
        runtime_budget_cap_sec = max(
            0.01,
            requested_time_limit_sec * max(1.0, float(guard_multiplier)) + max(0.0, float(guard_slack_sec)),
        )

    runtime_clipped = False
    budget_violation = False
    if runtime_budget_cap_sec is not None and accounted_runtime_sec > runtime_budget_cap_sec:
        accounted_runtime_sec = runtime_budget_cap_sec
        runtime_clipped = True
        budget_violation = True

    return {
        "runtime_sec": float(accounted_runtime_sec),
        "raw_wall_time_sec": float(raw_runtime_sec),
        "solver_wall_time_sec": None if solver_wall_time_sec is None else float(solver_wall_time_sec),
        "solver_user_time_sec": None if solver_user_time_sec is None else float(solver_user_time_sec),
        "requested_time_limit_sec": None
        if requested_time_limit_sec is None
        else float(requested_time_limit_sec),
        "runtime_budget_cap_sec": None
        if runtime_budget_cap_sec is None
        else float(runtime_budget_cap_sec),
        "runtime_accounting_source": accounting_source,
        "raw_timing_anomaly": bool(raw_timing_anomaly),
        "runtime_clipped": bool(runtime_clipped),
        "budget_violation": bool(budget_violation),
    }
