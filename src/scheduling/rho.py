from __future__ import annotations

from typing import Any, Iterable

from src.data.schema import ProblemInstance
from src.scheduling.incumbent import IncumbentSchedule
from src.scheduling.state_builder import StateSnapshot


def _pending_ops(snapshot: StateSnapshot, op_ids: Iterable[int]) -> list[int]:
    immutable = set(snapshot.completed_op_ids) | set(snapshot.active_op_ids)
    return sorted(int(op_id) for op_id in op_ids if int(op_id) not in immutable)


def assigned_processing_mass(
    *,
    instance: ProblemInstance,
    incumbent: IncumbentSchedule,
    op_ids: Iterable[int],
) -> float:
    total = 0.0
    for op_id in op_ids:
        schedule = incumbent.operations.get(int(op_id))
        if schedule is None:
            raise KeyError(f"Missing incumbent schedule for op_id={op_id}.")
        if schedule.machine_id is None:
            raise ValueError(f"Operation {op_id} has no current assigned machine.")
        total += float(instance.get_operation(int(op_id)).processing_time_on(int(schedule.machine_id)))
    return total


def compute_rho_descriptors(
    *,
    instance: ProblemInstance,
    incumbent: IncumbentSchedule,
    snapshot: StateSnapshot,
    makespan_before: float,
) -> dict[str, Any]:
    l_t = float(makespan_before)
    if l_t <= 0.0:
        raise ValueError(f"makespan_before must be positive to compute rho_t; got {makespan_before}.")
    pending_window_ops = _pending_ops(snapshot, snapshot.window_op_ids)
    pending_footprint_ops = _pending_ops(snapshot, snapshot.directly_impacted_op_ids)
    window_mass = assigned_processing_mass(
        instance=instance,
        incumbent=incumbent,
        op_ids=pending_window_ops,
    )
    footprint_mass = assigned_processing_mass(
        instance=instance,
        incumbent=incumbent,
        op_ids=pending_footprint_ops,
    )
    return {
        "rho_t": float(window_mass / l_t),
        "rho_t_foot": float(footprint_mass / l_t),
        "rho_window_work_mass": float(window_mass),
        "rho_footprint_work_mass": float(footprint_mass),
        "rho_l_t_makespan_proxy": l_t,
        "rho_pending_window_ops": len(pending_window_ops),
        "rho_pending_footprint_ops": len(pending_footprint_ops),
    }
