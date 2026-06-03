from __future__ import annotations

import math

from src.baselines.base import BaselineOutput
from src.solver.base import RepairDecision


def _remaining_job_work(instance, op_id: int) -> float:
    op = instance.get_operation(op_id)
    job = instance.get_job(op.job_id)
    return sum(job_op.min_processing_time for job_op in job.operations[op.op_index :])


def _critical_ratio(instance, snapshot, op_id: int) -> float:
    op = instance.get_operation(op_id)
    remaining_work = max(_remaining_job_work(instance, op_id), 1e-6)
    due_date = float(op.due_date) if op.due_date is not None else float(snapshot.current_time) + remaining_work
    return (due_date - float(snapshot.current_time)) / remaining_work


def _atc_priority(instance, snapshot, op_id: int, candidates: list[int], kappa: float = 2.0) -> float:
    op = instance.get_operation(op_id)
    proc_time = max(float(op.min_processing_time), 1e-6)
    due_date = op.due_date
    remaining_work = _remaining_job_work(instance, op_id)
    avg_proc = sum(instance.get_operation(candidate_id).min_processing_time for candidate_id in candidates) / max(1, len(candidates))
    if due_date is None:
        slack = remaining_work
    else:
        slack = max(float(due_date) - float(snapshot.current_time) - remaining_work, 0.0)
    return (1.0 / proc_time) * math.exp(-(slack / max(kappa * avg_proc, 1e-6)))


def release_candidates_from_snapshot(snapshot) -> list[int]:
    return [
        op_id
        for op_id in snapshot.window_op_ids
        if op_id not in snapshot.completed_op_ids and op_id not in snapshot.active_op_ids
    ]


def order_dispatching_candidates(instance, snapshot, rule: str = "SPT") -> list[int]:
    candidates = release_candidates_from_snapshot(snapshot)
    if rule.upper() == "SPT":
        return sorted(candidates, key=lambda op_id: instance.get_operation(op_id).min_processing_time)
    if rule.upper() == "MWKR":
        return sorted(
            candidates,
            key=lambda op_id: -_remaining_job_work(instance, op_id),
        )
    if rule.upper() == "EDD":
        return sorted(
            candidates,
            key=lambda op_id: (
                float(instance.get_operation(op_id).due_date) if instance.get_operation(op_id).due_date is not None else float("inf"),
                op_id,
            ),
        )
    if rule.upper() == "CR":
        return sorted(candidates, key=lambda op_id: (_critical_ratio(instance, snapshot, op_id), op_id))
    if rule.upper() == "ATC":
        return sorted(candidates, key=lambda op_id: (-_atc_priority(instance, snapshot, op_id, candidates), op_id))
    return sorted(candidates)


def dispatching_release_count(num_candidates: int) -> int:
    return min(num_candidates, max(1, num_candidates // 3))


def dispatching_release_decision(instance, incumbent, snapshot, rule: str = "SPT") -> BaselineOutput:
    ordered = order_dispatching_candidates(instance, snapshot, rule=rule)
    release = ordered[: dispatching_release_count(len(ordered))]
    keep = [op_id for op_id in snapshot.window_op_ids if op_id not in release]
    immutable = sorted(set(snapshot.completed_op_ids + snapshot.active_op_ids))
    return BaselineOutput(
        decision=RepairDecision(immutable_op_ids=immutable, kept_op_ids=keep, released_op_ids=release, metadata={"rule": rule}),
        name=f"dispatching_{rule.lower()}",
    )
