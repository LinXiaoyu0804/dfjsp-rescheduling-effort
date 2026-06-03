from __future__ import annotations

from dataclasses import dataclass

from src.data.schema import ProblemInstance
from src.scheduling.incumbent import IncumbentSchedule


@dataclass(slots=True)
class ObjectiveBreakdown:
    makespan: float
    total_tardiness: float
    instability: float
    weighted_sum: float


def compute_makespan(incumbent: IncumbentSchedule) -> float:
    end_times = [op.end_time for op in incumbent.operations.values() if op.end_time is not None]
    return max(end_times) if end_times else 0.0


def compute_total_tardiness(instance: ProblemInstance, incumbent: IncumbentSchedule) -> float:
    tardiness = 0.0
    for job in instance.jobs:
        last_end = 0.0
        for op in job.operations:
            sched = incumbent.operations.get(op.op_global_id)
            if sched is not None and sched.end_time is not None:
                last_end = max(last_end, sched.end_time)
        due_date = job.due_date if job.due_date is not None else last_end
        tardiness += max(0.0, last_end - due_date)
    return tardiness


def compute_instability(incumbent: IncumbentSchedule) -> float:
    instability = 0.0
    for sched in incumbent.operations.values():
        if sched.start_time is not None and sched.original_start_time is not None:
            original_duration = 1.0
            if sched.original_end_time is not None:
                original_duration = max(1.0, float(sched.original_end_time) - float(sched.original_start_time))
            instability += abs(float(sched.start_time) - float(sched.original_start_time)) / original_duration
        if (
            sched.machine_id is not None
            and sched.original_machine_id is not None
            and sched.machine_id != sched.original_machine_id
        ):
            instability += 2.0
    return instability


def compute_weighted_objective(
    instance: ProblemInstance,
    incumbent: IncumbentSchedule,
    weights: dict[str, float],
) -> ObjectiveBreakdown:
    makespan = compute_makespan(incumbent)
    tardiness = compute_total_tardiness(instance, incumbent)
    instability = compute_instability(incumbent)
    weighted_sum = (
        weights.get("makespan", 1.0) * makespan
        + weights.get("tardiness", 1.0) * tardiness
        + weights.get("instability", 0.0) * instability
    )
    return ObjectiveBreakdown(makespan, tardiness, instability, weighted_sum)
