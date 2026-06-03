from __future__ import annotations

from src.scheduling.objectives import compute_instability, compute_makespan, compute_total_tardiness


def _candidate_schedules(incumbent, op_ids: list[int] | None = None) -> list:
    if op_ids is None:
        schedules = list(incumbent.operations.values())
    else:
        schedules = [incumbent.operations[op_id] for op_id in op_ids if op_id in incumbent.operations]
    return [
        schedule
        for schedule in schedules
        if schedule.original_start_time is not None or schedule.original_end_time is not None or schedule.original_machine_id is not None
    ]


def _operation_changed(schedule) -> bool:
    return (
        schedule.start_time != schedule.original_start_time
        or schedule.end_time != schedule.original_end_time
        or schedule.machine_id != schedule.original_machine_id
    )


def compute_changed_operation_ratio(incumbent, op_ids: list[int] | None = None) -> float:
    schedules = _candidate_schedules(incumbent, op_ids=op_ids)
    if not schedules:
        return 0.0
    changed = sum(1 for schedule in schedules if _operation_changed(schedule))
    return changed / len(schedules)


def compute_changed_machine_ratio(incumbent, op_ids: list[int] | None = None) -> float:
    schedules = _candidate_schedules(incumbent, op_ids=op_ids)
    if not schedules:
        return 0.0
    changed = sum(1 for schedule in schedules if schedule.machine_id != schedule.original_machine_id)
    return changed / len(schedules)


def compute_mean_absolute_start_time_deviation(incumbent, op_ids: list[int] | None = None) -> float:
    schedules = _candidate_schedules(incumbent, op_ids=op_ids)
    comparable = [
        abs(float(schedule.start_time) - float(schedule.original_start_time))
        for schedule in schedules
        if schedule.start_time is not None and schedule.original_start_time is not None
    ]
    if not comparable:
        return 0.0
    return sum(comparable) / len(comparable)


def compute_instability_components(incumbent) -> tuple[float, float]:
    start_time_displacement = 0.0
    machine_reassignment = 0.0
    for sched in incumbent.operations.values():
        if sched.start_time is not None and sched.original_start_time is not None:
            start_time_displacement += abs(float(sched.start_time) - float(sched.original_start_time))
        if (
            sched.machine_id is not None
            and sched.original_machine_id is not None
            and sched.machine_id != sched.original_machine_id
        ):
            machine_reassignment += 1.0
    return start_time_displacement, machine_reassignment


def evaluate_schedule(instance, incumbent, runtime_sec: float, changed_op_ratio: float, changed_machine_ratio: float, feasible: bool) -> dict[str, float]:
    return {
        "makespan": compute_makespan(incumbent),
        "total_tardiness": compute_total_tardiness(instance, incumbent),
        "instability": compute_instability(incumbent),
        "mean_abs_start_time_deviation": compute_mean_absolute_start_time_deviation(incumbent),
        "runtime": runtime_sec,
        "changed_operation_ratio": changed_op_ratio,
        "changed_machine_ratio": changed_machine_ratio,
        "feasibility_rate": 1.0 if feasible else 0.0,
    }
