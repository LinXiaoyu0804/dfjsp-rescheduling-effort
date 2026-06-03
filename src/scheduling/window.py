from __future__ import annotations

from dataclasses import dataclass, field

from src.data.schema import ProblemInstance
from src.events.schema import DynamicEvent
from src.scheduling.incumbent import IncumbentSchedule


@dataclass(slots=True)
class RollingWindow:
    time: float
    op_ids: list[int]
    directly_impacted_op_ids: list[int] = field(default_factory=list)
    affected_machine_ids: list[int] = field(default_factory=list)
    triggering_event: DynamicEvent | None = None


def identify_directly_impacted_operations(
    instance: ProblemInstance,
    incumbent: IncumbentSchedule,
    event: DynamicEvent,
) -> tuple[list[int], list[int]]:
    impacted_ops: list[int] = []
    affected_machine_ids: list[int] = []

    if event.event_type == "compound":
        for subevent in event.payload.get("subevents", []):
            sub_impacted, sub_machines = identify_directly_impacted_operations(instance, incumbent, subevent)
            impacted_ops.extend(sub_impacted)
            affected_machine_ids.extend(sub_machines)
        return sorted(set(impacted_ops)), sorted(set(affected_machine_ids))

    if event.event_type == "machine_breakdown":
        machine_id = int(event.payload["machine_id"])
        affected_machine_ids.append(machine_id)
        for op_id in incumbent.unfinished_ops():
            sched = incumbent.get(op_id)
            if sched.machine_id == machine_id or (
                sched.machine_id is None and machine_id in instance.get_operation(op_id).eligible_machine_ids
            ):
                impacted_ops.append(op_id)
    elif event.event_type == "job_arrival":
        job_object = event.payload.get("job_object")
        if job_object is not None:
            impacted_ops.extend(op.op_global_id for op in job_object.operations)

    return impacted_ops, affected_machine_ids


def build_rolling_window(
    instance: ProblemInstance,
    incumbent: IncumbentSchedule,
    event: DynamicEvent,
    horizon: float,
    max_window_ops: int,
    include_downstream_successors: bool = False,
) -> RollingWindow:
    directly_impacted, affected_machine_ids = identify_directly_impacted_operations(instance, incumbent, event)
    candidates: list[tuple[float, int]] = []
    current_time = event.time

    for op_id in incumbent.unfinished_ops():
        sched = incumbent.get(op_id)
        reference_start = sched.start_time
        if reference_start is None:
            reference_start = sched.original_start_time
        if reference_start is None:
            reference_start = current_time
        if current_time <= reference_start <= current_time + horizon:
            candidates.append((reference_start, op_id))

    for op_id in directly_impacted:
        sched = incumbent.operations.get(op_id)
        reference = current_time if sched is None else (sched.start_time or sched.original_start_time or current_time)
        candidates.append((reference, op_id))

    if include_downstream_successors:
        for op_id in directly_impacted:
            op = instance.get_operation(op_id)
            job = instance.get_job(op.job_id)
            for successor in job.operations[op.op_index + 1 :]:
                sched = incumbent.operations.get(successor.op_global_id)
                reference = current_time if sched is None else (sched.start_time or sched.original_start_time or current_time)
                candidates.append((reference, successor.op_global_id))

    deduped = sorted(set(candidates), key=lambda x: (x[0], x[1]))
    window_op_ids = [op_id for _, op_id in deduped[:max_window_ops]]
    return RollingWindow(
        time=current_time,
        op_ids=window_op_ids,
        directly_impacted_op_ids=sorted(set(directly_impacted)),
        affected_machine_ids=sorted(set(affected_machine_ids)),
        triggering_event=event,
    )
