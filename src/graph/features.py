from __future__ import annotations

from typing import Any

import torch

from src.data.schema import ProblemInstance
from src.events.schema import DynamicEvent
from src.scheduling.incumbent import IncumbentSchedule
from src.scheduling.state_builder import StateSnapshot


def _build_operation_features(
    instance: ProblemInstance,
    incumbent: IncumbentSchedule,
    snapshot: StateSnapshot,
) -> tuple[torch.Tensor, list[int]]:
    op_ids = snapshot.window_op_ids
    current_time = snapshot.current_time
    features: list[list[float]] = []
    for op_id in op_ids:
        op = instance.get_operation(op_id)
        job = instance.get_job(op.job_id)
        sched = incumbent.operations[op_id]
        predecessor_count = float(op.op_index)
        successor_count = float(len(job.operations) - op.op_index - 1)
        min_pt = op.min_processing_time
        max_pt = max(opt.processing_time for opt in op.options)
        original_start = sched.original_start_time if sched.original_start_time is not None else current_time
        original_end = sched.original_end_time if sched.original_end_time is not None else current_time
        machine_id = float(-1 if sched.machine_id is None else sched.machine_id)
        due_date = float(op.due_date if op.due_date is not None else current_time)
        remaining_job_min_pt = sum(x.min_processing_time for x in job.operations[op.op_index:])
        original_duration = max(0.0, original_end - original_start)
        time_to_start = original_start - current_time
        slack = due_date - max(current_time, original_end)
        num_jobs = max(1.0, float(instance.num_jobs))
        num_machines = max(1.0, float(instance.num_machines))
        max_due = max(due_date, 1.0)
        max_time_ref = max(original_end, due_date, current_time + 1.0)
        features.append(
            [
                float(op.job_id) / num_jobs,
                float(op.op_index) / max(1.0, float(len(job.operations))),
                predecessor_count / max(1.0, float(len(job.operations))),
                successor_count / max(1.0, float(len(job.operations))),
                float(len(op.options)) / num_machines,
                float(min_pt) / max_time_ref,
                float(max_pt) / max_time_ref,
                float(remaining_job_min_pt) / max_time_ref,
                float(time_to_start) / max_time_ref,
                float(slack) / max_due,
                float(op_id in snapshot.directly_impacted_op_ids),
                float(op_id in snapshot.active_op_ids),
                float(op_id in snapshot.completed_op_ids),
                float(machine_id) / num_machines,
                float(original_duration) / max_time_ref,
                1.0 if op_id in snapshot.window_op_ids else 0.0,
            ]
        )
    if not features:
        return torch.empty((0, 16), dtype=torch.float32), op_ids
    return torch.tensor(features, dtype=torch.float32), op_ids


def _build_machine_features(
    instance: ProblemInstance,
    incumbent: IncumbentSchedule,
    snapshot: StateSnapshot,
) -> tuple[torch.Tensor, list[int]]:
    features: list[list[float]] = []
    machine_ids: list[int] = []
    current_time = snapshot.current_time
    max_machines = max(1.0, float(instance.num_machines))
    max_time_ref = max(
        1.0,
        current_time,
        max((sched.end_time or 0.0) for sched in incumbent.operations.values()),
    )
    for machine in instance.machines:
        cal = incumbent.machine_calendars[machine.machine_id]
        breakdown_count = float(len(cal.breakdowns))
        next_block_start = min((b[0] for b in cal.breakdowns), default=0.0)
        next_block_end = min((b[1] for b in cal.breakdowns), default=0.0)
        current_load = sum(
            1.0
            for op_id in snapshot.window_op_ids
            if machine.machine_id in instance.get_operation(op_id).eligible_machine_ids
        )
        available_gap = cal.available_time - current_time
        features.append(
            [
                float(machine.machine_id) / max_machines,
                float(cal.available_time) / max_time_ref,
                breakdown_count,
                float(next_block_start - current_time) / max_time_ref,
                float(next_block_end - current_time) / max_time_ref,
                float(machine.machine_id in snapshot.affected_machine_ids),
                float(current_load) / max(1.0, float(len(snapshot.window_op_ids))),
                float(available_gap) / max_time_ref,
            ]
        )
        machine_ids.append(machine.machine_id)
    if not features:
        return torch.empty((0, 8), dtype=torch.float32), machine_ids
    return torch.tensor(features, dtype=torch.float32), machine_ids


def _build_event_features(event: DynamicEvent) -> torch.Tensor:
    event_type_to_idx = {
        "job_arrival": 0.0,
        "machine_breakdown": 1.0,
        "processing_time_perturbation": 2.0,
    }
    machine_id = float(event.payload.get("machine_id", -1))
    duration = float(event.payload.get("end_time", event.time) - event.payload.get("start_time", event.time))
    return torch.tensor(
        [[event_type_to_idx[event.event_type], float(event.time), machine_id, duration, 1.0, 0.0]],
        dtype=torch.float32,
    )


def build_graph_tensors(
    instance: ProblemInstance,
    incumbent: IncumbentSchedule,
    snapshot: StateSnapshot,
    event: DynamicEvent,
) -> dict[str, Any]:
    op_x, op_ids = _build_operation_features(instance, incumbent, snapshot)
    machine_x, machine_ids = _build_machine_features(instance, incumbent, snapshot)
    event_x = _build_event_features(event)
    return {
        "op_x": op_x,
        "machine_x": machine_x,
        "event_x": event_x,
        "op_ids": op_ids,
        "machine_ids": machine_ids,
        "snapshot": snapshot,
        "instance": instance,
        "incumbent": incumbent,
        "event": event,
    }
