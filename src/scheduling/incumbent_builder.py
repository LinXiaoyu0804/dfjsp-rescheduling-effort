from __future__ import annotations

import hashlib
import json
import time
from copy import deepcopy
from typing import Any

from src.data.schema import ProblemInstance
from src.events.schema import DynamicEvent
from src.scheduling.incumbent import IncumbentSchedule, MachineCalendar, OperationSchedule
from src.scheduling.objectives import ObjectiveBreakdown, compute_weighted_objective
from src.scheduling.window import RollingWindow
from src.solver.base import RepairDecision
from src.solver.cp_repair_solver import CPRepairSolver


def build_greedy_incumbent(instance: ProblemInstance) -> IncumbentSchedule:
    machine_calendars = {m.machine_id: MachineCalendar(machine_id=m.machine_id, available_time=0.0) for m in instance.machines}
    operations: dict[int, OperationSchedule] = {}

    for job in instance.jobs:
        current_time = job.release_time
        for op in job.operations:
            best_option = min(op.options, key=lambda x: (x.processing_time, x.machine_id))
            start_time = max(current_time, machine_calendars[best_option.machine_id].available_time, op.release_time)
            end_time = start_time + best_option.processing_time
            operations[op.op_global_id] = OperationSchedule(
                op_global_id=op.op_global_id,
                job_id=op.job_id,
                op_index=op.op_index,
                machine_id=best_option.machine_id,
                start_time=start_time,
                end_time=end_time,
                status="unstarted",
                original_start_time=start_time,
                original_end_time=end_time,
                original_machine_id=best_option.machine_id,
            )
            machine_calendars[best_option.machine_id].available_time = end_time
            current_time = end_time

    return IncumbentSchedule(operations=operations, machine_calendars=machine_calendars, current_time=0.0)


def rebuild_machine_calendars(instance: ProblemInstance, operations: dict[int, OperationSchedule]) -> dict[int, MachineCalendar]:
    calendars = {m.machine_id: MachineCalendar(machine_id=m.machine_id, available_time=0.0) for m in instance.machines}
    for schedule in operations.values():
        if schedule.machine_id is None or schedule.end_time is None:
            continue
        calendars[schedule.machine_id].available_time = max(calendars[schedule.machine_id].available_time, float(schedule.end_time))
    return calendars


def _build_static_subproblem(instance: ProblemInstance, incumbent: IncumbentSchedule) -> dict[str, Any]:
    all_op_ids = sorted(op.op_global_id for op in instance.iter_operations())
    window = RollingWindow(time=0.0, op_ids=all_op_ids, directly_impacted_op_ids=[], affected_machine_ids=[], triggering_event=None)
    event = DynamicEvent(event_id=-1, time=0.0, event_type="job_arrival", payload={"job_object": None})
    decision = RepairDecision(
        immutable_op_ids=[],
        kept_op_ids=[],
        released_op_ids=all_op_ids,
        metadata={"source": "offline_incumbent_builder"},
    )
    return {
        "instance": instance,
        "incumbent": incumbent,
        "window": window,
        "event": event,
        "decision": decision,
    }


def _normalize_incumbent(incumbent: IncumbentSchedule) -> IncumbentSchedule:
    for schedule in incumbent.operations.values():
        schedule.status = "unstarted"
        schedule.original_start_time = schedule.start_time
        schedule.original_end_time = schedule.end_time
        schedule.original_machine_id = schedule.machine_id
    incumbent.current_time = 0.0
    return incumbent


def build_offline_incumbent(
    instance: ProblemInstance,
    solver_cfg: dict[str, Any],
    offline_budget_sec: float = 60.0,
    objective_weights: dict[str, float] | None = None,
) -> tuple[IncumbentSchedule, dict[str, Any]]:
    greedy = build_greedy_incumbent(instance)
    effective_solver_cfg = deepcopy(solver_cfg)
    effective_solver_cfg["time_limit_sec"] = float(offline_budget_sec)
    effective_solver_cfg["fix_kept_operations"] = False
    effective_solver_cfg["num_workers"] = int(effective_solver_cfg.get("offline_num_workers", 1))
    effective_solver_cfg["random_seed"] = int(effective_solver_cfg.get("random_seed", 0))
    weights = deepcopy(objective_weights or effective_solver_cfg.get("objective_weights", {}))
    weights.setdefault("makespan", 1.0)
    weights.setdefault("tardiness", 0.0)
    weights["instability"] = 0.0
    effective_solver_cfg["objective_weights"] = weights

    start_time = time.perf_counter()
    result = CPRepairSolver(effective_solver_cfg).solve(_build_static_subproblem(instance, greedy))
    runtime_sec = time.perf_counter() - start_time

    if result.feasible and result.updated_operations:
        incumbent = deepcopy(greedy)
        incumbent.operations.update(result.updated_operations)
        incumbent.machine_calendars = rebuild_machine_calendars(instance, incumbent.operations)
        incumbent = _normalize_incumbent(incumbent)
        solver_status = result.solver_status
        fallback_used = False
    else:
        incumbent = _normalize_incumbent(greedy)
        solver_status = "greedy_fallback"
        fallback_used = True

    objective = compute_weighted_objective(instance, incumbent, weights)
    return incumbent, {
        "solver_status": solver_status,
        "solver_runtime_sec": runtime_sec,
        "fallback_used": fallback_used,
        "objective": objective,
        "objective_weights": weights,
    }


def incumbent_schedule_hash(incumbent: IncumbentSchedule) -> str:
    payload = [
        {
            "op_id": int(op_id),
            "machine": None if schedule.machine_id is None else int(schedule.machine_id),
            "start": None if schedule.start_time is None else round(float(schedule.start_time), 6),
            "end": None if schedule.end_time is None else round(float(schedule.end_time), 6),
        }
        for op_id, schedule in sorted(incumbent.operations.items())
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _serialize_operation_schedule(schedule: OperationSchedule) -> dict[str, Any]:
    return {
        "machine": None if schedule.machine_id is None else int(schedule.machine_id),
        "start": None if schedule.start_time is None else float(schedule.start_time),
        "end": None if schedule.end_time is None else float(schedule.end_time),
    }


def serialize_objective(objective: ObjectiveBreakdown) -> dict[str, Any]:
    return {
        "makespan": float(objective.makespan),
        "tardiness": float(objective.total_tardiness),
        "instability": float(objective.instability),
        "weighted_objective": float(objective.weighted_sum),
    }


def serialize_incumbent(
    instance_id: str,
    instance_path: str,
    seed: int,
    incumbent: IncumbentSchedule,
    solver_name: str,
    offline_budget_sec: float,
    objective: ObjectiveBreakdown,
    solver_status: str,
    solver_runtime_sec: float,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    machine_summary: dict[str, dict[str, Any]] = {}
    for machine_id, calendar in sorted(incumbent.machine_calendars.items()):
        assigned_count = sum(1 for schedule in incumbent.operations.values() if schedule.machine_id == machine_id)
        machine_summary[str(machine_id)] = {
            "assigned_operation_count": assigned_count,
            "available_time": float(calendar.available_time),
            "breakdowns": [[float(start), float(end)] for start, end in calendar.breakdowns],
        }

    payload = {
        "instance_id": instance_id,
        "instance_path": instance_path,
        "seed": int(seed),
        "solver": solver_name,
        "offline_budget_sec": float(offline_budget_sec),
        "schedule": {
            str(op_id): _serialize_operation_schedule(schedule)
            for op_id, schedule in sorted(incumbent.operations.items())
        },
        "objective": serialize_objective(objective),
        "solver_status": solver_status,
        "solver_runtime_sec": float(solver_runtime_sec),
        "machine_summary": machine_summary,
        "schedule_hash": incumbent_schedule_hash(incumbent),
    }
    if extra_metadata:
        payload["metadata"] = deepcopy(extra_metadata)
    return payload


def load_incumbent_schedule(instance: ProblemInstance, data: dict[str, Any]) -> IncumbentSchedule:
    operations: dict[int, OperationSchedule] = {}
    for op_id_str, schedule_data in data.get("schedule", {}).items():
        op_id = int(op_id_str)
        operation = instance.get_operation(op_id)
        machine_id = schedule_data.get("machine")
        start_time = schedule_data.get("start")
        end_time = schedule_data.get("end")
        operations[op_id] = OperationSchedule(
            op_global_id=op_id,
            job_id=operation.job_id,
            op_index=operation.op_index,
            machine_id=None if machine_id is None else int(machine_id),
            start_time=None if start_time is None else float(start_time),
            end_time=None if end_time is None else float(end_time),
            status="unstarted",
            original_start_time=None if start_time is None else float(start_time),
            original_end_time=None if end_time is None else float(end_time),
            original_machine_id=None if machine_id is None else int(machine_id),
        )
    calendars = rebuild_machine_calendars(instance, operations)
    return IncumbentSchedule(operations=operations, machine_calendars=calendars, current_time=0.0)
