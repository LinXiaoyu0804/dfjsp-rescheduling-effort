from __future__ import annotations

import random
from copy import deepcopy
from typing import Any

from src.data.schema import Job, Operation, OperationOption, ProblemInstance
from src.events.schema import DynamicEvent


RHO_INTENSITY_PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    "R1": {
        "num_epochs": 4,
        "earliest_ratio": 0.08,
        "latest_ratio": 0.40,
        "arrivals_per_epoch": 1,
        "breakdowns_per_epoch": 1,
        "bottleneck_breakdowns": True,
        "duration_min_ratio": 0.05,
        "duration_max_ratio": 0.12,
    },
    "R2": {
        "num_epochs": 4,
        "earliest_ratio": 0.05,
        "latest_ratio": 0.30,
        "arrivals_per_epoch": 2,
        "breakdowns_per_epoch": 1,
        "bottleneck_breakdowns": True,
        "duration_min_ratio": 0.08,
        "duration_max_ratio": 0.18,
    },
    "R3": {
        "num_epochs": 3,
        "earliest_ratio": 0.03,
        "latest_ratio": 0.22,
        "arrivals_per_epoch": 2,
        "breakdowns_per_epoch": 2,
        "bottleneck_breakdowns": True,
        "duration_min_ratio": 0.15,
        "duration_max_ratio": 0.30,
    },
    "R4": {
        "num_epochs": 3,
        "earliest_ratio": 0.02,
        "latest_ratio": 0.16,
        "arrivals_per_epoch": 3,
        "breakdowns_per_epoch": 3,
        "bottleneck_breakdowns": True,
        "duration_min_ratio": 0.25,
        "duration_max_ratio": 0.45,
    },
    "R5": {
        "num_epochs": 2,
        "earliest_ratio": 0.01,
        "latest_ratio": 0.10,
        "arrivals_per_epoch": 4,
        "breakdowns_per_epoch": 4,
        "bottleneck_breakdowns": True,
        "duration_min_ratio": 0.40,
        "duration_max_ratio": 0.70,
    },
}


def _copy_job_with_new_ids(template_job: Job, new_job_id: int, next_global_op_id: int, release_time: float) -> Job:
    operations: list[Operation] = []
    cursor = next_global_op_id
    for op in template_job.operations:
        options = [OperationOption(machine_id=opt.machine_id, processing_time=opt.processing_time) for opt in op.options]
        operations.append(
            Operation(
                op_global_id=cursor,
                job_id=new_job_id,
                op_index=op.op_index,
                options=options,
                release_time=release_time,
                due_date=template_job.due_date,
            )
        )
        cursor += 1
    return Job(
        job_id=new_job_id,
        operations=operations,
        release_time=release_time,
        due_date=template_job.due_date,
        name=f"dynamic_job_{new_job_id}",
    )


def estimate_nominal_makespan(instance: ProblemInstance) -> float:
    """
    Lightweight nominal makespan estimate used only for relative event timing.

    Implementation assumption:
    Use the same greedy machine-availability logic as the initial incumbent builder
    to derive a stable instance-scale time reference.
    """
    machine_available = {m.machine_id: 0.0 for m in instance.machines}
    last_end = 0.0
    for job in instance.jobs:
        current_time = job.release_time
        for op in job.operations:
            best_option = min(op.options, key=lambda x: (x.processing_time, x.machine_id))
            start_time = max(current_time, machine_available[best_option.machine_id], op.release_time)
            end_time = start_time + best_option.processing_time
            machine_available[best_option.machine_id] = end_time
            current_time = end_time
            last_end = max(last_end, end_time)
    return last_end


def _resolve_time_window(cfg: dict[str, Any], nominal_makespan: float) -> tuple[float, float]:
    time_mode = cfg.get("time_mode", "absolute")
    if time_mode == "relative_to_nominal_makespan":
        earliest = float(cfg.get("earliest_ratio", 0.05)) * nominal_makespan
        latest = float(cfg.get("latest_ratio", 0.25)) * nominal_makespan
        return earliest, latest
    earliest = float(cfg.get("earliest_time", 0.0))
    latest = float(cfg.get("latest_time", 0.0))
    return earliest, latest


def _resolve_duration_window(cfg: dict[str, Any], nominal_makespan: float) -> tuple[float, float]:
    duration_mode = cfg.get("duration_mode", "absolute")
    if duration_mode == "relative_to_nominal_makespan":
        min_duration = float(cfg.get("min_duration_ratio", 0.01)) * nominal_makespan
        max_duration = float(cfg.get("max_duration_ratio", 0.05)) * nominal_makespan
        return min_duration, max_duration
    min_duration = float(cfg.get("min_duration", 1.0))
    max_duration = float(cfg.get("max_duration", 1.0))
    return min_duration, max_duration


def _profile_name(event_cfg: dict[str, Any]) -> str | None:
    request = event_cfg.get("rho_intensity_profile", event_cfg.get("rho_boundary_profile"))
    if request is None:
        return None
    if isinstance(request, str):
        return request.strip().upper()
    if isinstance(request, dict):
        if not bool(request.get("enabled", True)):
            return None
        return str(request.get("label", request.get("name", ""))).strip().upper()
    raise TypeError(f"Unsupported rho intensity profile request: {type(request)!r}")


def _profile_config(event_cfg: dict[str, Any], profile_name: str) -> dict[str, Any]:
    if profile_name in {"R0", "STANDARD"}:
        return {}
    if profile_name not in RHO_INTENSITY_PROFILE_DEFAULTS:
        raise ValueError(f"Unknown rho intensity profile {profile_name!r}; expected R0-R5.")
    cfg = dict(RHO_INTENSITY_PROFILE_DEFAULTS[profile_name])
    overrides = event_cfg.get("rho_intensity_profiles", {}).get(profile_name, {})
    if overrides:
        cfg.update(dict(overrides))
    return cfg


def _bottleneck_machine_order(instance: ProblemInstance, incumbent: Any | None, current_time: float) -> list[int]:
    loads = {machine.machine_id: 0.0 for machine in instance.machines}
    if incumbent is not None:
        for sched in getattr(incumbent, "operations", {}).values():
            machine_id = getattr(sched, "machine_id", None)
            start_time = getattr(sched, "start_time", None)
            end_time = getattr(sched, "end_time", None)
            if machine_id is None or start_time is None or end_time is None:
                continue
            if float(end_time) <= float(current_time):
                continue
            loads[int(machine_id)] = loads.get(int(machine_id), 0.0) + max(
                0.0,
                float(end_time) - max(float(start_time), float(current_time)),
            )
    return [machine_id for machine_id, _ in sorted(loads.items(), key=lambda item: (-item[1], item[0]))]


def _make_profile_arrival(
    *,
    instance: ProblemInstance,
    rng: random.Random,
    release_time: float,
    event_id: int,
    next_job_id: int,
    next_op_id: int,
) -> tuple[DynamicEvent, int, int]:
    template_job = rng.choice(instance.jobs)
    payload: dict[str, Any] = {
        "new_job_id": next_job_id,
        "release_time": release_time,
        "template_job_id": template_job.job_id,
        "job_object": _copy_job_with_new_ids(template_job, next_job_id, next_op_id, release_time),
    }
    return (
        DynamicEvent(event_id=event_id, time=release_time, event_type="job_arrival", payload=payload),
        next_job_id + 1,
        next_op_id + len(template_job.operations),
    )


def _generate_profiled_dynamic_events(
    instance: ProblemInstance,
    event_cfg: dict[str, Any],
    seed: int,
    *,
    profile_name: str,
    incumbent: Any | None = None,
) -> list[DynamicEvent]:
    profile = _profile_config(event_cfg, profile_name)
    if not profile:
        return []
    rng = random.Random(seed)
    reference_makespan = estimate_nominal_makespan(instance)
    if incumbent is not None:
        end_times = [
            float(getattr(schedule, "end_time"))
            for schedule in getattr(incumbent, "operations", {}).values()
            if getattr(schedule, "end_time", None) is not None
        ]
        if end_times:
            reference_makespan = max(end_times)

    earliest = float(profile["earliest_ratio"]) * reference_makespan
    latest = float(profile["latest_ratio"]) * reference_makespan
    if latest < earliest:
        raise ValueError(f"Invalid profile time window for {profile_name}: latest < earliest.")
    num_epochs = max(1, int(profile["num_epochs"]))
    arrivals_per_epoch = max(0, int(profile.get("arrivals_per_epoch", 0)))
    breakdowns_per_epoch = max(0, int(profile.get("breakdowns_per_epoch", 0)))
    min_duration = float(profile["duration_min_ratio"]) * reference_makespan
    max_duration = float(profile["duration_max_ratio"]) * reference_makespan

    next_job_id = max(job.job_id for job in instance.jobs) + 1 if instance.jobs else 0
    next_op_id = max(op.op_global_id for op in instance.iter_operations()) + 1 if instance.num_operations > 0 else 0
    events: list[DynamicEvent] = []
    atomic_event_id = 0
    epoch_times = sorted(rng.uniform(earliest, latest) for _ in range(num_epochs))
    for epoch_index, event_time in enumerate(epoch_times):
        subevents: list[DynamicEvent] = []
        machine_order = _bottleneck_machine_order(instance, incumbent, event_time)
        machine_offset = rng.randrange(max(1, len(machine_order))) if machine_order else 0
        for breakdown_index in range(breakdowns_per_epoch):
            if machine_order:
                machine_id = machine_order[(machine_offset + breakdown_index) % len(machine_order)]
            else:
                machine_id = rng.randrange(instance.num_machines)
            duration = rng.uniform(min_duration, max_duration)
            subevents.append(
                DynamicEvent(
                    event_id=atomic_event_id,
                    time=event_time,
                    event_type="machine_breakdown",
                    payload={
                        "machine_id": int(machine_id),
                        "start_time": float(event_time),
                        "end_time": float(event_time + duration),
                    },
                )
            )
            atomic_event_id += 1
        for _ in range(arrivals_per_epoch):
            arrival, next_job_id, next_op_id = _make_profile_arrival(
                instance=instance,
                rng=rng,
                release_time=event_time,
                event_id=atomic_event_id,
                next_job_id=next_job_id,
                next_op_id=next_op_id,
            )
            subevents.append(arrival)
            atomic_event_id += 1
        if not subevents:
            continue
        events.append(
            DynamicEvent(
                event_id=epoch_index,
                time=event_time,
                event_type="compound",
                payload={
                    "subevents": subevents,
                    "profile": profile_name,
                    "epoch_index": epoch_index,
                },
            )
        )
    return sorted(events, key=lambda event: (event.time, event.event_id))


def generate_dynamic_events(
    instance: ProblemInstance,
    event_cfg: dict[str, Any],
    seed: int,
    incumbent: Any | None = None,
) -> list[DynamicEvent]:
    requested_profile = _profile_name(event_cfg)
    if requested_profile and requested_profile not in {"R0", "STANDARD"}:
        return _generate_profiled_dynamic_events(
            instance,
            event_cfg,
            seed,
            profile_name=requested_profile,
            incumbent=incumbent,
        )

    rng = random.Random(seed)
    events: list[DynamicEvent] = []
    event_id = 0
    nominal_makespan = estimate_nominal_makespan(instance)

    job_cfg = event_cfg.get("job_arrival", {})
    if job_cfg.get("enabled", False):
        num_new_jobs = int(job_cfg.get("num_new_jobs", 0))
        earliest, latest = _resolve_time_window(job_cfg, nominal_makespan)
        copied_job_template = bool(job_cfg.get("copied_job_template", True))
        next_job_id = max(job.job_id for job in instance.jobs) + 1 if instance.jobs else 0
        next_op_id = max(op.op_global_id for op in instance.iter_operations()) + 1 if instance.num_operations > 0 else 0
        for _ in range(num_new_jobs):
            release_time = rng.uniform(earliest, latest)
            template_job = rng.choice(instance.jobs) if copied_job_template and instance.jobs else None
            payload: dict[str, Any] = {"new_job_id": next_job_id, "release_time": release_time}
            if template_job is not None:
                payload["template_job_id"] = template_job.job_id
                payload["job_object"] = _copy_job_with_new_ids(template_job, next_job_id, next_op_id, release_time)
                next_op_id += len(template_job.operations)
            else:
                payload["template_job_id"] = None
                payload["job_object"] = None
            events.append(DynamicEvent(event_id=event_id, time=release_time, event_type="job_arrival", payload=payload))
            next_job_id += 1
            event_id += 1

    bd_cfg = event_cfg.get("machine_breakdown", {})
    if bd_cfg.get("enabled", False):
        num_breakdowns = int(bd_cfg.get("num_breakdowns", 0))
        earliest, latest = _resolve_time_window(bd_cfg, nominal_makespan)
        min_duration, max_duration = _resolve_duration_window(bd_cfg, nominal_makespan)
        for _ in range(num_breakdowns):
            start_time = rng.uniform(earliest, latest)
            duration = rng.uniform(min_duration, max_duration)
            machine_id = rng.randrange(instance.num_machines)
            events.append(
                DynamicEvent(
                    event_id=event_id,
                    time=start_time,
                    event_type="machine_breakdown",
                    payload={"machine_id": machine_id, "start_time": start_time, "end_time": start_time + duration},
                )
            )
            event_id += 1

    events.sort(key=lambda e: (e.time, e.event_id))
    return events


def materialize_job_arrival(instance: ProblemInstance, event: DynamicEvent) -> ProblemInstance:
    if event.event_type != "job_arrival":
        return instance
    new_instance = deepcopy(instance)
    job_object = event.payload.get("job_object")
    if job_object is None:
        raise ValueError("Job arrival event does not contain 'job_object'.")
    new_instance.jobs.append(job_object)
    return new_instance
