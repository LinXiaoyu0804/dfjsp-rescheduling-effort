from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

from src.baselines.full_reopt import full_reoptimization_decision
from src.baselines.heuristic_rh import heuristic_rh_decision
from src.data.schema import ProblemInstance
from src.events.schema import DynamicEvent
from src.events.serialization import serialize_event_payload
from src.motifs.extractors import extract_candidate_motifs
from src.scheduling.incumbent import IncumbentSchedule
from src.scheduling.state_builder import StateSnapshot
from src.solver.base import RepairDecision


IntensityLevel = Literal["L0", "L1", "L2", "L3"]


INTENSITY_DEFINITIONS: dict[str, dict[str, Any]] = {
    "L0": {
        "backend": "heuristic_rh_compatible_right_shift",
        "release_rule": "existing heuristic_rh release set for reproduction compatibility",
        "cp_backend": "same evaluation backend as heuristic_rh baseline",
    },
    "L1": {
        "backend": "cp_sat_repair",
        "release_rule": "L0 plus direct job predecessor/successor and incumbent-machine neighbors",
        "trust_region": {"backward_ratio": 0.05, "forward_ratio": 0.15},
    },
    "L2": {
        "backend": "cp_sat_repair",
        "release_rule": "L1 plus M3 propagation and M4 disturbance motif operation clusters",
        "trust_region": {"backward_ratio": 0.10, "forward_ratio": 0.30},
    },
    "L3": {
        "backend": "cp_sat_repair",
        "release_rule": "entire pending rolling-window release set, matching full_reoptimization",
        "trust_region": {"backward_ratio": 0.10, "forward_ratio": 0.30},
    },
}


@dataclass(slots=True)
class IntensityRepairPlan:
    level: str
    decision: RepairDecision
    forced_release_op_ids: list[int]
    released_op_ids_by_level: dict[str, list[int]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def normalize_intensity_level(level: str) -> str:
    normalized = str(level).strip().upper()
    if normalized not in INTENSITY_DEFINITIONS:
        raise ValueError(f"Unknown intensity level {level!r}; expected one of L0, L1, L2, L3.")
    return normalized


def forced_release_ops(snapshot: StateSnapshot) -> list[int]:
    immutable = set(snapshot.completed_op_ids) | set(snapshot.active_op_ids)
    return sorted(int(op_id) for op_id in snapshot.directly_impacted_op_ids if int(op_id) not in immutable)


def _eligible_window_ops(snapshot: StateSnapshot) -> set[int]:
    immutable = set(snapshot.completed_op_ids) | set(snapshot.active_op_ids)
    return {int(op_id) for op_id in snapshot.window_op_ids if int(op_id) not in immutable}


def _filter_release(snapshot: StateSnapshot, op_ids: Iterable[int]) -> list[int]:
    eligible = _eligible_window_ops(snapshot)
    return sorted(int(op_id) for op_id in op_ids if int(op_id) in eligible)


def _make_decision(
    snapshot: StateSnapshot,
    released_op_ids: Iterable[int],
    *,
    level: str,
    budget_sec: float,
    source: str,
    metadata: dict[str, Any] | None = None,
) -> RepairDecision:
    immutable = sorted(set(snapshot.completed_op_ids) | set(snapshot.active_op_ids))
    released = _filter_release(snapshot, released_op_ids)
    released_set = set(released)
    kept = [
        int(op_id)
        for op_id in snapshot.window_op_ids
        if int(op_id) not in released_set and int(op_id) not in set(immutable)
    ]
    decision_metadata = {
        "source": source,
        "intensity_level": level,
        "solver_time_limit_sec": float(budget_sec),
        "warm_start": True,
    }
    if metadata:
        decision_metadata.update(metadata)
    return RepairDecision(
        immutable_op_ids=immutable,
        kept_op_ids=kept,
        released_op_ids=released,
        metadata=decision_metadata,
    )


def _with_budget_and_level(
    decision: RepairDecision,
    *,
    level: str,
    budget_sec: float,
    metadata: dict[str, Any] | None = None,
) -> RepairDecision:
    updated_metadata = dict(decision.metadata)
    updated_metadata.update(
        {
            "intensity_level": level,
            "solver_time_limit_sec": float(budget_sec),
            "warm_start": True,
        }
    )
    if metadata:
        updated_metadata.update(metadata)
    return RepairDecision(
        immutable_op_ids=list(decision.immutable_op_ids),
        kept_op_ids=list(decision.kept_op_ids),
        released_op_ids=list(decision.released_op_ids),
        metadata=updated_metadata,
    )


def _same_job_neighbors(instance: ProblemInstance, op_id: int) -> list[int]:
    operation = instance.get_operation(int(op_id))
    job = instance.get_job(operation.job_id)
    neighbors: list[int] = []
    if operation.op_index > 0:
        neighbors.append(job.operations[operation.op_index - 1].op_global_id)
    if operation.op_index + 1 < len(job.operations):
        neighbors.append(job.operations[operation.op_index + 1].op_global_id)
    return neighbors


def _machine_neighbors(
    incumbent: IncumbentSchedule,
    snapshot: StateSnapshot,
    anchor_ops: Iterable[int],
    *,
    radius: int,
) -> list[int]:
    if radius <= 0:
        return []
    anchors = {int(op_id) for op_id in anchor_ops}
    by_machine: dict[int, list[tuple[float, int]]] = {}
    for op_id in snapshot.window_op_ids:
        sched = incumbent.operations.get(int(op_id))
        if sched is None or sched.machine_id is None:
            continue
        reference = sched.start_time
        if reference is None:
            reference = sched.original_start_time
        if reference is None:
            reference = snapshot.current_time
        by_machine.setdefault(int(sched.machine_id), []).append((float(reference), int(op_id)))

    selected: set[int] = set()
    for machine_ops in by_machine.values():
        ordered = [op_id for _, op_id in sorted(machine_ops, key=lambda item: (item[0], item[1]))]
        positions = {op_id: idx for idx, op_id in enumerate(ordered)}
        for anchor in anchors:
            if anchor not in positions:
                continue
            center = positions[anchor]
            for idx in range(max(0, center - radius), min(len(ordered), center + radius + 1)):
                selected.add(ordered[idx])
    return sorted(selected)


def _snapshot_row_from_context(
    *,
    instance_id: str,
    seed: int,
    episode_id: str,
    event_id: str,
    instance: ProblemInstance,
    incumbent: IncumbentSchedule,
    snapshot: StateSnapshot,
    event: DynamicEvent,
) -> dict[str, Any]:
    relevant_ops = sorted(set(snapshot.window_op_ids) | set(snapshot.directly_impacted_op_ids))
    assignments: dict[str, dict[str, float | int | None]] = {}
    for op_id in relevant_ops:
        schedule = incumbent.operations.get(int(op_id))
        if schedule is None:
            continue
        assignments[str(op_id)] = {
            "machine": None if schedule.machine_id is None else int(schedule.machine_id),
            "start": None if schedule.start_time is None else float(schedule.start_time),
            "end": None if schedule.end_time is None else float(schedule.end_time),
        }

    return {
        "episode_id": str(episode_id),
        "event_id": str(event_id),
        "instance_id": str(instance_id),
        "seed": int(seed),
        "tau": float(snapshot.current_time),
        "completed_ops": sorted(int(op_id) for op_id in snapshot.completed_op_ids),
        "active_ops": sorted(int(op_id) for op_id in snapshot.active_op_ids),
        "unfinished_ops": sorted(int(op_id) for op_id in snapshot.unfinished_op_ids),
        "window_ops": sorted(int(op_id) for op_id in snapshot.window_op_ids),
        "forced_release_ops": forced_release_ops(snapshot),
        "incumbent_assignments": assignments,
        "problem_size": int(instance.num_operations),
        "event_context": {
            "type": str(snapshot.triggering_event_type or event.event_type),
            "affected_ops": sorted(int(op_id) for op_id in snapshot.directly_impacted_op_ids),
            "affected_machines": sorted(int(machine_id) for machine_id in snapshot.affected_machine_ids),
            "payload": serialize_event_payload(event),
        },
    }


def _motif_cluster_ops(
    *,
    instance: ProblemInstance,
    incumbent: IncumbentSchedule,
    snapshot: StateSnapshot,
    event: DynamicEvent,
    motif_cfg: dict[str, Any] | None,
    instance_id: str,
    seed: int,
    episode_id: str,
    event_id: str,
) -> list[int]:
    snapshot_row = _snapshot_row_from_context(
        instance_id=instance_id,
        seed=seed,
        episode_id=episode_id,
        event_id=event_id,
        instance=instance,
        incumbent=incumbent,
        snapshot=snapshot,
        event=event,
    )
    selected: set[int] = set()
    for motif in extract_candidate_motifs(instance, snapshot_row, motif_cfg=motif_cfg or {}):
        if motif.family in {"M3", "M4"}:
            selected.update(int(op_id) for op_id in motif.operation_ids)
    return sorted(selected)


def build_intensity_release_sets(
    *,
    instance: ProblemInstance,
    incumbent: IncumbentSchedule,
    snapshot: StateSnapshot,
    event: DynamicEvent,
    motif_cfg: dict[str, Any] | None = None,
    include_machine_neighbors: bool = True,
    machine_neighbor_radius: int = 1,
    instance_id: str = "",
    seed: int = 0,
    episode_id: str = "",
    event_id: str = "",
) -> dict[str, list[int]]:
    l0 = _filter_release(snapshot, heuristic_rh_decision(snapshot).decision.released_op_ids)

    l1_seed: set[int] = set(l0) | set(_filter_release(snapshot, forced_release_ops(snapshot)))
    for op_id in list(l1_seed):
        for neighbor in _same_job_neighbors(instance, op_id):
            l1_seed.add(int(neighbor))
    if include_machine_neighbors:
        l1_seed.update(
            _machine_neighbors(
                incumbent,
                snapshot,
                l1_seed,
                radius=int(machine_neighbor_radius),
            )
        )
    l1 = sorted(set(l0) | set(_filter_release(snapshot, l1_seed)))

    motif_ops = _motif_cluster_ops(
        instance=instance,
        incumbent=incumbent,
        snapshot=snapshot,
        event=event,
        motif_cfg=motif_cfg,
        instance_id=instance_id,
        seed=seed,
        episode_id=episode_id,
        event_id=event_id,
    )
    l2 = sorted(set(l1) | set(_filter_release(snapshot, motif_ops)))

    l3 = _filter_release(snapshot, full_reoptimization_decision(snapshot).decision.released_op_ids)
    l2 = sorted(set(l2) & set(l3))
    l1 = sorted(set(l1) & set(l2))
    l0 = sorted(set(l0) & set(l1))

    return {"L0": l0, "L1": l1, "L2": l2, "L3": l3}


def _build_l1_release_sets(
    *,
    instance: ProblemInstance,
    incumbent: IncumbentSchedule,
    snapshot: StateSnapshot,
    include_machine_neighbors: bool = True,
    machine_neighbor_radius: int = 1,
) -> dict[str, list[int]]:
    l0 = _filter_release(snapshot, heuristic_rh_decision(snapshot).decision.released_op_ids)
    l1_seed: set[int] = set(l0) | set(_filter_release(snapshot, forced_release_ops(snapshot)))
    for op_id in list(l1_seed):
        for neighbor in _same_job_neighbors(instance, op_id):
            l1_seed.add(int(neighbor))
    if include_machine_neighbors:
        l1_seed.update(
            _machine_neighbors(
                incumbent,
                snapshot,
                l1_seed,
                radius=int(machine_neighbor_radius),
            )
        )
    l3 = _filter_release(snapshot, full_reoptimization_decision(snapshot).decision.released_op_ids)
    l1 = sorted((set(l0) | set(_filter_release(snapshot, l1_seed))) & set(l3))
    l0 = sorted(set(l0) & set(l1))
    return {"L0": l0, "L1": l1, "L3": l3}


def repair_at_intensity(
    incumbent: IncumbentSchedule,
    event: DynamicEvent,
    budget_sec: float,
    level: str,
    *,
    instance: ProblemInstance,
    snapshot: StateSnapshot,
    motif_cfg: dict[str, Any] | None = None,
    include_machine_neighbors: bool = True,
    machine_neighbor_radius: int = 1,
    instance_id: str = "",
    seed: int = 0,
    episode_id: str = "",
    event_id: str = "",
    compute_all_release_sets: bool = True,
) -> IntensityRepairPlan:
    normalized_level = normalize_intensity_level(level)
    if compute_all_release_sets or normalized_level == "L2":
        release_sets = build_intensity_release_sets(
            instance=instance,
            incumbent=incumbent,
            snapshot=snapshot,
            event=event,
            motif_cfg=motif_cfg,
            include_machine_neighbors=include_machine_neighbors,
            machine_neighbor_radius=machine_neighbor_radius,
            instance_id=instance_id,
            seed=seed,
            episode_id=episode_id,
            event_id=event_id,
        )
    elif normalized_level == "L1":
        release_sets = _build_l1_release_sets(
            instance=instance,
            incumbent=incumbent,
            snapshot=snapshot,
            include_machine_neighbors=include_machine_neighbors,
            machine_neighbor_radius=machine_neighbor_radius,
        )
    elif normalized_level == "L0":
        release_sets = {"L0": _filter_release(snapshot, heuristic_rh_decision(snapshot).decision.released_op_ids)}
    elif normalized_level == "L3":
        release_sets = {"L3": _filter_release(snapshot, full_reoptimization_decision(snapshot).decision.released_op_ids)}
    else:
        raise RuntimeError(f"Unhandled intensity level: {normalized_level}")

    if normalized_level == "L0":
        decision = _with_budget_and_level(
            heuristic_rh_decision(snapshot).decision,
            level="L0",
            budget_sec=budget_sec,
            metadata={"source": "intensity_ladder_L0_heuristic_rh"},
        )
    elif normalized_level == "L3":
        decision = _with_budget_and_level(
            full_reoptimization_decision(snapshot).decision,
            level="L3",
            budget_sec=budget_sec,
            metadata={"source": "intensity_ladder_L3_full_reoptimization"},
        )
    else:
        trust = INTENSITY_DEFINITIONS[normalized_level].get("trust_region", {})
        decision = _make_decision(
            snapshot,
            release_sets[normalized_level],
            level=normalized_level,
            budget_sec=budget_sec,
            source=f"intensity_ladder_{normalized_level}",
            metadata={
                "trust_region_backward_ratio": float(trust.get("backward_ratio", 0.1)),
                "trust_region_forward_ratio": float(trust.get("forward_ratio", 0.3)),
            },
        )

    return IntensityRepairPlan(
        level=normalized_level,
        decision=decision,
        forced_release_op_ids=forced_release_ops(snapshot),
        released_op_ids_by_level=release_sets,
        metadata={
            "intensity_definition": INTENSITY_DEFINITIONS[normalized_level],
            "monotone_release_counts": {key: len(value) for key, value in release_sets.items()},
        },
    )
