from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.data.schema import ProblemInstance
from src.motifs.schema import MotifCandidate, MotifMember, finalize_motif_ids


@dataclass(slots=True)
class SnapshotView:
    episode_id: str
    event_id: str
    tau: float
    event_type: str
    completed_ops: set[int]
    active_ops: set[int]
    unfinished_ops: set[int]
    window_ops: list[int]
    forced_release_ops: set[int]
    affected_ops: list[int]
    affected_machines: list[int]
    event_payload: dict[str, Any]
    assignments: dict[int, dict[str, Any]]


def snapshot_view_from_row(row: dict[str, Any]) -> SnapshotView:
    assignments = {int(op_id): value for op_id, value in row.get("incumbent_assignments", {}).items()}
    event_context = row.get("event_context", {})
    return SnapshotView(
        episode_id=str(row["episode_id"]),
        event_id=str(row["event_id"]),
        tau=float(row["tau"]),
        event_type=str(event_context.get("type", "")),
        completed_ops=set(int(op_id) for op_id in row.get("completed_ops", [])),
        active_ops=set(int(op_id) for op_id in row.get("active_ops", [])),
        unfinished_ops=set(int(op_id) for op_id in row.get("unfinished_ops", [])),
        window_ops=[int(op_id) for op_id in row.get("window_ops", [])],
        forced_release_ops=set(int(op_id) for op_id in row.get("forced_release_ops", [])),
        affected_ops=[int(op_id) for op_id in event_context.get("affected_ops", [])],
        affected_machines=[int(machine_id) for machine_id in event_context.get("affected_machines", [])],
        event_payload=dict(event_context.get("payload", {})),
        assignments=assignments,
    )


def _scheduled_window(instance: ProblemInstance, snapshot: SnapshotView, op_id: int) -> tuple[float, float]:
    assignment = snapshot.assignments.get(op_id, {})
    start = assignment.get("start")
    end = assignment.get("end")
    if start is None:
        start = snapshot.tau
    if end is None:
        end = float(start) + float(instance.get_operation(op_id).min_processing_time)
    return float(start), float(end)


def _slack(instance: ProblemInstance, snapshot: SnapshotView, op_id: int) -> float:
    operation = instance.get_operation(op_id)
    end = _scheduled_window(instance, snapshot, op_id)[1]
    due_date = float(operation.due_date if operation.due_date is not None else end)
    return due_date - max(snapshot.tau, end)


def _inverse_slack(instance: ProblemInstance, snapshot: SnapshotView, op_id: int) -> float:
    slack = _slack(instance, snapshot, op_id)
    if slack <= 0.0:
        return 1.0 + abs(slack)
    return 1.0 / max(1.0, slack)


def _downstream_criticality(instance: ProblemInstance, snapshot: SnapshotView, op_id: int) -> float:
    operation = instance.get_operation(op_id)
    job = instance.get_job(operation.job_id)
    remaining = [candidate for candidate in job.operations[operation.op_index + 1 :] if candidate.op_global_id in snapshot.unfinished_ops]
    if not remaining:
        return 0.0
    remaining_work = sum(candidate.min_processing_time for candidate in remaining)
    normalizer = max(1.0, sum(candidate.min_processing_time for candidate in job.operations))
    return float(remaining_work) / normalizer


def _overlap_risk(instance: ProblemInstance, snapshot: SnapshotView, op_id: int) -> float:
    operation = instance.get_operation(op_id)
    start, end = _scheduled_window(instance, snapshot, op_id)
    overlaps = 0
    total = 0
    op_eligible = set(operation.eligible_machine_ids)
    op_machine = snapshot.assignments.get(op_id, {}).get("machine")
    for other_op_id in snapshot.window_ops:
        if other_op_id == op_id:
            continue
        total += 1
        other = instance.get_operation(other_op_id)
        other_start, other_end = _scheduled_window(instance, snapshot, other_op_id)
        overlaps_in_time = not (other_end <= start or end <= other_start)
        shared_machine = bool(op_eligible & set(other.eligible_machine_ids))
        incumbent_machine_match = (
            op_machine is not None
            and snapshot.assignments.get(other_op_id, {}).get("machine") == op_machine
        )
        if overlaps_in_time and (shared_machine or incumbent_machine_match):
            overlaps += 1
    if total <= 0:
        return 0.0
    return overlaps / total


def operation_urgency(instance: ProblemInstance, snapshot: SnapshotView, op_id: int) -> tuple[float, dict[str, float]]:
    inverse_slack = _inverse_slack(instance, snapshot, op_id)
    downstream_criticality = _downstream_criticality(instance, snapshot, op_id)
    overlap_risk = _overlap_risk(instance, snapshot, op_id)
    urgency = 0.5 * inverse_slack + 0.3 * downstream_criticality + 0.2 * overlap_risk
    return urgency, {
        "inverse_slack": inverse_slack,
        "downstream_criticality": downstream_criticality,
        "incumbent_overlap_risk": overlap_risk,
    }


def _make_member(member_type: str, member_id: str | int, role: str) -> MotifMember:
    return MotifMember(member_type=member_type, member_id=str(member_id), role=role)


def _build_m1_flexibility_motifs(instance: ProblemInstance, snapshot: SnapshotView) -> list[MotifCandidate]:
    motifs: list[MotifCandidate] = []
    for op_id in snapshot.window_ops:
        operation = instance.get_operation(op_id)
        if len(operation.eligible_machine_ids) < 2:
            continue
        urgency, components = operation_urgency(instance, snapshot, op_id)
        motifs.append(
            MotifCandidate(
                family="M1",
                anchor_type="operation",
                anchor_id=str(op_id),
                operation_ids=[op_id],
                machine_ids=sorted(operation.eligible_machine_ids),
                members=[_make_member("operation", op_id, "anchor_operation")]
                + [_make_member("machine", machine_id, "candidate_machine") for machine_id in sorted(operation.eligible_machine_ids)],
                urgency_score=urgency,
                metadata={"urgency_components": components},
            )
        )
    return motifs


def _build_m2_contention_motifs(instance: ProblemInstance, snapshot: SnapshotView, cfg: dict[str, Any]) -> list[MotifCandidate]:
    motifs: list[MotifCandidate] = []
    k_contention = int(cfg.get("k_contention", 8))
    for machine in instance.machines:
        candidates: list[tuple[float, int, dict[str, float]]] = []
        for op_id in snapshot.window_ops:
            operation = instance.get_operation(op_id)
            if machine.machine_id not in operation.eligible_machine_ids:
                continue
            urgency, components = operation_urgency(instance, snapshot, op_id)
            candidates.append((urgency, op_id, components))
        if len(candidates) < 2:
            continue
        ranked = sorted(candidates, key=lambda item: (-item[0], item[1]))[:k_contention]
        motifs.append(
            MotifCandidate(
                family="M2",
                anchor_type="machine",
                anchor_id=str(machine.machine_id),
                operation_ids=[op_id for _, op_id, _ in ranked],
                machine_ids=[machine.machine_id],
                members=[_make_member("machine", machine.machine_id, "anchor_machine")]
                + [_make_member("operation", op_id, "competing_operation") for _, op_id, _ in ranked],
                urgency_score=sum(score for score, _, _ in ranked),
                metadata={
                    "ranked_operations": [
                        {"op_id": op_id, "urgency": score, "components": components}
                        for score, op_id, components in ranked
                    ]
                },
            )
        )
    return motifs


def _build_m3_propagation_motifs(instance: ProblemInstance, snapshot: SnapshotView, cfg: dict[str, Any]) -> list[MotifCandidate]:
    motifs: list[MotifCandidate] = []
    l_pre = int(cfg.get("l_pre", 2))
    l_suc = int(cfg.get("l_suc", 2))
    anchors = snapshot.affected_ops or snapshot.window_ops[: min(3, len(snapshot.window_ops))]
    for op_id in anchors:
        operation = instance.get_operation(op_id)
        job = instance.get_job(operation.job_id)
        unfinished_pre = [candidate.op_global_id for candidate in job.operations[: operation.op_index] if candidate.op_global_id in snapshot.unfinished_ops]
        unfinished_suc = [candidate.op_global_id for candidate in job.operations[operation.op_index + 1 :] if candidate.op_global_id in snapshot.unfinished_ops]
        selected_pre = unfinished_pre[-l_pre:]
        selected_suc = unfinished_suc[:l_suc]
        operation_ids = selected_pre + [op_id] + selected_suc
        machine_ids: set[int] = set()
        members = [_make_member("operation", op_id, "segment_anchor")]
        for predecessor_id in selected_pre:
            members.append(_make_member("operation", predecessor_id, "upstream_operation"))
        for successor_id in selected_suc:
            members.append(_make_member("operation", successor_id, "downstream_operation"))
        for member_op_id in operation_ids:
            machine = snapshot.assignments.get(member_op_id, {}).get("machine")
            if machine is not None:
                machine_ids.add(int(machine))
            machine_ids.update(instance.get_operation(member_op_id).eligible_machine_ids)
        for machine_id in sorted(machine_ids):
            members.append(_make_member("machine", machine_id, "support_machine"))
        urgency = sum(operation_urgency(instance, snapshot, member_op_id)[0] for member_op_id in operation_ids)
        motifs.append(
            MotifCandidate(
                family="M3",
                anchor_type="operation",
                anchor_id=str(op_id),
                operation_ids=operation_ids,
                machine_ids=sorted(machine_ids),
                members=members,
                urgency_score=urgency,
                metadata={"predecessor_count": len(selected_pre), "successor_count": len(selected_suc)},
            )
        )
    return motifs


def _job_arrival_prefix_ops(instance: ProblemInstance, snapshot: SnapshotView) -> list[int]:
    new_job = snapshot.event_payload.get("new_job")
    if not isinstance(new_job, dict):
        return []
    operations = [int(operation["op_global_id"]) for operation in new_job.get("operations", [])]
    return operations[: min(2, len(operations))]


def _build_m4_disturbance_motifs(instance: ProblemInstance, snapshot: SnapshotView, cfg: dict[str, Any]) -> list[MotifCandidate]:
    k_event = int(cfg.get("k_event", 12))
    candidate_ops: set[int] = set(snapshot.affected_ops)
    for op_id in snapshot.affected_ops:
        operation = instance.get_operation(op_id)
        job = instance.get_job(operation.job_id)
        if operation.op_index + 1 < len(job.operations):
            successor = job.operations[operation.op_index + 1]
            if successor.op_global_id in snapshot.unfinished_ops:
                candidate_ops.add(successor.op_global_id)
    candidate_ops.update(_job_arrival_prefix_ops(instance, snapshot))
    ranked_ops = sorted(
        (
            operation_urgency(instance, snapshot, op_id)[0],
            op_id,
        )
        for op_id in candidate_ops
    )
    selected_ops = [op_id for _, op_id in sorted(ranked_ops, key=lambda item: (-item[0], item[1]))[:k_event]]
    machine_ids: set[int] = set(snapshot.affected_machines)
    members = [_make_member("event", snapshot.event_id, "event_anchor")]
    for op_id in selected_ops:
        role = "affected_operation" if op_id in snapshot.affected_ops else "spillover_operation"
        members.append(_make_member("operation", op_id, role))
        machine = snapshot.assignments.get(op_id, {}).get("machine")
        if machine is not None:
            machine_ids.add(int(machine))
    for machine_id in sorted(machine_ids):
        members.append(_make_member("machine", machine_id, "support_machine"))
    urgency = sum(operation_urgency(instance, snapshot, op_id)[0] for op_id in selected_ops)
    return [
        MotifCandidate(
            family="M4",
            anchor_type="event",
            anchor_id=snapshot.event_id,
            operation_ids=selected_ops,
            machine_ids=sorted(machine_ids),
            members=members,
            urgency_score=urgency,
            metadata={"event_type": snapshot.event_type},
        )
    ]


def deduplicate_motifs(motifs: list[MotifCandidate]) -> list[MotifCandidate]:
    deduped: list[MotifCandidate] = []
    for motif in sorted(motifs, key=lambda item: (item.family, item.anchor_type, item.anchor_id, -item.urgency_score)):
        replaced = False
        op_set = set(motif.operation_ids)
        for index, existing in enumerate(deduped):
            if motif.family != existing.family or motif.anchor_type != existing.anchor_type:
                continue
            existing_ops = set(existing.operation_ids)
            union = op_set | existing_ops
            jaccard = 1.0 if not union else len(op_set & existing_ops) / len(union)
            if jaccard < 0.8:
                continue
            better = motif.urgency_score > existing.urgency_score
            tie_break = motif.canonical_hash() < existing.canonical_hash()
            if better or (motif.urgency_score == existing.urgency_score and tie_break):
                deduped[index] = motif
            replaced = True
            break
        if not replaced:
            deduped.append(motif)
    return deduped


def truncate_motifs(motifs: list[MotifCandidate], snapshot: SnapshotView, cfg: dict[str, Any]) -> list[MotifCandidate]:
    cap = int(cfg.get("K_motif_pool", 48))
    if len(motifs) <= cap:
        return motifs
    forced_or_event = [
        motif
        for motif in motifs
        if motif.family == "M4" or bool(set(motif.operation_ids) & snapshot.forced_release_ops)
    ]
    forced_ids = {motif.canonical_hash() for motif in forced_or_event}
    ranked_rest = [
        motif
        for motif in sorted(
            motifs,
            key=lambda item: (-item.urgency_score, item.family, item.anchor_type, item.anchor_id, item.canonical_hash()),
        )
        if motif.canonical_hash() not in forced_ids
    ]
    kept = forced_or_event + ranked_rest[: max(0, cap - len(forced_or_event))]
    return kept[:cap]


def apply_precedence_closure(
    operation_ids: list[int],
    instance: ProblemInstance,
    unfinished_ops: set[int],
    backward_depth: int = 1,
    forward_depth: int = 0,
) -> list[int]:
    closure = set(int(op_id) for op_id in operation_ids)
    frontier = set(closure)
    for _ in range(max(0, backward_depth)):
        next_frontier: set[int] = set()
        for op_id in frontier:
            operation = instance.get_operation(op_id)
            job = instance.get_job(operation.job_id)
            if operation.op_index <= 0:
                continue
            predecessor = job.operations[operation.op_index - 1].op_global_id
            if predecessor in unfinished_ops and predecessor not in closure:
                next_frontier.add(predecessor)
        closure.update(next_frontier)
        frontier = next_frontier

    frontier = set(closure)
    for _ in range(max(0, forward_depth)):
        next_frontier = set()
        for op_id in frontier:
            operation = instance.get_operation(op_id)
            job = instance.get_job(operation.job_id)
            if operation.op_index + 1 >= len(job.operations):
                continue
            successor = job.operations[operation.op_index + 1].op_global_id
            if successor in unfinished_ops and successor not in closure:
                next_frontier.add(successor)
        closure.update(next_frontier)
        frontier = next_frontier
    return sorted(closure)


def extract_candidate_motifs(instance: ProblemInstance, snapshot_row: dict[str, Any], motif_cfg: dict[str, Any] | None = None) -> list[MotifCandidate]:
    cfg = motif_cfg or {}
    snapshot = snapshot_view_from_row(snapshot_row)
    requested_families = cfg.get("families", ["M1", "M2", "M3", "M4"])
    families = {str(family).upper() for family in requested_families}
    motifs = []
    if "M1" in families:
        motifs.extend(_build_m1_flexibility_motifs(instance, snapshot))
    if "M2" in families:
        motifs.extend(_build_m2_contention_motifs(instance, snapshot, cfg))
    if "M3" in families:
        motifs.extend(_build_m3_propagation_motifs(instance, snapshot, cfg))
    if "M4" in families:
        motifs.extend(_build_m4_disturbance_motifs(instance, snapshot, cfg))
    motifs = deduplicate_motifs(motifs)
    motifs = truncate_motifs(motifs, snapshot, cfg)
    return finalize_motif_ids(motifs)
