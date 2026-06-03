from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np
from scipy.optimize import nnls

from src.data.schema import ProblemInstance
from src.motifs.extractors import apply_precedence_closure


def _jaccard(values_a: set[int], values_b: set[int]) -> float:
    union = values_a | values_b
    if not union:
        return 1.0
    return len(values_a & values_b) / len(union)


def motif_overlap_similarity(motif_a: dict[str, Any], motif_b: dict[str, Any]) -> float:
    op_jaccard = _jaccard(set(motif_a.get("operation_ids", [])), set(motif_b.get("operation_ids", [])))
    machine_jaccard = _jaccard(set(motif_a.get("machine_ids", [])), set(motif_b.get("machine_ids", [])))
    same_family = 1.0 if motif_a.get("family") == motif_b.get("family") else 0.0
    same_anchor = 1.0 if (
        motif_a.get("anchor_type") == motif_b.get("anchor_type")
        and motif_a.get("anchor_id") == motif_b.get("anchor_id")
    ) else 0.0
    return 0.6 * op_jaccard + 0.2 * machine_jaccard + 0.1 * same_family + 0.1 * same_anchor


def group_motifs_by_overlap(
    motif_rows: list[dict[str, Any]],
    target_size: int = 4,
    max_size: int = 6,
    min_similarity: float = 0.15,
) -> list[dict[str, Any]]:
    if not motif_rows:
        return []

    ranked = sorted(
        motif_rows,
        key=lambda row: (
            -float(row.get("urgency_score", 0.0)),
            str(row.get("family", "")),
            str(row.get("motif_id", "")),
        ),
    )
    remaining = {str(row["motif_id"]): row for row in ranked}
    groups: list[dict[str, Any]] = []
    group_index = 0

    while remaining:
        anchor_id = next(iter(remaining))
        anchor = remaining.pop(anchor_id)
        candidates = []
        for motif_id, row in remaining.items():
            similarity = motif_overlap_similarity(anchor, row)
            if similarity < min_similarity:
                continue
            candidates.append((similarity, row))
        candidates.sort(
            key=lambda item: (
                -item[0],
                -float(item[1].get("urgency_score", 0.0)),
                str(item[1].get("motif_id", "")),
            )
        )

        members = [anchor]
        for _, row in candidates[: max(0, max_size - 1)]:
            if len(members) >= max_size:
                break
            members.append(row)
            if len(members) >= target_size:
                continue

        for row in members[1:]:
            remaining.pop(str(row["motif_id"]), None)

        group_id = f"G{group_index:03d}"
        group_index += 1
        groups.append(
            {
                "group_id": group_id,
                "motif_ids": [str(row["motif_id"]) for row in members],
                "group_size": len(members),
            }
        )

    return groups


def split_group_halves(motif_ids: list[str]) -> list[list[str]]:
    ordered = sorted(str(motif_id) for motif_id in motif_ids)
    if len(ordered) < 4:
        return []
    first_half = ordered[::2]
    second_half = ordered[1::2]
    halves = [half for half in [first_half, second_half] if half and len(half) < len(ordered)]
    return halves


def motif_release_support(
    motif_row: dict[str, Any],
    instance: ProblemInstance,
    snapshot_row: dict[str, Any],
    teacher_release_ops: set[int],
    backward_depth: int = 1,
    forward_depth: int = 0,
) -> dict[str, list[int]]:
    unfinished_ops = set(int(op_id) for op_id in snapshot_row.get("unfinished_ops", []))
    window_ops = set(int(op_id) for op_id in snapshot_row.get("window_ops", []))
    forced_release_ops = set(int(op_id) for op_id in snapshot_row.get("forced_release_ops", []))
    teacher_droppable_ops = set(int(op_id) for op_id in teacher_release_ops) - forced_release_ops

    closure = apply_precedence_closure(
        [int(op_id) for op_id in motif_row.get("operation_ids", [])],
        instance=instance,
        unfinished_ops=unfinished_ops,
        backward_depth=backward_depth,
        forward_depth=forward_depth,
    )
    induced_release_ops = sorted(set(closure) & window_ops)
    teacher_overlap_ops = sorted(set(induced_release_ops) & teacher_droppable_ops)
    return {
        "induced_release_ops": induced_release_ops,
        "teacher_overlap_ops": teacher_overlap_ops,
    }


def select_calibration_state(
    episode_id: str,
    event_id: str,
    fraction: float,
    salt: str = "grouped_probing_calibration",
) -> bool:
    fraction = max(0.0, min(1.0, float(fraction)))
    if fraction <= 0.0:
        return False
    if fraction >= 1.0:
        return True
    payload = f"{salt}::{episode_id}::{event_id}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    normalized = int(digest[:12], 16) / float(16**12)
    return normalized < fraction


def solve_grouped_nnls(
    observations: list[dict[str, Any]],
    motif_ids: list[str],
    target_key: str,
    ridge_beta: float = 1e-2,
) -> dict[str, float]:
    ordered_ids = [str(motif_id) for motif_id in motif_ids]
    if not ordered_ids:
        return {}
    if not observations:
        return {motif_id: 0.0 for motif_id in ordered_ids}

    design = []
    targets = []
    for observation in observations:
        value = max(0.0, float(observation.get(target_key, 0.0)))
        dropped = {str(motif_id) for motif_id in observation.get("dropped_motif_ids", [])}
        row = [1.0 if motif_id in dropped else 0.0 for motif_id in ordered_ids]
        if not any(row):
            continue
        design.append(row)
        targets.append(value)

    if not design:
        return {motif_id: 0.0 for motif_id in ordered_ids}

    design_matrix = np.asarray(design, dtype=float)
    target_vector = np.asarray(targets, dtype=float)
    ridge = math.sqrt(max(0.0, float(ridge_beta)))
    augmented_matrix = np.vstack([design_matrix, ridge * np.eye(len(ordered_ids), dtype=float)])
    augmented_targets = np.concatenate([target_vector, np.zeros(len(ordered_ids), dtype=float)])
    weights, _ = nnls(augmented_matrix, augmented_targets)
    return {motif_id: float(weight) for motif_id, weight in zip(ordered_ids, weights)}


def build_sparse_pair_candidates(
    motif_rows: list[dict[str, Any]],
    max_pairs: int = 8,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    ordered = sorted(motif_rows, key=lambda row: str(row.get("motif_id", "")))
    for index, left in enumerate(ordered):
        left_ops = set(int(op_id) for op_id in left.get("operation_ids", []))
        left_machines = set(int(machine_id) for machine_id in left.get("machine_ids", []))
        for right in ordered[index + 1 :]:
            right_ops = set(int(op_id) for op_id in right.get("operation_ids", []))
            right_machines = set(int(machine_id) for machine_id in right.get("machine_ids", []))
            shared_ops = left_ops & right_ops
            shared_machines = left_machines & right_machines
            if not shared_ops and not shared_machines:
                continue
            similarity = motif_overlap_similarity(left, right)
            score = 2.0 * len(shared_ops) + 1.0 * len(shared_machines) + similarity
            candidates.append(
                {
                    "motif_id_a": str(left["motif_id"]),
                    "motif_id_b": str(right["motif_id"]),
                    "shared_op_count": len(shared_ops),
                    "shared_machine_count": len(shared_machines),
                    "similarity": similarity,
                    "pair_score": score,
                }
            )
    candidates.sort(
        key=lambda row: (
            -float(row["pair_score"]),
            str(row["motif_id_a"]),
            str(row["motif_id_b"]),
        )
    )
    return candidates[: max(0, int(max_pairs))]
