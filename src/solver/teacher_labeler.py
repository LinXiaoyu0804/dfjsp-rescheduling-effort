from __future__ import annotations

from copy import deepcopy

import torch

from src.solver.cp_repair_solver import CPRepairSolver
from src.solver.base import RepairDecision, RepairSolverResult


def labels_from_teacher_decision(
    op_ids: list[int],
    teacher_decision: RepairDecision,
) -> tuple[torch.Tensor, torch.Tensor]:
    keep = torch.zeros(len(op_ids), dtype=torch.float32)
    release = torch.zeros(len(op_ids), dtype=torch.float32)
    keep_set = set(teacher_decision.kept_op_ids)
    release_set = set(teacher_decision.released_op_ids)
    for idx, op_id in enumerate(op_ids):
        release[idx] = 1.0 if op_id in release_set else 0.0
        keep[idx] = 1.0 if op_id in keep_set or op_id not in release_set else 0.0
    return keep, release


def labels_from_teacher_result(
    op_ids: list[int],
    teacher_result: RepairSolverResult,
    incumbent_operations: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    return labels_from_teacher_decision(
        op_ids=op_ids,
        teacher_decision=teacher_decision_from_result(op_ids, teacher_result, incumbent_operations),
    )


def teacher_decision_from_result(
    op_ids: list[int],
    teacher_result: RepairSolverResult,
    incumbent_operations: dict,
    directly_impacted_op_ids: list[int] | None = None,
    trust_region_backward_ratio: float = 0.1,
    trust_region_forward_ratio: float = 0.3,
) -> RepairDecision:
    released = []
    kept = []
    directly_impacted = set(directly_impacted_op_ids or [])
    for op_id in op_ids:
        updated = teacher_result.updated_operations.get(op_id)
        incumbent = incumbent_operations[op_id]
        changed = False
        if updated is not None:
            if updated.machine_id != incumbent.machine_id:
                changed = True
            elif updated.start_time is not None and incumbent.start_time is not None and incumbent.machine_id is not None:
                duration = max(0.0, float(incumbent.end_time or incumbent.start_time) - float(incumbent.start_time))
                if duration <= 0.0:
                    duration = 1.0
                lower_bound = float(incumbent.start_time) - trust_region_backward_ratio * duration
                upper_bound = float(incumbent.start_time) + trust_region_forward_ratio * duration
                changed = not (lower_bound <= float(updated.start_time) <= upper_bound)
        if changed or op_id in directly_impacted:
            released.append(op_id)
        else:
            kept.append(op_id)
    return RepairDecision(immutable_op_ids=[], kept_op_ids=kept, released_op_ids=released, metadata={"source": "teacher"})


def build_teacher_trace_decision(
    subproblem: dict,
    solver_cfg: dict,
    immutable_op_ids: list[int],
    releasable_op_ids: list[int],
    directly_impacted_op_ids: list[int],
    shrink_cfg: dict | None = None,
) -> tuple[RepairDecision, RepairSolverResult]:
    """
    Build a stronger teacher trace than "release everything and inspect changes".

    Procedure:
    1. Solve a full-release teacher subproblem with a larger budget.
    2. Start from the resulting changed-set + directly impacted ops as the release set.
    3. Greedily try to keep additional unchanged ops fixed to their incumbent values while
       preserving near-teacher objective quality.

    This is still an implementation assumption, but it is materially closer to a
    solver-trace-derived keep/release teacher than the previous all-release shortcut.
    """
    shrink_cfg = shrink_cfg or {}
    solver = CPRepairSolver(solver_cfg)
    incumbent = subproblem["incumbent"]
    window = subproblem["window"]

    full_release_decision = RepairDecision(
        immutable_op_ids=sorted(set(immutable_op_ids)),
        kept_op_ids=[],
        released_op_ids=sorted(set(releasable_op_ids)),
        metadata={"teacher_stage": "full_release"},
    )
    full_subproblem = {**subproblem, "decision": full_release_decision}
    full_result = solver.solve(full_subproblem)
    if not full_result.feasible:
        return full_release_decision, full_result

    reference_obj = float(full_result.objective_value or 0.0)
    tolerance_abs = float(shrink_cfg.get("objective_tolerance_abs", 1e-6))
    tolerance_rel = float(shrink_cfg.get("objective_tolerance_rel", 1e-4))
    max_attempts = int(shrink_cfg.get("max_attempts", 24))
    shrink_time_limit = float(shrink_cfg.get("time_limit_sec", min(float(solver_cfg.get("time_limit_sec", 5.0)), 1.0)))

    changed_based_decision = teacher_decision_from_result(
        op_ids=window.op_ids,
        teacher_result=full_result,
        incumbent_operations=incumbent.operations,
        directly_impacted_op_ids=directly_impacted_op_ids,
        trust_region_backward_ratio=float(solver_cfg.get("trust_region_backward_ratio", 0.1)),
        trust_region_forward_ratio=float(solver_cfg.get("trust_region_forward_ratio", 0.3)),
    )

    current_release = sorted(set(changed_based_decision.released_op_ids))
    current_keep = sorted(set(op_id for op_id in releasable_op_ids if op_id not in current_release))
    best_decision = RepairDecision(
        immutable_op_ids=sorted(set(immutable_op_ids)),
        kept_op_ids=current_keep,
        released_op_ids=current_release,
        metadata={
            "teacher_stage": "shrunk",
            "reference_objective": reference_obj,
            "full_release_count": len(full_release_decision.released_op_ids),
        },
    )

    best_result = full_result
    candidate_keep_ops = [
        op_id
        for op_id in best_decision.kept_op_ids
        if op_id not in directly_impacted_op_ids
    ]
    candidate_keep_ops.sort(
        key=lambda op_id: (
            incumbent.operations[op_id].original_start_time
            if incumbent.operations[op_id].original_start_time is not None
            else float("inf"),
            op_id,
        )
    )

    attempts = 0
    for op_id in candidate_keep_ops:
        if attempts >= max_attempts:
            break
        attempts += 1

        trial_keep = [x for x in best_decision.kept_op_ids if x != op_id]
        trial_release = sorted(set(best_decision.released_op_ids + [op_id]))
        trial_decision = RepairDecision(
            immutable_op_ids=best_decision.immutable_op_ids,
            kept_op_ids=trial_keep,
            released_op_ids=trial_release,
            metadata={"teacher_stage": "shrink_trial", "op_id": op_id},
        )
        trial_cfg = deepcopy(solver_cfg)
        trial_cfg["time_limit_sec"] = shrink_time_limit
        trial_cfg["fix_kept_operations"] = True
        trial_result = CPRepairSolver(trial_cfg).solve({**subproblem, "decision": trial_decision})
        if not trial_result.feasible or trial_result.objective_value is None:
            continue
        max_allowed = reference_obj * (1.0 + tolerance_rel) + tolerance_abs
        if float(trial_result.objective_value) <= max_allowed:
            best_decision = RepairDecision(
                immutable_op_ids=best_decision.immutable_op_ids,
                kept_op_ids=trial_keep,
                released_op_ids=trial_release,
                metadata={
                    "teacher_stage": "shrunk",
                    "reference_objective": reference_obj,
                    "full_release_count": len(full_release_decision.released_op_ids),
                    "accepted_trials": attempts,
                },
            )
            best_result = trial_result

    return best_decision, best_result
