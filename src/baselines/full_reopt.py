from __future__ import annotations

from src.baselines.base import BaselineOutput
from src.solver.base import RepairDecision


def full_reoptimization_decision(snapshot) -> BaselineOutput:
    immutable = sorted(set(snapshot.completed_op_ids + snapshot.active_op_ids))
    release = [op_id for op_id in snapshot.window_op_ids if op_id not in immutable]
    keep = [op_id for op_id in snapshot.window_op_ids if op_id in immutable]
    return BaselineOutput(
        decision=RepairDecision(immutable_op_ids=immutable, kept_op_ids=keep, released_op_ids=release, metadata={"mode": "full_window"}),
        name="full_reoptimization",
    )
