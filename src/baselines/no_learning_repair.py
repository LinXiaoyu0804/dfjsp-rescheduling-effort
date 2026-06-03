from __future__ import annotations

from src.baselines.base import BaselineOutput
from src.solver.base import RepairDecision


def no_learning_repair_decision(snapshot) -> BaselineOutput:
    immutable = sorted(set(snapshot.completed_op_ids + snapshot.active_op_ids))
    release = [op_id for op_id in snapshot.directly_impacted_op_ids if op_id not in immutable]
    if not release:
        release = [op_id for op_id in snapshot.window_op_ids if op_id not in immutable][:1]
    keep = [op_id for op_id in snapshot.window_op_ids if op_id not in release]
    return BaselineOutput(
        decision=RepairDecision(immutable_op_ids=immutable, kept_op_ids=keep, released_op_ids=release, metadata={"mode": "no_learning"}),
        name="no_learning_repair",
    )
