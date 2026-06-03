from __future__ import annotations

from src.baselines.base import BaselineOutput
from src.solver.base import RepairDecision


def heuristic_rh_decision(snapshot, release_cap: int = 12) -> BaselineOutput:
    ranked = list(snapshot.directly_impacted_op_ids) + [op_id for op_id in snapshot.window_op_ids if op_id not in snapshot.directly_impacted_op_ids]
    candidates = [op_id for op_id in ranked if op_id not in snapshot.completed_op_ids and op_id not in snapshot.active_op_ids]
    release = candidates[:release_cap]
    keep = [op_id for op_id in snapshot.window_op_ids if op_id not in release]
    immutable = sorted(set(snapshot.completed_op_ids + snapshot.active_op_ids))
    return BaselineOutput(
        decision=RepairDecision(immutable_op_ids=immutable, kept_op_ids=keep, released_op_ids=release, metadata={"heuristic": "event_impacted_then_window"}),
        name="heuristic_rh",
    )
