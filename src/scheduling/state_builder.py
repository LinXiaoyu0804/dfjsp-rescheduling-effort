from __future__ import annotations

from dataclasses import dataclass, field

from src.data.schema import ProblemInstance
from src.events.schema import DynamicEvent
from src.scheduling.incumbent import IncumbentSchedule
from src.scheduling.window import RollingWindow, build_rolling_window


@dataclass(slots=True)
class StateSnapshot:
    current_time: float
    completed_op_ids: list[int]
    active_op_ids: list[int]
    unfinished_op_ids: list[int]
    window_op_ids: list[int]
    directly_impacted_op_ids: list[int]
    affected_machine_ids: list[int]
    triggering_event_type: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


def build_state_snapshot(
    instance: ProblemInstance,
    incumbent: IncumbentSchedule,
    event: DynamicEvent,
    horizon: float,
    max_window_ops: int,
    include_downstream_successors: bool = False,
) -> tuple[StateSnapshot, RollingWindow]:
    window = build_rolling_window(
        instance,
        incumbent,
        event,
        horizon,
        max_window_ops,
        include_downstream_successors=include_downstream_successors,
    )
    snapshot = StateSnapshot(
        current_time=event.time,
        completed_op_ids=incumbent.completed_ops(),
        active_op_ids=incumbent.active_ops(),
        unfinished_op_ids=incumbent.unfinished_ops(),
        window_op_ids=window.op_ids,
        directly_impacted_op_ids=window.directly_impacted_op_ids,
        affected_machine_ids=window.affected_machine_ids,
        triggering_event_type=event.event_type,
        metadata={"event_payload": event.payload},
    )
    return snapshot, window
