from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


OperationStatus = Literal["unstarted", "active", "completed"]


@dataclass(slots=True)
class OperationSchedule:
    op_global_id: int
    job_id: int
    op_index: int
    machine_id: int | None
    start_time: float | None
    end_time: float | None
    status: OperationStatus = "unstarted"
    original_start_time: float | None = None
    original_end_time: float | None = None
    original_machine_id: int | None = None

    def clone_as_original_if_missing(self) -> "OperationSchedule":
        if self.original_start_time is None:
            self.original_start_time = self.start_time
        if self.original_end_time is None:
            self.original_end_time = self.end_time
        if self.original_machine_id is None:
            self.original_machine_id = self.machine_id
        return self


@dataclass(slots=True)
class MachineCalendar:
    machine_id: int
    available_time: float = 0.0
    breakdowns: list[tuple[float, float]] = field(default_factory=list)


@dataclass(slots=True)
class IncumbentSchedule:
    operations: dict[int, OperationSchedule]
    machine_calendars: dict[int, MachineCalendar]
    current_time: float = 0.0

    def completed_ops(self) -> list[int]:
        return [op_id for op_id, op in self.operations.items() if op.status == "completed"]

    def active_ops(self) -> list[int]:
        return [op_id for op_id, op in self.operations.items() if op.status == "active"]

    def unfinished_ops(self) -> list[int]:
        return [op_id for op_id, op in self.operations.items() if op.status != "completed"]

    def started_ops(self) -> list[int]:
        return [op_id for op_id, op in self.operations.items() if op.status in {"active", "completed"}]

    def get(self, op_global_id: int) -> OperationSchedule:
        return self.operations[op_global_id]
