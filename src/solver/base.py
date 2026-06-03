from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from src.scheduling.incumbent import OperationSchedule


@dataclass(slots=True)
class RepairDecision:
    immutable_op_ids: list[int]
    kept_op_ids: list[int]
    released_op_ids: list[int]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RepairSolverResult:
    feasible: bool
    solver_status: str
    updated_operations: dict[int, OperationSchedule]
    objective_value: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class RepairSolver(Protocol):
    def solve(self, subproblem: dict[str, Any]) -> RepairSolverResult:
        ...
