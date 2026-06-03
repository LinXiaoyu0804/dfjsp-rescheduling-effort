from __future__ import annotations

from dataclasses import dataclass

from src.solver.base import RepairDecision


@dataclass(slots=True)
class BaselineOutput:
    decision: RepairDecision
    name: str
