from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


EventType = Literal["job_arrival", "machine_breakdown", "processing_time_perturbation", "compound"]


@dataclass(slots=True)
class DynamicEvent:
    event_id: int
    time: float
    event_type: EventType
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class JobArrivalPayload:
    new_job_id: int
    template_job_id: int | None = None
    release_time: float = 0.0


@dataclass(slots=True)
class MachineBreakdownPayload:
    machine_id: int
    start_time: float
    end_time: float


@dataclass(slots=True)
class ProcessingTimePerturbationPayload:
    op_global_id: int
    machine_id: int
    old_processing_time: float
    new_processing_time: float
