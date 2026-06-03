from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(slots=True)
class Machine:
    machine_id: int
    name: str | None = None


@dataclass(slots=True)
class OperationOption:
    machine_id: int
    processing_time: float


@dataclass(slots=True)
class Operation:
    op_global_id: int
    job_id: int
    op_index: int
    options: list[OperationOption]
    release_time: float = 0.0
    due_date: float | None = None

    @property
    def eligible_machine_ids(self) -> list[int]:
        return [opt.machine_id for opt in self.options]

    def processing_time_on(self, machine_id: int) -> float:
        for option in self.options:
            if option.machine_id == machine_id:
                return option.processing_time
        raise KeyError(f"Operation {self.op_global_id} not eligible on machine {machine_id}")

    @property
    def min_processing_time(self) -> float:
        return min(opt.processing_time for opt in self.options)


@dataclass(slots=True)
class Job:
    job_id: int
    operations: list[Operation]
    release_time: float = 0.0
    due_date: float | None = None
    name: str | None = None

    @property
    def total_min_processing_time(self) -> float:
        return sum(op.min_processing_time for op in self.operations)


@dataclass(slots=True)
class ProblemInstance:
    family: str
    jobs: list[Job]
    machines: list[Machine]
    metadata: dict[str, object] = field(default_factory=dict)

    def iter_operations(self) -> Iterable[Operation]:
        for job in self.jobs:
            yield from job.operations

    @property
    def num_jobs(self) -> int:
        return len(self.jobs)

    @property
    def num_machines(self) -> int:
        return len(self.machines)

    @property
    def num_operations(self) -> int:
        return sum(len(job.operations) for job in self.jobs)

    def get_job(self, job_id: int) -> Job:
        for job in self.jobs:
            if job.job_id == job_id:
                return job
        raise KeyError(f"Unknown job_id={job_id}")

    def get_operation(self, op_global_id: int) -> Operation:
        for op in self.iter_operations():
            if op.op_global_id == op_global_id:
                return op
        raise KeyError(f"Unknown op_global_id={op_global_id}")

    def precedence_pairs(self) -> list[tuple[int, int]]:
        pairs: list[tuple[int, int]] = []
        for job in self.jobs:
            for prev_op, next_op in zip(job.operations[:-1], job.operations[1:]):
                pairs.append((prev_op.op_global_id, next_op.op_global_id))
        return pairs


def assign_due_dates_by_factor(instance: ProblemInstance, factor: float) -> ProblemInstance:
    for job in instance.jobs:
        if job.due_date is None:
            due_date = job.release_time + factor * job.total_min_processing_time
            job.due_date = due_date
        for op in job.operations:
            if op.due_date is None:
                op.due_date = job.due_date
    return instance
