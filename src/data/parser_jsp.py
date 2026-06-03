from __future__ import annotations

from pathlib import Path

from .schema import Job, Machine, Operation, OperationOption, ProblemInstance, assign_due_dates_by_factor


def parse_jsp_file(path: str | Path, due_date_factor: float = 1.5) -> ProblemInstance:
    path = Path(path)
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Empty JSP file: {path}")

    header = lines[0].split()
    num_jobs = int(header[0])
    num_machines = int(header[1])
    machines = [Machine(machine_id=i, name=f"M{i}") for i in range(num_machines)]
    jobs: list[Job] = []
    global_op_id = 0

    for job_id in range(num_jobs):
        tokens = [int(x) for x in lines[job_id + 1].split()]
        if len(tokens) % 2 != 0:
            raise ValueError(f"JSP job line must have even length, got {len(tokens)} tokens")
        machine_tokens = tokens[0::2]
        machine_offset = 0 if min(machine_tokens) == 0 else 1
        operations: list[Operation] = []
        for op_idx in range(0, len(tokens), 2):
            machine_id = tokens[op_idx] - machine_offset
            proc_time = float(tokens[op_idx + 1])
            operations.append(
                Operation(
                    op_global_id=global_op_id,
                    job_id=job_id,
                    op_index=op_idx // 2,
                    options=[OperationOption(machine_id=machine_id, processing_time=proc_time)],
                    release_time=0.0,
                    due_date=None,
                )
            )
            global_op_id += 1
        jobs.append(Job(job_id=job_id, operations=operations, release_time=0.0, due_date=None))

    instance = ProblemInstance(family="jsp", jobs=jobs, machines=machines, metadata={"source_path": str(path)})
    return assign_due_dates_by_factor(instance, factor=due_date_factor)
