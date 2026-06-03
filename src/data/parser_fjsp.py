from __future__ import annotations

from pathlib import Path

from .schema import Job, Machine, Operation, OperationOption, ProblemInstance, assign_due_dates_by_factor


def parse_fjsp_file(path: str | Path, due_date_factor: float = 1.5) -> ProblemInstance:
    path = Path(path)
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Empty FJSP file: {path}")

    header = lines[0].split()
    num_jobs = int(header[0])
    num_machines = int(header[1])
    machines = [Machine(machine_id=i, name=f"M{i}") for i in range(num_machines)]
    jobs: list[Job] = []
    global_op_id = 0

    for job_id in range(num_jobs):
        tokens = [int(x) for x in lines[job_id + 1].split()]
        cursor = 0
        num_ops = tokens[cursor]
        cursor += 1
        operations: list[Operation] = []
        for op_idx in range(num_ops):
            num_options = tokens[cursor]
            cursor += 1
            options: list[OperationOption] = []
            for _ in range(num_options):
                machine_id = tokens[cursor] - 1
                proc_time = float(tokens[cursor + 1])
                cursor += 2
                options.append(OperationOption(machine_id=machine_id, processing_time=proc_time))
            operations.append(
                Operation(
                    op_global_id=global_op_id,
                    job_id=job_id,
                    op_index=op_idx,
                    options=options,
                    release_time=0.0,
                    due_date=None,
                )
            )
            global_op_id += 1
        jobs.append(Job(job_id=job_id, operations=operations, release_time=0.0, due_date=None))

    instance = ProblemInstance(family="fjsp", jobs=jobs, machines=machines, metadata={"source_path": str(path)})
    return assign_due_dates_by_factor(instance, factor=due_date_factor)
