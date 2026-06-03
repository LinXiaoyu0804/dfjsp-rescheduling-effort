from __future__ import annotations

import unittest
from pathlib import Path
import shutil
import uuid

from src.baselines.daniel_local import (
    _parse_checkpoint_shape,
    _select_checkpoint,
    build_daniel_reduced_subproblem,
)
from src.data.schema import Job, Machine, Operation, OperationOption, ProblemInstance
from src.scheduling.state_builder import StateSnapshot


def _build_toy_instance() -> ProblemInstance:
    job0 = Job(
        job_id=0,
        operations=[
            Operation(0, 0, 0, [OperationOption(0, 3), OperationOption(1, 4)]),
            Operation(1, 0, 1, [OperationOption(0, 5), OperationOption(1, 2)]),
            Operation(2, 0, 2, [OperationOption(1, 6)]),
        ],
    )
    job1 = Job(
        job_id=1,
        operations=[
            Operation(3, 1, 0, [OperationOption(0, 4)]),
            Operation(4, 1, 1, [OperationOption(1, 3)]),
        ],
    )
    return ProblemInstance(
        family="fjsp",
        jobs=[job0, job1],
        machines=[Machine(0), Machine(1)],
    )


class DanielLocalTests(unittest.TestCase):
    def test_parse_checkpoint_shape(self) -> None:
        self.assertEqual(_parse_checkpoint_shape(Path("20x10+mix.pth")), (20, 10))
        self.assertEqual(_parse_checkpoint_shape(Path("15x10.pth")), (15, 10))
        self.assertIsNone(_parse_checkpoint_shape(Path("invalid_name.pth")))

    def test_select_checkpoint_chooses_nearest_shape(self) -> None:
        root = Path("tests/_tmp") / f"daniel_ckpt_{uuid.uuid4().hex}"
        root.mkdir(parents=True, exist_ok=True)
        try:
            for name in ["10x5.pth", "15x10.pth", "20x10+mix.pth"]:
                (root / name).write_bytes(b"checkpoint")
            path, label = _select_checkpoint(root, num_jobs=14, num_machines=9)
            self.assertEqual(label, "15x10")
            self.assertEqual(path.name, "15x10.pth")
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_reduced_subproblem_includes_unfinished_predecessor_closure(self) -> None:
        instance = _build_toy_instance()
        snapshot = StateSnapshot(
            current_time=10.0,
            completed_op_ids=[0],
            active_op_ids=[],
            unfinished_op_ids=[1, 2, 3, 4],
            window_op_ids=[2, 4],
            directly_impacted_op_ids=[4],
            affected_machine_ids=[1],
        )
        reduced = build_daniel_reduced_subproblem(instance, snapshot)
        self.assertEqual(reduced.ranking_candidate_op_ids, [2, 4])
        self.assertEqual(reduced.closure_original_op_ids, [1, 2, 3, 4])
        self.assertEqual(reduced.job_lengths, [2, 2])
        self.assertEqual(reduced.local_to_original_op_ids, [1, 2, 3, 4])
        self.assertEqual(reduced.op_pt_matrix.shape, (4, 2))


if __name__ == "__main__":
    unittest.main()
