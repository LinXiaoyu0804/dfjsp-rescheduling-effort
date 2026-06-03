from __future__ import annotations

import unittest

from src.data.schema import Job, Machine, Operation, OperationOption, ProblemInstance, assign_due_dates_by_factor
from src.motifs.grouped_probing import (
    build_sparse_pair_candidates,
    group_motifs_by_overlap,
    motif_release_support,
    select_calibration_state,
    solve_grouped_nnls,
)


class GroupedProbingTests(unittest.TestCase):
    def _build_instance(self) -> ProblemInstance:
        jobs = [
            Job(
                job_id=0,
                operations=[
                    Operation(0, 0, 0, [OperationOption(0, 3), OperationOption(1, 4)]),
                    Operation(1, 0, 1, [OperationOption(1, 2)]),
                    Operation(2, 0, 2, [OperationOption(0, 4), OperationOption(2, 5)]),
                ],
            ),
            Job(
                job_id=1,
                operations=[
                    Operation(3, 1, 0, [OperationOption(0, 2), OperationOption(1, 3)]),
                    Operation(4, 1, 1, [OperationOption(0, 4), OperationOption(2, 2)]),
                ],
            ),
        ]
        instance = ProblemInstance(
            family="fjsp",
            jobs=jobs,
            machines=[Machine(0), Machine(1), Machine(2)],
        )
        return assign_due_dates_by_factor(instance, 1.5)

    def test_group_motifs_by_overlap_is_deterministic(self) -> None:
        motif_rows = [
            {"motif_id": "m1", "family": "M2", "anchor_type": "machine", "anchor_id": "0", "operation_ids": [0, 1], "machine_ids": [0], "urgency_score": 4.0},
            {"motif_id": "m2", "family": "M2", "anchor_type": "machine", "anchor_id": "0", "operation_ids": [0, 2], "machine_ids": [0], "urgency_score": 3.5},
            {"motif_id": "m3", "family": "M1", "anchor_type": "operation", "anchor_id": "3", "operation_ids": [3], "machine_ids": [0, 1], "urgency_score": 2.0},
        ]
        groups = group_motifs_by_overlap(motif_rows, target_size=2, max_size=3, min_similarity=0.2)
        self.assertEqual(groups[0]["motif_ids"], ["m1", "m2"])
        self.assertEqual(groups[1]["motif_ids"], ["m3"])

    def test_motif_release_support_intersects_teacher_release(self) -> None:
        support = motif_release_support(
            motif_row={"motif_id": "m", "operation_ids": [2], "machine_ids": [0]},
            instance=self._build_instance(),
            snapshot_row={
                "unfinished_ops": [0, 1, 2, 3, 4],
                "window_ops": [0, 1, 2, 3, 4],
                "forced_release_ops": [0],
            },
            teacher_release_ops={0, 1, 2},
            backward_depth=1,
            forward_depth=0,
        )
        self.assertEqual(support["induced_release_ops"], [1, 2])
        self.assertEqual(support["teacher_overlap_ops"], [1, 2])

    def test_solve_grouped_nnls_recovers_nonnegative_weights(self) -> None:
        weights = solve_grouped_nnls(
            observations=[
                {"dropped_motif_ids": ["a"], "delta_objective": 3.0},
                {"dropped_motif_ids": ["b"], "delta_objective": 2.0},
                {"dropped_motif_ids": ["a", "b"], "delta_objective": 5.0},
            ],
            motif_ids=["a", "b"],
            target_key="delta_objective",
            ridge_beta=1e-6,
        )
        self.assertAlmostEqual(weights["a"], 3.0, places=3)
        self.assertAlmostEqual(weights["b"], 2.0, places=3)

    def test_calibration_state_selection_is_stable(self) -> None:
        first = select_calibration_state("ep1", "ev1", 0.5)
        second = select_calibration_state("ep1", "ev1", 0.5)
        self.assertEqual(first, second)

    def test_sparse_pair_candidates_require_overlap(self) -> None:
        pairs = build_sparse_pair_candidates(
            [
                {"motif_id": "a", "operation_ids": [0, 1], "machine_ids": [0]},
                {"motif_id": "b", "operation_ids": [1, 2], "machine_ids": [1]},
                {"motif_id": "c", "operation_ids": [3], "machine_ids": [2]},
            ],
            max_pairs=8,
        )
        self.assertEqual(len(pairs), 1)
        self.assertEqual((pairs[0]["motif_id_a"], pairs[0]["motif_id_b"]), ("a", "b"))


if __name__ == "__main__":
    unittest.main()
