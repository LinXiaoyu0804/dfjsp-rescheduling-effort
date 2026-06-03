from __future__ import annotations

import unittest

from src.data.schema import Job, Machine, Operation, OperationOption, ProblemInstance, assign_due_dates_by_factor
from src.motifs.extractors import apply_precedence_closure, extract_candidate_motifs


class MotifExtractorTests(unittest.TestCase):
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

    def _build_snapshot(self) -> dict:
        return {
            "episode_id": "ep1",
            "event_id": "bd_0",
            "instance_id": "toy",
            "seed": 0,
            "tau": 5.0,
            "completed_ops": [],
            "active_ops": [],
            "unfinished_ops": [0, 1, 2, 3, 4],
            "window_ops": [0, 1, 2, 3, 4],
            "forced_release_ops": [0, 3],
            "incumbent_assignments": {
                "0": {"machine": 0, "start": 5.0, "end": 8.0},
                "1": {"machine": 1, "start": 8.0, "end": 10.0},
                "2": {"machine": 0, "start": 10.0, "end": 14.0},
                "3": {"machine": 0, "start": 5.0, "end": 7.0},
                "4": {"machine": 2, "start": 7.0, "end": 9.0},
            },
            "event_context": {
                "type": "machine_breakdown",
                "affected_ops": [0, 3],
                "affected_machines": [0],
                "payload": {"machine_id": 0, "down_start": 5.0, "down_end": 6.0},
            },
        }

    def test_extract_candidate_motifs_covers_all_families(self) -> None:
        motifs = extract_candidate_motifs(self._build_instance(), self._build_snapshot(), motif_cfg={"K_motif_pool": 48})
        families = {motif.family for motif in motifs}
        self.assertTrue({"M1", "M2", "M3", "M4"}.issubset(families))
        motif_ids = [motif.motif_id for motif in motifs]
        self.assertEqual(len(motif_ids), len(set(motif_ids)))

    def test_precedence_closure_adds_predecessor(self) -> None:
        instance = self._build_instance()
        closure = apply_precedence_closure([2], instance, unfinished_ops={0, 1, 2, 3, 4}, backward_depth=1, forward_depth=0)
        self.assertEqual([1, 2], closure)


if __name__ == "__main__":
    unittest.main()
