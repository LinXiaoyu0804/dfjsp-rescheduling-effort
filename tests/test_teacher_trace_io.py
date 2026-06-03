from __future__ import annotations

import unittest

import torch

from src.data.teacher_trace_io import build_operation_dataset_record, freeze_graph_tensors


class TeacherTraceIOTests(unittest.TestCase):
    def test_freeze_graph_tensors_clones_mutable_payload(self) -> None:
        graph = {
            "op_x": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
            "op_ids": [1, 2, 3],
            "payload": {"nested": [1, 2]},
        }
        frozen = freeze_graph_tensors(graph)
        graph["op_x"][0, 0] = 9.0
        graph["op_ids"].append(4)
        graph["payload"]["nested"].append(3)
        self.assertAlmostEqual(float(frozen["op_x"][0, 0]), 1.0)
        self.assertEqual(frozen["op_ids"], [1, 2, 3])
        self.assertEqual(frozen["payload"]["nested"], [1, 2])

    def test_build_operation_dataset_record_joins_teacher_and_snapshot_fields(self) -> None:
        record = build_operation_dataset_record(
            sample_metadata={
                "episode_id": "mk1_seed00_ep001",
                "event_id": "bd_0",
                "event_type": "machine_breakdown",
                "num_window_ops": 7,
                "teacher_release_count": 3,
                "teacher_keep_count": 4,
                "teacher_objective": 123.5,
                "teacher_feasible": True,
                "teacher_solver_status": "OPTIMAL",
                "teacher_runtime_sec": 0.75,
                "teacher_stage": "shrunk",
            },
            snapshot_row={
                "episode_id": "mk1_seed00_ep001",
                "event_id": "bd_0",
                "instance_id": "mk1",
                "instance_path": "data/raw/mk1.fjs",
                "seed": 0,
                "tau": 17.0,
                "window_size": 7,
                "forced_release_count": 2,
            },
            shard_path="outputs/teacher/mk1_seed00_ep001.pt",
            sample_index=5,
        )
        self.assertEqual(record["dataset_index"], 5)
        self.assertEqual(record["instance_id"], "mk1")
        self.assertEqual(record["forced_release_count"], 2)
        self.assertEqual(record["teacher_release_count"], 3)
        self.assertAlmostEqual(record["teacher_runtime_sec"], 0.75)


if __name__ == "__main__":
    unittest.main()
