from __future__ import annotations

import shutil
import unittest
from pathlib import Path

import torch

from src.data.schema import Job, Machine, Operation, OperationOption, ProblemInstance, assign_due_dates_by_factor
from src.eval.external_baselines import (
    acceptance_reference_gate_passes,
    acceptance_reference_margin_abs,
    build_external_baseline_output,
    compute_baseline_proxy_selection,
    merge_teacher_trace_shards,
)
from src.scheduling.incumbent import IncumbentSchedule, MachineCalendar, OperationSchedule
from src.scheduling.state_builder import StateSnapshot


class ExternalBaselinesTests(unittest.TestCase):
    def _build_instance(self) -> ProblemInstance:
        instance = ProblemInstance(
            family="fjsp",
            jobs=[
                Job(
                    job_id=0,
                    operations=[
                        Operation(0, 0, 0, [OperationOption(0, 3), OperationOption(1, 4)]),
                        Operation(1, 0, 1, [OperationOption(1, 2)]),
                    ],
                ),
                Job(
                    job_id=1,
                    operations=[
                        Operation(2, 1, 0, [OperationOption(0, 2), OperationOption(1, 3)]),
                        Operation(3, 1, 1, [OperationOption(1, 2)]),
                    ],
                ),
            ],
            machines=[Machine(0), Machine(1)],
        )
        return assign_due_dates_by_factor(instance, 1.5)

    def test_compute_baseline_proxy_selection_counts_only_extra_release(self) -> None:
        selection_flag, extra_release_count = compute_baseline_proxy_selection([0, 2, 3], [0, 2])
        self.assertEqual(selection_flag, 1)
        self.assertEqual(extra_release_count, 1)

        selection_flag, extra_release_count = compute_baseline_proxy_selection([0, 2], [0, 2])
        self.assertEqual(selection_flag, 0)
        self.assertEqual(extra_release_count, 0)

    def test_merge_teacher_trace_shards_concatenates_rows_and_builds_summary(self) -> None:
        tmp_path = Path("tests_tmp_external_baselines")
        if tmp_path.exists():
            shutil.rmtree(tmp_path)
        tmp_path.mkdir(parents=True, exist_ok=True)
        try:
            shard_a = tmp_path / "a.pt"
            shard_b = tmp_path / "b.pt"
            torch.save(
                [
                    {
                        "metadata": {
                            "instance_id": "mk1",
                            "seed": 0,
                            "teacher_feasible": True,
                            "teacher_release_count": 3,
                        }
                    },
                    {
                        "metadata": {
                            "instance_id": "mk1",
                            "seed": 0,
                            "teacher_feasible": False,
                            "teacher_release_count": 1,
                        }
                    },
                ],
                shard_a,
            )
            torch.save(
                [
                    {
                        "metadata": {
                            "instance_id": "mk2",
                            "seed": 1,
                            "teacher_feasible": True,
                            "teacher_release_count": 4,
                        }
                    }
                ],
                shard_b,
            )

            merged_rows, summary_rows = merge_teacher_trace_shards([shard_b, shard_a])
            self.assertEqual(len(merged_rows), 3)
            self.assertEqual(len(summary_rows), 2)
            mk1_row = next(row for row in summary_rows if row["instance_id"] == "mk1")
            self.assertEqual(mk1_row["num_events"], 2)
            self.assertEqual(mk1_row["num_feasible_events"], 1)
            self.assertAlmostEqual(mk1_row["feasible_rate"], 0.5, places=6)
            self.assertAlmostEqual(mk1_row["mean_teacher_release_count"], 2.0, places=6)
        finally:
            if tmp_path.exists():
                shutil.rmtree(tmp_path)

    def test_build_external_baseline_output_supports_dispatching_atc(self) -> None:
        instance = self._build_instance()
        incumbent = IncumbentSchedule(
            operations={
                0: OperationSchedule(0, 0, 0, 0, 0.0, 3.0, "unstarted", 0.0, 3.0, 0),
                1: OperationSchedule(1, 0, 1, 1, 3.0, 5.0, "unstarted", 3.0, 5.0, 1),
                2: OperationSchedule(2, 1, 0, 0, 3.0, 5.0, "unstarted", 3.0, 5.0, 0),
                3: OperationSchedule(3, 1, 1, 1, 5.0, 7.0, "unstarted", 5.0, 7.0, 1),
            },
            machine_calendars={
                0: MachineCalendar(0, available_time=5.0),
                1: MachineCalendar(1, available_time=7.0),
            },
            current_time=1.0,
        )
        snapshot = StateSnapshot(
            current_time=1.0,
            completed_op_ids=[],
            active_op_ids=[],
            unfinished_op_ids=[0, 1, 2, 3],
            window_op_ids=[0, 1, 2, 3],
            directly_impacted_op_ids=[0, 2],
            affected_machine_ids=[0],
            triggering_event_type="machine_breakdown",
        )
        baseline = build_external_baseline_output(
            baseline_name="dispatching_atc",
            instance=instance,
            incumbent=incumbent,
            snapshot=snapshot,
            graph=None,
            cfg={},
        )
        self.assertEqual(baseline.name, "dispatching_atc")
        self.assertEqual(baseline.decision.metadata["rule"], "ATC")

    def test_acceptance_reference_gate_can_require_complexity_and_no_learned_selection(self) -> None:
        controller_cfg = {
            "acceptance_reference_margin_abs": 1.0,
            "acceptance_reference_gates": {
                "dispatching_atc": {
                    "min_window_size": 10,
                    "min_forced_release_count": 2,
                    "require_no_learned_selection": True,
                    "allowed_event_types": ["breakdown"],
                }
            },
        }
        snapshot_row = {
            "event_id": "bd_0",
            "window_ops": list(range(12)),
            "forced_release_ops": [0, 1],
            "motif_count": 20,
            "event_context": {"type": "machine_breakdown"},
        }
        self.assertTrue(
            acceptance_reference_gate_passes(
                "dispatching_atc",
                snapshot_row,
                controller_cfg,
                learned_selection_active=False,
            )
        )
        self.assertFalse(
            acceptance_reference_gate_passes(
                "dispatching_atc",
                snapshot_row,
                controller_cfg,
                learned_selection_active=True,
            )
        )
        self.assertAlmostEqual(acceptance_reference_margin_abs("dispatching_atc", controller_cfg), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
