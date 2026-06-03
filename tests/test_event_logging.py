from __future__ import annotations

import unittest

from src.eval.event_logging import build_event_log_row


class EventLoggingTests(unittest.TestCase):
    def test_build_event_log_row_contains_core_schema(self) -> None:
        row = build_event_log_row(
            method="slr2_control",
            instance_id="mk6",
            seed=0,
            episode_id="mk6_seed00_ep001",
            event_id="arr_0",
            tau=11.0,
            budget_sec=5.0,
            window_size=12,
            forced_release_count=2,
            motif_count=0,
            selected_motif_count=0,
            released_op_count=5,
            pred_gain_sum=None,
            inference_runtime_sec=0.02,
            selector_runtime_sec=0.0,
            solver_runtime_sec=0.4,
            makespan_after=100.0,
            tardiness_after=12.0,
            instability_after=3.5,
            weighted_objective_after=112.35,
            changed_op_ratio=0.4,
            changed_machine_ratio=0.1,
            mean_abs_start_time_deviation=1.25,
            status="feasible",
        )
        self.assertEqual(row["method"], "slr2_control")
        self.assertEqual(row["window_size"], 12)
        self.assertEqual(row["released_op_count"], 5)
        self.assertEqual(row["status"], "feasible")


if __name__ == "__main__":
    unittest.main()
