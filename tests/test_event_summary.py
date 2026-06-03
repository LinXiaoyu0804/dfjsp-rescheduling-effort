from __future__ import annotations

import unittest

from src.eval.event_summary import (
    build_complexity_summary,
    build_instance_group_summary,
    build_instance_summary,
    normalize_event_metrics,
)


class EventSummaryTests(unittest.TestCase):
    def _rows(self) -> list[dict]:
        return [
            {
                "method": "pairwise_v5",
                "instance_id": "mk6",
                "event_id": "arr_0",
                "status": "feasible",
                "selection_source": "pairwise",
                "operator_selection_mode": "pairwise",
                "window_size": 70,
                "forced_release_count": 1,
                "motif_count": 20,
                "selected_motif_count": 1,
                "released_op_count": 3,
                "reward_delta": 0.5,
                "weighted_objective_after": 100.0,
                "solver_runtime_sec": 1.2,
                "changed_op_ratio": 0.10,
                "changed_machine_ratio": 0.05,
                "operator_release_match_score": 0.0,
                "makespan_after": 50.0,
                "tardiness_after": 0.0,
            },
            {
                "method": "pairwise_v5",
                "instance_id": "mk7",
                "event_id": "bd_0",
                "status": "infeasible",
                "selection_source": "pairwise",
                "operator_selection_mode": "pairwise",
                "window_size": 180,
                "forced_release_count": 4,
                "motif_count": 40,
                "selected_motif_count": 0,
                "released_op_count": 1,
                "reward_delta": 0.0,
                "weighted_objective_after": 110.0,
                "solver_runtime_sec": 1.5,
                "changed_op_ratio": 0.20,
                "changed_machine_ratio": 0.10,
                "operator_release_match_score": 0.0,
                "makespan_after": 55.0,
                "tardiness_after": 2.0,
            },
            {
                "method": "alns_v10",
                "instance_id": "mk8",
                "event_id": "arr_1",
                "status": "feasible",
                "selection_source": "alns_lite",
                "operator_selection_mode": "relaxed_family_fallback",
                "window_size": 250,
                "forced_release_count": 6,
                "motif_count": 70,
                "selected_motif_count": 2,
                "released_op_count": 5,
                "reward_delta": 1.0,
                "weighted_objective_after": 95.0,
                "solver_runtime_sec": 1.8,
                "changed_op_ratio": 0.30,
                "changed_machine_ratio": 0.15,
                "operator_release_match_score": 0.8,
                "makespan_after": 45.0,
                "tardiness_after": 0.0,
            },
        ]

    def test_normalize_event_metrics_adds_scale_bins(self) -> None:
        normalized = normalize_event_metrics(self._rows())
        self.assertEqual(list(normalized["instance_group"]), ["mk6_mk7", "mk6_mk7", "mk8_mk10"])
        self.assertEqual(list(normalized["window_size_bin"]), ["0-80", "161-240", "241+"])
        self.assertEqual(list(normalized["forced_release_bin"]), ["1-2", "3-5", "6+"])
        self.assertEqual(list(normalized["motif_count_bin"]), ["16-31", "32-63", "64+"])

    def test_build_instance_summary_preserves_legacy_shape(self) -> None:
        summary = build_instance_summary(self._rows())
        self.assertEqual(
            list(summary.columns),
            [
                "method",
                "instance_id",
                "num_events",
                "mean_motif_count",
                "mean_selected_motif_count",
                "mean_released_op_count",
                "mean_makespan",
                "mean_tardiness",
                "mean_changed_op_ratio",
                "mean_changed_machine_ratio",
                "feasible_rate",
            ],
        )
        mk6_row = summary[(summary["method"] == "pairwise_v5") & (summary["instance_id"] == "mk6")].iloc[0]
        self.assertEqual(int(mk6_row["num_events"]), 1)
        self.assertAlmostEqual(float(mk6_row["mean_selected_motif_count"]), 1.0, places=6)
        self.assertAlmostEqual(float(mk6_row["feasible_rate"]), 1.0, places=6)

    def test_group_and_complexity_summaries_capture_controller_activity(self) -> None:
        group_summary = build_instance_group_summary(self._rows())
        pairwise_group = group_summary[
            (group_summary["method"] == "pairwise_v5") & (group_summary["instance_group"] == "mk6_mk7")
        ].iloc[0]
        self.assertEqual(int(pairwise_group["num_events"]), 2)
        self.assertAlmostEqual(float(pairwise_group["selection_rate"]), 0.5, places=6)
        self.assertAlmostEqual(float(pairwise_group["positive_reward_rate"]), 0.5, places=6)

        complexity_summary = build_complexity_summary(self._rows())
        alns_release_row = complexity_summary[
            (complexity_summary["method"] == "alns_v10")
            & (complexity_summary["instance_group"] == "mk8_mk10")
            & (complexity_summary["stratifier"] == "forced_release_count")
            & (complexity_summary["bucket"] == "6+")
        ].iloc[0]
        self.assertAlmostEqual(float(alns_release_row["alns_usage_rate"]), 1.0, places=6)
        self.assertAlmostEqual(float(alns_release_row["relaxed_usage_rate"]), 1.0, places=6)
        self.assertAlmostEqual(float(alns_release_row["mean_operator_release_match_score"]), 0.8, places=6)

    def test_synthetic_instances_are_grouped_by_scale(self) -> None:
        normalized = normalize_event_metrics(
            [
                {
                    "method": "ood_probe",
                    "instance_id": "syn_50x15_01",
                    "event_id": "bd_1",
                    "status": "feasible",
                    "selection_source": "pairwise",
                    "operator_selection_mode": "pairwise_rank_fallback",
                    "window_size": 220,
                    "forced_release_count": 3,
                    "motif_count": 48,
                    "selected_motif_count": 1,
                    "released_op_count": 2,
                    "reward_delta": 0.2,
                    "weighted_objective_after": 123.0,
                    "solver_runtime_sec": 2.1,
                    "changed_op_ratio": 0.12,
                    "changed_machine_ratio": 0.04,
                    "operator_release_match_score": 0.9,
                }
            ]
        )
        self.assertEqual(normalized.iloc[0]["instance_group"], "synthetic_50x15")


if __name__ == "__main__":
    unittest.main()
