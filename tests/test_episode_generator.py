from __future__ import annotations

import unittest
from pathlib import Path

from src.data.unified_parser import parse_instance
from src.events.generator import estimate_nominal_makespan, generate_dynamic_events
from src.utils.config import load_yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


class EpisodeGeneratorTests(unittest.TestCase):
    def test_stronger_v2_protocol_is_seed_stable_and_in_range(self) -> None:
        cfg = load_yaml(REPO_ROOT / "configs" / "env" / "formal_dynamic_stronger_v2.yaml")
        instance = parse_instance(
            REPO_ROOT / "data" / "raw" / "fjsp" / "FJSP-benchmarks-main" / "1_Brandimarte" / "BrandimarteMk6.fjs",
            family="fjsp",
            due_date_factor=1.5,
        )
        nominal = estimate_nominal_makespan(instance)
        first = generate_dynamic_events(instance, cfg["events"], seed=7)
        second = generate_dynamic_events(instance, cfg["events"], seed=7)

        self.assertEqual(
            [(event.event_type, event.time, event.payload) for event in first],
            [(event.event_type, event.time, event.payload) for event in second],
        )

        arrivals = [event for event in first if event.event_type == "job_arrival"]
        breakdowns = [event for event in first if event.event_type == "machine_breakdown"]
        self.assertEqual(4, len(arrivals))
        self.assertEqual(4, len(breakdowns))

        for event in arrivals:
            self.assertGreaterEqual(event.time, 0.15 * nominal)
            self.assertLessEqual(event.time, 0.65 * nominal)

        for event in breakdowns:
            duration = float(event.payload["end_time"]) - float(event.payload["start_time"])
            self.assertGreaterEqual(event.time, 0.10 * nominal)
            self.assertLessEqual(event.time, 0.70 * nominal)
            self.assertGreaterEqual(duration, 0.02 * nominal)
            self.assertLessEqual(duration, 0.10 * nominal)


if __name__ == "__main__":
    unittest.main()
