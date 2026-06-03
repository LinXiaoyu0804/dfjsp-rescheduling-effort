from __future__ import annotations

import unittest

from src.solver.runtime_accounting import resolve_solver_runtime_accounting


class SolverRuntimeAccountingTests(unittest.TestCase):
    def test_prefers_solver_wall_time_when_raw_runtime_is_pathologically_large(self) -> None:
        resolved = resolve_solver_runtime_accounting(
            raw_wall_time_sec=13009.877001,
            solver_metadata={"solver_wall_time_sec": 5.021, "time_limit_sec": 5.0},
            decision_metadata={"solver_time_limit_sec": 5.0},
        )
        self.assertTrue(resolved["raw_timing_anomaly"])
        self.assertFalse(resolved["budget_violation"])
        self.assertEqual(resolved["runtime_accounting_source"], "solver_wall_time")
        self.assertAlmostEqual(resolved["runtime_sec"], 5.021, places=6)

    def test_flags_budget_violation_when_no_solver_report_is_available(self) -> None:
        resolved = resolve_solver_runtime_accounting(
            raw_wall_time_sec=8.0,
            solver_metadata={"time_limit_sec": 5.0},
            decision_metadata={"solver_time_limit_sec": 5.0},
        )
        self.assertTrue(resolved["budget_violation"])
        self.assertTrue(resolved["runtime_clipped"])
        self.assertAlmostEqual(resolved["runtime_budget_cap_sec"], 6.0, places=6)
        self.assertAlmostEqual(resolved["runtime_sec"], 6.0, places=6)


if __name__ == "__main__":
    unittest.main()
