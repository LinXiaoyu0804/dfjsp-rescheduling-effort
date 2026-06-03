from __future__ import annotations

import unittest

from src.eval.metrics import compute_changed_machine_ratio, compute_changed_operation_ratio, compute_mean_absolute_start_time_deviation
from src.scheduling.incumbent import IncumbentSchedule, MachineCalendar, OperationSchedule


class ScheduleMetricTests(unittest.TestCase):
    def test_changed_ratios_and_start_time_deviation(self) -> None:
        incumbent = IncumbentSchedule(
            operations={
                0: OperationSchedule(
                    op_global_id=0,
                    job_id=0,
                    op_index=0,
                    machine_id=0,
                    start_time=0.0,
                    end_time=5.0,
                    original_start_time=0.0,
                    original_end_time=5.0,
                    original_machine_id=0,
                ),
                1: OperationSchedule(
                    op_global_id=1,
                    job_id=0,
                    op_index=1,
                    machine_id=1,
                    start_time=8.0,
                    end_time=13.0,
                    original_start_time=6.0,
                    original_end_time=11.0,
                    original_machine_id=0,
                ),
                2: OperationSchedule(
                    op_global_id=2,
                    job_id=1,
                    op_index=0,
                    machine_id=1,
                    start_time=4.0,
                    end_time=9.0,
                    original_start_time=4.0,
                    original_end_time=9.0,
                    original_machine_id=1,
                ),
                3: OperationSchedule(
                    op_global_id=3,
                    job_id=1,
                    op_index=1,
                    machine_id=0,
                    start_time=12.0,
                    end_time=17.0,
                    original_start_time=10.0,
                    original_end_time=15.0,
                    original_machine_id=0,
                ),
            },
            machine_calendars={
                0: MachineCalendar(machine_id=0, available_time=17.0),
                1: MachineCalendar(machine_id=1, available_time=13.0),
            },
            current_time=0.0,
        )

        self.assertAlmostEqual(0.5, compute_changed_operation_ratio(incumbent))
        self.assertAlmostEqual(0.25, compute_changed_machine_ratio(incumbent))
        self.assertAlmostEqual(1.0, compute_mean_absolute_start_time_deviation(incumbent))


if __name__ == "__main__":
    unittest.main()
