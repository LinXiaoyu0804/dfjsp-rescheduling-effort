from __future__ import annotations

import unittest

from src.baselines.full_reopt import full_reoptimization_decision
from src.baselines.heuristic_rh import heuristic_rh_decision
from src.data.schema import Job, Machine, Operation, OperationOption, ProblemInstance, assign_due_dates_by_factor
from src.events.schema import DynamicEvent
from src.scheduling.incumbent import IncumbentSchedule, MachineCalendar, OperationSchedule
from src.scheduling.intensity_ladder import build_intensity_release_sets, repair_at_intensity
from src.scheduling.state_builder import StateSnapshot


class IntensityLadderTests(unittest.TestCase):
    def _build_instance(self) -> ProblemInstance:
        jobs: list[Job] = []
        op_id = 0
        for job_id in range(6):
            operations: list[Operation] = []
            for op_index in range(3):
                operations.append(
                    Operation(
                        op_global_id=op_id,
                        job_id=job_id,
                        op_index=op_index,
                        options=[
                            OperationOption(machine_id=op_index % 3, processing_time=3.0 + op_index),
                            OperationOption(machine_id=(op_index + 1) % 3, processing_time=4.0 + op_index),
                        ],
                    )
                )
                op_id += 1
            jobs.append(Job(job_id=job_id, operations=operations))
        return assign_due_dates_by_factor(
            ProblemInstance(family="fjsp", jobs=jobs, machines=[Machine(0), Machine(1), Machine(2)]),
            2.0,
        )

    def _build_incumbent(self, instance: ProblemInstance) -> IncumbentSchedule:
        operations: dict[int, OperationSchedule] = {}
        for op in instance.iter_operations():
            start = float(op.op_global_id * 3)
            end = start + op.min_processing_time
            machine_id = op.eligible_machine_ids[0]
            operations[op.op_global_id] = OperationSchedule(
                op_global_id=op.op_global_id,
                job_id=op.job_id,
                op_index=op.op_index,
                machine_id=machine_id,
                start_time=start,
                end_time=end,
                status="unstarted",
                original_start_time=start,
                original_end_time=end,
                original_machine_id=machine_id,
            )
        calendars = {machine.machine_id: MachineCalendar(machine.machine_id) for machine in instance.machines}
        return IncumbentSchedule(operations=operations, machine_calendars=calendars, current_time=5.0)

    def _build_snapshot(self, instance: ProblemInstance) -> StateSnapshot:
        op_ids = [op.op_global_id for op in instance.iter_operations()]
        return StateSnapshot(
            current_time=5.0,
            completed_op_ids=[],
            active_op_ids=[],
            unfinished_op_ids=op_ids,
            window_op_ids=op_ids,
            directly_impacted_op_ids=[0, 6],
            affected_machine_ids=[0],
            triggering_event_type="machine_breakdown",
        )

    def test_release_sets_are_monotone_and_match_endpoints(self) -> None:
        instance = self._build_instance()
        incumbent = self._build_incumbent(instance)
        snapshot = self._build_snapshot(instance)
        event = DynamicEvent(
            event_id=0,
            time=5.0,
            event_type="machine_breakdown",
            payload={"machine_id": 0, "start_time": 5.0, "end_time": 8.0},
        )

        release_sets = build_intensity_release_sets(
            instance=instance,
            incumbent=incumbent,
            snapshot=snapshot,
            event=event,
            instance_id="toy",
            seed=0,
            episode_id="toy_seed00_ep001",
            event_id="bd_0",
        )

        self.assertEqual(set(release_sets["L0"]), set(heuristic_rh_decision(snapshot).decision.released_op_ids))
        self.assertEqual(set(release_sets["L3"]), set(full_reoptimization_decision(snapshot).decision.released_op_ids))
        self.assertTrue(set(release_sets["L0"]).issubset(release_sets["L1"]))
        self.assertTrue(set(release_sets["L1"]).issubset(release_sets["L2"]))
        self.assertTrue(set(release_sets["L2"]).issubset(release_sets["L3"]))

    def test_repair_at_intensity_sets_budget_and_trust_region_metadata(self) -> None:
        instance = self._build_instance()
        incumbent = self._build_incumbent(instance)
        snapshot = self._build_snapshot(instance)
        event = DynamicEvent(
            event_id=0,
            time=5.0,
            event_type="machine_breakdown",
            payload={"machine_id": 0, "start_time": 5.0, "end_time": 8.0},
        )

        plan = repair_at_intensity(
            incumbent=incumbent,
            event=event,
            budget_sec=2.0,
            level="L1",
            instance=instance,
            snapshot=snapshot,
        )

        self.assertEqual(plan.level, "L1")
        self.assertEqual(plan.decision.metadata["solver_time_limit_sec"], 2.0)
        self.assertLess(plan.decision.metadata["trust_region_forward_ratio"], 0.3)
        self.assertEqual(plan.forced_release_op_ids, [0, 6])


if __name__ == "__main__":
    unittest.main()
