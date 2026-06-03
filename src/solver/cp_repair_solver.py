from __future__ import annotations

from copy import deepcopy

from ortools.sat.python import cp_model

from src.data.schema import ProblemInstance
from src.events.schema import DynamicEvent
from src.scheduling.incumbent import IncumbentSchedule, OperationSchedule
from src.scheduling.window import RollingWindow
from src.solver.base import RepairDecision, RepairSolverResult


class CPRepairSolver:
    """
    Minimal CP-SAT repair solver.

    Implementation assumptions in the minimal version:
    - kept operations are fixed by default if configured
    - instability is approximated with start-time deviation and machine-change penalties
    - machine breakdown only blocks released/unscheduled operations inside forbidden intervals
    """

    def __init__(self, solver_cfg: dict):
        self.cfg = solver_cfg

    def _horizon_upper_bound(self, instance: ProblemInstance, incumbent: IncumbentSchedule) -> int:
        scheduled_max = max((sched.end_time or 0.0) for sched in incumbent.operations.values())
        remaining = sum(op.min_processing_time for op in instance.iter_operations())
        return int(round(scheduled_max + remaining + 100))

    def solve(self, subproblem: dict) -> RepairSolverResult:
        instance: ProblemInstance = subproblem["instance"]
        incumbent: IncumbentSchedule = subproblem["incumbent"]
        window: RollingWindow = subproblem["window"]
        event: DynamicEvent = subproblem["event"]
        decision: RepairDecision = subproblem["decision"]

        model = cp_model.CpModel()
        horizon = self._horizon_upper_bound(instance, incumbent)
        trust_region_backward_ratio = float(
            decision.metadata.get("trust_region_backward_ratio", self.cfg.get("trust_region_backward_ratio", 0.1))
        )
        trust_region_forward_ratio = float(
            decision.metadata.get("trust_region_forward_ratio", self.cfg.get("trust_region_forward_ratio", 0.3))
        )

        released_set = set(decision.released_op_ids)
        kept_set = set(decision.kept_op_ids)
        immutable_set = set(decision.immutable_op_ids)
        considered_ops = sorted(set(window.op_ids) | released_set | kept_set)

        start_vars = {}
        end_vars = {}
        machine_choice = {}
        intervals_by_machine: dict[int, list] = {m.machine_id: [] for m in instance.machines}
        machine_presence: dict[tuple[int, int], cp_model.IntVar] = {}
        machine_change_terms: list[tuple[cp_model.IntVar, int]] = []
        start_shift_terms: list[tuple[cp_model.IntVar, int]] = []

        # Block machine capacity with already scheduled operations that remain outside
        # the repair model but still occupy future machine time.
        for op_id, sched in incumbent.operations.items():
            if op_id in considered_ops:
                continue
            if sched.machine_id is None or sched.start_time is None or sched.end_time is None:
                continue
            if sched.end_time <= event.time:
                continue
            start_fixed = int(round(sched.start_time))
            end_fixed = int(round(sched.end_time))
            duration = max(0, end_fixed - start_fixed)
            if duration == 0:
                continue
            interval = model.NewIntervalVar(
                model.NewIntVar(start_fixed, start_fixed, f"fixed_out_start_{op_id}"),
                duration,
                model.NewIntVar(end_fixed, end_fixed, f"fixed_out_end_{op_id}"),
                f"fixed_out_interval_{op_id}_{sched.machine_id}",
            )
            intervals_by_machine[sched.machine_id].append(interval)

        for op_id in considered_ops:
            op = instance.get_operation(op_id)
            sched = incumbent.operations[op_id]

            if op_id in immutable_set:
                fixed_start = int(round(sched.start_time or 0))
                fixed_end = int(round(sched.end_time or fixed_start))
                start_vars[op_id] = model.NewIntVar(fixed_start, fixed_start, f"start_{op_id}")
                end_vars[op_id] = model.NewIntVar(fixed_end, fixed_end, f"end_{op_id}")
                if sched.machine_id is not None:
                    duration = max(0, fixed_end - fixed_start)
                    if duration > 0:
                        interval = model.NewIntervalVar(
                            start_vars[op_id],
                            duration,
                            end_vars[op_id],
                            f"interval_{op_id}_{sched.machine_id}",
                        )
                        intervals_by_machine[sched.machine_id].append(interval)
                continue

            min_start = int(round(max(op.release_time, event.time)))
            start_vars[op_id] = model.NewIntVar(min_start, horizon, f"start_{op_id}")
            end_vars[op_id] = model.NewIntVar(min_start, horizon, f"end_{op_id}")

            if op_id in kept_set and self.cfg.get("fix_kept_operations", True) and sched.start_time is not None and sched.end_time is not None:
                if sched.machine_id is not None:
                    machine_choice[op_id] = sched.machine_id
                    duration = int(round(instance.get_operation(op_id).processing_time_on(sched.machine_id)))
                    original_start = int(round(sched.start_time))
                    delta_minus = int(round(trust_region_backward_ratio * duration))
                    delta_plus = int(round(trust_region_forward_ratio * duration))
                    lower_bound = max(min_start, original_start - delta_minus)
                    upper_bound = max(lower_bound, min(horizon, original_start + delta_plus))
                    start_vars[op_id] = model.NewIntVar(lower_bound, upper_bound, f"start_{op_id}")
                    end_vars[op_id] = model.NewIntVar(lower_bound + duration, upper_bound + duration, f"end_{op_id}")
                    interval = model.NewIntervalVar(start_vars[op_id], duration, end_vars[op_id], f"interval_{op_id}_{sched.machine_id}")
                    intervals_by_machine[sched.machine_id].append(interval)
                else:
                    fixed_start = int(round(sched.start_time))
                    fixed_end = int(round(sched.end_time))
                    model.Add(start_vars[op_id] == fixed_start)
                    model.Add(end_vars[op_id] == fixed_end)
                continue

            local_presence_vars = []
            local_end_vars = []
            for option in op.options:
                presence = model.NewBoolVar(f"presence_{op_id}_{option.machine_id}")
                machine_presence[(op_id, option.machine_id)] = presence
                duration = int(round(option.processing_time))
                interval = model.NewOptionalIntervalVar(
                    start_vars[op_id],
                    duration,
                    end_vars[op_id],
                    presence,
                    f"interval_{op_id}_{option.machine_id}",
                )
                intervals_by_machine[option.machine_id].append(interval)
                local_presence_vars.append(presence)
                local_end_vars.append(end_vars[op_id])
            model.AddExactlyOne(local_presence_vars)

            if sched.original_start_time is not None:
                shift = model.NewIntVar(0, horizon, f"shift_{op_id}")
                model.AddAbsEquality(shift, start_vars[op_id] - int(round(sched.original_start_time)))
                reference_duration = 1
                if sched.original_end_time is not None and sched.original_start_time is not None:
                    reference_duration = max(1, int(round(sched.original_end_time - sched.original_start_time)))
                start_shift_terms.append((shift, reference_duration))

            if sched.original_machine_id is not None and len(op.options) > 1:
                changed = model.NewBoolVar(f"machine_changed_{op_id}")
                if (op_id, sched.original_machine_id) in machine_presence:
                    model.Add(machine_presence[(op_id, sched.original_machine_id)] == 0).OnlyEnforceIf(changed)
                    model.Add(machine_presence[(op_id, sched.original_machine_id)] == 1).OnlyEnforceIf(changed.Not())
                else:
                    model.Add(changed == 1)
                machine_change_terms.append((changed, 1))

        for src, dst in instance.precedence_pairs():
            if src in end_vars and dst in start_vars:
                model.Add(start_vars[dst] >= end_vars[src])
            elif dst in start_vars and src not in end_vars:
                predecessor_sched = incumbent.operations.get(src)
                if predecessor_sched is not None and predecessor_sched.end_time is not None:
                    model.Add(start_vars[dst] >= int(round(predecessor_sched.end_time)))

        for machine_id, intervals in intervals_by_machine.items():
            if intervals:
                model.AddNoOverlap(intervals)

        for machine_id, calendar in incumbent.machine_calendars.items():
            for start, end in calendar.breakdowns:
                for op_id in considered_ops:
                    op = instance.get_operation(op_id)
                    for option in op.options:
                        if option.machine_id != machine_id or (op_id, machine_id) not in machine_presence:
                            continue
                        presence = machine_presence[(op_id, machine_id)]
                        before = model.NewBoolVar(f"bd_before_{op_id}_{machine_id}_{int(start)}")
                        after = model.NewBoolVar(f"bd_after_{op_id}_{machine_id}_{int(start)}")
                        model.Add(end_vars[op_id] <= int(round(start))).OnlyEnforceIf(before)
                        model.Add(start_vars[op_id] >= int(round(end))).OnlyEnforceIf(after)
                        model.AddBoolOr([before, after]).OnlyEnforceIf(presence)

        makespan = model.NewIntVar(0, horizon, "makespan")
        model.AddMaxEquality(makespan, [end_vars[op_id] for op_id in end_vars])

        tardiness_terms = []
        for job in instance.jobs:
            due = int(round(job.due_date or 0))
            job_end = model.NewIntVar(0, horizon, f"job_end_{job.job_id}")
            job_op_ends = [end_vars[op.op_global_id] for op in job.operations if op.op_global_id in end_vars]
            if not job_op_ends:
                continue
            model.AddMaxEquality(job_end, job_op_ends)
            tardiness = model.NewIntVar(0, horizon, f"tardiness_{job.job_id}")
            model.Add(tardiness >= job_end - due)
            tardiness_terms.append(tardiness)

        weights = self.cfg.get("objective_weights", {})
        instability_weight = float(weights.get("instability", 0.0))
        makespan_weight = float(weights.get("makespan", 1.0))
        tardiness_weight = float(weights.get("tardiness", 1.0))
        instability_cfg = self.cfg.get("instability", {})
        start_shift_weight = float(instability_cfg.get("start_shift_weight", 1.0))
        machine_change_weight = float(instability_cfg.get("machine_change_weight", 2.0))

        objective_terms = [int(round(makespan_weight * 100)) * makespan]
        if tardiness_terms:
            objective_terms.append(int(round(tardiness_weight * 100)) * sum(tardiness_terms))
        if start_shift_terms:
            for shift, duration in start_shift_terms:
                coef = int(round(instability_weight * start_shift_weight * 100 / max(1, duration)))
                if coef > 0:
                    objective_terms.append(coef * shift)
        if machine_change_terms:
            for changed, base_weight in machine_change_terms:
                coef = int(round(instability_weight * machine_change_weight * 100 * base_weight))
                if coef > 0:
                    objective_terms.append(coef * changed)
        model.Minimize(sum(objective_terms))

        use_incumbent_hints = bool(self.cfg.get("use_incumbent_hints", False)) or bool(
            decision.metadata.get("use_incumbent_hints", False)
        )
        if use_incumbent_hints and bool(self.cfg.get("warm_start", True)) and bool(decision.metadata.get("warm_start", True)):
            self._add_incumbent_hints(model, incumbent, start_vars, end_vars, machine_presence)

        solver = cp_model.CpSolver()
        requested_time_limit_sec = float(
            decision.metadata.get("solver_time_limit_sec", self.cfg.get("time_limit_sec", 5.0))
        )
        solver.parameters.max_time_in_seconds = max(0.01, requested_time_limit_sec)
        solver.parameters.num_search_workers = int(self.cfg.get("num_workers", 8))
        solver.parameters.random_seed = int(self.cfg.get("random_seed", 0))
        status = solver.Solve(model)

        feasible = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        updated_operations: dict[int, OperationSchedule] = {}
        if feasible:
            for op_id in considered_ops:
                original = incumbent.operations[op_id]
                updated = deepcopy(original)
                updated.start_time = float(solver.Value(start_vars[op_id]))
                updated.end_time = float(solver.Value(end_vars[op_id]))

                if op_id in immutable_set or (op_id in kept_set and self.cfg.get("fix_kept_operations", True) and original.machine_id is not None):
                    updated.machine_id = original.machine_id
                else:
                    chosen_machine = original.machine_id
                    for option in instance.get_operation(op_id).options:
                        presence = machine_presence.get((op_id, option.machine_id))
                        if presence is not None and solver.Value(presence) == 1:
                            chosen_machine = option.machine_id
                            break
                    updated.machine_id = chosen_machine
                updated_operations[op_id] = updated

        return RepairSolverResult(
            feasible=feasible,
            solver_status=str(status),
            updated_operations=updated_operations,
            objective_value=solver.ObjectiveValue() if feasible else None,
            metadata={
                "considered_ops": considered_ops,
                "time_limit_sec": float(max(0.01, requested_time_limit_sec)),
                "solver_wall_time_sec": float(solver.WallTime()),
                "solver_user_time_sec": float(solver.UserTime()),
            },
        )

    @staticmethod
    def _nearest_domain_value(variable: cp_model.IntVar, value: int) -> int:
        domain = list(variable.Proto().domain)
        if not domain:
            return int(value)
        for index in range(0, len(domain), 2):
            lower = int(domain[index])
            upper = int(domain[index + 1])
            if lower <= int(value) <= upper:
                return int(value)
        candidates = [int(bound) for bound in domain]
        return min(candidates, key=lambda bound: (abs(bound - int(value)), bound))

    @classmethod
    def _add_incumbent_hints(
        cls,
        model: cp_model.CpModel,
        incumbent: IncumbentSchedule,
        start_vars: dict[int, cp_model.IntVar],
        end_vars: dict[int, cp_model.IntVar],
        machine_presence: dict[tuple[int, int], cp_model.IntVar],
    ) -> None:
        for op_id, start_var in start_vars.items():
            schedule = incumbent.operations.get(op_id)
            if schedule is None:
                continue
            if schedule.start_time is not None:
                model.AddHint(start_var, cls._nearest_domain_value(start_var, int(round(schedule.start_time))))
            end_var = end_vars.get(op_id)
            if end_var is not None and schedule.end_time is not None:
                model.AddHint(end_var, cls._nearest_domain_value(end_var, int(round(schedule.end_time))))
        for (op_id, machine_id), presence_var in machine_presence.items():
            schedule = incumbent.operations.get(op_id)
            hinted = 1 if schedule is not None and schedule.machine_id == machine_id else 0
            model.AddHint(presence_var, hinted)
