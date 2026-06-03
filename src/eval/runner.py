from __future__ import annotations

from typing import Callable

import pandas as pd

from src.baselines.ddpg import ddpg_decision, load_ddpg_bundle
from src.baselines.daniel_local import daniel_local_decision
from src.baselines.dispatching import dispatching_release_decision
from src.baselines.full_reopt import full_reoptimization_decision
from src.baselines.heuristic_rh import heuristic_rh_decision
from src.baselines.learned_rule_selector import learned_rule_selector_decision, load_selector_bundle
from src.baselines.no_learning_repair import no_learning_repair_decision
from src.eval.metrics import compute_instability_components, evaluate_schedule


def _build_event_row(
    env,
    event,
    method: str,
    runtime_sec: float,
    changed_op_ratio: float,
    changed_machine_ratio: float,
    feasible: bool,
    event_idx: int,
    released_neighborhood_size: int,
    export_extended: bool,
) -> dict:
    metrics = evaluate_schedule(
        env.instance,
        env.incumbent,
        runtime_sec=runtime_sec,
        changed_op_ratio=changed_op_ratio,
        changed_machine_ratio=changed_machine_ratio,
        feasible=feasible,
    )
    row = {"event_time": event.time, "event_type": event.event_type, "method": method, **metrics}
    if export_extended:
        start_disp, machine_reassignment = compute_instability_components(env.incumbent)
        row.update(
            {
                "event_idx": event_idx,
                "tardiness": metrics["total_tardiness"],
                "feasibility": metrics["feasibility_rate"],
                "original_instability": metrics["instability"],
                "start_time_displacement_component": start_disp,
                "machine_reassignment_component": machine_reassignment,
                "released_neighborhood_size": released_neighborhood_size,
            }
        )
    return row


def _append_noop_row(rows: list[dict], env, event, method: str, event_idx: int = -1, export_extended: bool = False) -> None:
    rows.append(
        _build_event_row(
            env=env,
            event=event,
            method=method,
            runtime_sec=0.0,
            changed_op_ratio=0.0,
            changed_machine_ratio=0.0,
            feasible=True,
            event_idx=event_idx,
            released_neighborhood_size=0,
            export_extended=export_extended,
        )
    )


def run_baseline_episode(env, events, solver, baseline_name: str, export_extended: bool = False) -> pd.DataFrame:
    rows = []
    env.reset()
    selector_bundle = None
    ddpg_bundle = None
    if baseline_name == "learned_rule_selector":
        selector_bundle = load_selector_bundle(env.config)
    if baseline_name == "ddpg":
        ddpg_bundle = load_ddpg_bundle(env.config)
    for event_idx, event in enumerate(events):
        env.apply_event(event)
        env.build_window()
        snapshot = env.state_snapshot
        if not snapshot.window_op_ids:
            _append_noop_row(rows, env, event, baseline_name, event_idx=event_idx, export_extended=export_extended)
            continue
        if baseline_name == "dispatching_spt":
            baseline = dispatching_release_decision(env.instance, env.incumbent, snapshot, rule="SPT")
        elif baseline_name == "dispatching_mwkr":
            baseline = dispatching_release_decision(env.instance, env.incumbent, snapshot, rule="MWKR")
        elif baseline_name == "dispatching_edd":
            baseline = dispatching_release_decision(env.instance, env.incumbent, snapshot, rule="EDD")
        elif baseline_name == "dispatching_cr":
            baseline = dispatching_release_decision(env.instance, env.incumbent, snapshot, rule="CR")
        elif baseline_name == "dispatching_atc":
            baseline = dispatching_release_decision(env.instance, env.incumbent, snapshot, rule="ATC")
        elif baseline_name == "ddpg":
            graph = env.export_state_for_policy()
            baseline = ddpg_decision(env.instance, env.incumbent, snapshot, graph, ddpg_bundle)
        elif baseline_name == "daniel_local":
            baseline = daniel_local_decision(env.instance, env.incumbent, snapshot, env.config)
        elif baseline_name == "learned_rule_selector":
            graph = env.export_state_for_policy()
            baseline = learned_rule_selector_decision(env.instance, env.incumbent, snapshot, graph, selector_bundle)
        elif baseline_name == "heuristic_rh":
            baseline = heuristic_rh_decision(snapshot)
        elif baseline_name == "full_reoptimization":
            baseline = full_reoptimization_decision(snapshot)
        elif baseline_name == "no_learning_repair":
            baseline = no_learning_repair_decision(snapshot)
        else:
            raise ValueError(f"Unknown baseline: {baseline_name}")
        step = env.step_reschedule(solver, baseline.decision)
        rows.append(
            _build_event_row(
                env=env,
                event=event,
                method=baseline.name,
                runtime_sec=step.runtime_sec,
                changed_op_ratio=step.changed_op_ratio,
                changed_machine_ratio=step.changed_machine_ratio,
                feasible=step.feasible,
                event_idx=event_idx,
                released_neighborhood_size=len(baseline.decision.released_op_ids),
                export_extended=export_extended,
            )
        )
    return pd.DataFrame(rows)


def run_policy_episode(env, events, solver, policy, decision_builder: Callable, export_extended: bool = False) -> pd.DataFrame:
    rows = []
    env.reset()
    for event_idx, event in enumerate(events):
        env.apply_event(event)
        env.build_window()
        graph = env.export_state_for_policy()
        if not graph["op_ids"]:
            _append_noop_row(rows, env, event, "policy", event_idx=event_idx, export_extended=export_extended)
            continue
        outputs = policy(graph)
        decision = decision_builder(
            op_ids=graph["op_ids"],
            keep_probs=outputs["keep_probs"],
            release_probs=outputs["release_probs"],
            snapshot=graph["snapshot"],
            graph=graph,
        )
        step = env.step_reschedule(solver, decision)
        rows.append(
            _build_event_row(
                env=env,
                event=event,
                method="policy",
                runtime_sec=step.runtime_sec,
                changed_op_ratio=step.changed_op_ratio,
                changed_machine_ratio=step.changed_machine_ratio,
                feasible=step.feasible,
                event_idx=event_idx,
                released_neighborhood_size=len(decision.released_op_ids),
                export_extended=export_extended,
            )
        )
    return pd.DataFrame(rows)
