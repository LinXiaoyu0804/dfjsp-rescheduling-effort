from __future__ import annotations

import argparse
import time
from copy import deepcopy
from pathlib import Path

from _bootstrap import REPO_ROOT  # noqa: F401

from src.data.unified_parser import parse_instance
from src.env.dfjsp_env import DFJSPReschedulingEnv
from src.eval.event_logging import build_event_log_row
from src.eval.external_baselines import build_external_baseline_output, compute_baseline_proxy_selection
from src.eval.event_summary import build_complexity_summary, build_instance_group_summary, build_instance_summary
from src.eval.metrics import compute_mean_absolute_start_time_deviation
from src.events.serialization import deserialize_dynamic_event
from src.motifs.runtime import build_runtime_snapshot_row
from src.scheduling.incumbent_builder import load_incumbent_schedule
from src.solver.cp_repair_solver import CPRepairSolver
from src.utils.config import load_merged_config
from src.utils.io import ensure_dir, load_json, save_json, save_jsonl
from src.utils.seed import set_global_seed


def _normalize_event_type_key(name: str) -> str:
    return str(name).strip().lower()


def _event_type_bucket_from_runtime_snapshot(runtime_snapshot: dict) -> str:
    event_context = runtime_snapshot.get("event_context", {})
    event_type = _normalize_event_type_key(str(event_context.get("type", "")))
    if "arrival" in event_type:
        return "arrival"
    if "breakdown" in event_type:
        return "breakdown"
    event_id = _normalize_event_type_key(str(runtime_snapshot.get("event_id", "")))
    if event_id.startswith("arr_"):
        return "arrival"
    if event_id.startswith("bd_"):
        return "breakdown"
    return "other"


def _repair_status_from_step(*, feasible: bool, metadata: dict) -> str:
    if feasible:
        return "feasible"
    solver_status = str(metadata.get("solver_status", "")).strip().upper()
    if bool(metadata.get("budget_violation", False)) or solver_status in {"0", "UNKNOWN"}:
        return "timeout"
    return "infeasible"


def _resolve_path(path_like: str | Path) -> Path:
    # Normalize Windows-style separators so episode paths stay cross-platform.
    path = Path(str(path_like).replace("\\", "/"))
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _evaluate_baseline(
    *,
    cfg: dict,
    baseline_name: str,
    episodes_dir: Path,
    output_path: Path,
) -> None:
    due_factor = float(cfg.get("data", {}).get("due_date_rule", {}).get("factor", 1.5))
    solver = CPRepairSolver(cfg["solver"])
    rows: list[dict] = []
    baseline_cache: dict[str, object] = {}

    for episode_path in sorted(episodes_dir.glob("*.json")):
        episode_data = load_json(episode_path)
        if not isinstance(episode_data, dict) or "episode_id" not in episode_data or "incumbent_ref" not in episode_data:
            continue

        instance = parse_instance(
            _resolve_path(episode_data["instance_path"]),
            family=cfg.get("data", {}).get("family", "fjsp"),
            due_date_factor=due_factor,
        )
        env = DFJSPReschedulingEnv(instance, cfg)
        env.reset()
        incumbent_data = load_json(_resolve_path(episode_data["incumbent_ref"]))
        env.incumbent = load_incumbent_schedule(instance, incumbent_data)
        env.initial_instance = deepcopy(instance)
        env.instance = deepcopy(instance)

        for fallback_id, event_data in enumerate(episode_data.get("events", [])):
            event = deserialize_dynamic_event(event_data, fallback_event_id=fallback_id)
            env.apply_event(event)
            env.build_window()
            snapshot = env.state_snapshot
            if snapshot is None:
                raise RuntimeError("State snapshot is not available during baseline evaluation.")
            objective_before = env.compute_objective()
            runtime_snapshot = build_runtime_snapshot_row(
                instance_id=str(episode_data["instance_id"]),
                seed=int(episode_data["seed"]),
                episode_id=str(episode_data["episode_id"]),
                event_id=str(event_data["event_id"]),
                env=env,
            )
            runtime_snapshot["event_type_bucket"] = _event_type_bucket_from_runtime_snapshot(runtime_snapshot)
            graph = env.export_state_for_policy()
            selector_start = time.perf_counter()
            baseline = build_external_baseline_output(
                baseline_name=baseline_name,
                instance=env.instance,
                incumbent=env.incumbent,
                snapshot=snapshot,
                graph=graph,
                cfg=cfg,
                device=str(cfg["experiment"].get("device", "cpu")),
                cache=baseline_cache,
            )
            selector_runtime_sec = time.perf_counter() - selector_start
            step = env.step_reschedule(solver, baseline.decision)
            objective = step.objective
            intervention_count, extra_release_count = compute_baseline_proxy_selection(
                baseline.decision.released_op_ids,
                runtime_snapshot["forced_release_ops"],
            )
            rows.append(
                build_event_log_row(
                    method=str(baseline.name),
                    instance_id=str(episode_data["instance_id"]),
                    seed=int(episode_data["seed"]),
                    episode_id=str(episode_data["episode_id"]),
                    event_id=str(event_data["event_id"]),
                    tau=float(event.time),
                    budget_sec=float(cfg["solver"].get("time_limit_sec", 0.0)),
                    window_size=len(snapshot.window_op_ids),
                    forced_release_count=len(runtime_snapshot["forced_release_ops"]),
                    motif_count=0,
                    selected_motif_count=int(intervention_count),
                    released_op_count=len(baseline.decision.released_op_ids),
                    pred_gain_sum=None,
                    inference_runtime_sec=0.0,
                    selector_runtime_sec=selector_runtime_sec,
                    solver_runtime_sec=float(step.runtime_sec),
                    makespan_after=objective.makespan,
                    tardiness_after=objective.total_tardiness,
                    instability_after=objective.instability,
                    weighted_objective_after=objective.weighted_sum,
                    changed_op_ratio=float(step.changed_op_ratio),
                    changed_machine_ratio=float(step.changed_machine_ratio),
                    mean_abs_start_time_deviation=compute_mean_absolute_start_time_deviation(env.incumbent),
                    status=_repair_status_from_step(feasible=bool(step.feasible), metadata=dict(step.metadata)),
                    extra={
                        "selection_source": str(baseline.name),
                        "operator_selection_mode": str(baseline_name),
                        "controller_mode": "external_baseline",
                        "objective_before": float(objective_before.weighted_sum),
                        "reward_delta": float(objective_before.weighted_sum - objective.weighted_sum),
                        "forced_release_ops": runtime_snapshot["forced_release_ops"],
                        "released_op_ids": baseline.decision.released_op_ids,
                        "baseline_name": str(baseline_name),
                        "baseline_decision_metadata": dict(baseline.decision.metadata),
                        "baseline_extra_release_count": int(extra_release_count),
                        "event_type_bucket": runtime_snapshot.get("event_type_bucket", "other"),
                        "solver_status": str(step.metadata.get("solver_status", "")),
                        "solver_runtime_sec_raw": step.metadata.get("raw_wall_time_sec"),
                        "solver_wall_time_sec": step.metadata.get("solver_wall_time_sec"),
                        "solver_user_time_sec": step.metadata.get("solver_user_time_sec"),
                        "solver_runtime_budget_cap_sec": step.metadata.get("runtime_budget_cap_sec"),
                        "solver_runtime_accounting_source": step.metadata.get("runtime_accounting_source"),
                        "solver_runtime_timing_anomaly": step.metadata.get("raw_timing_anomaly"),
                        "solver_runtime_clipped": step.metadata.get("runtime_clipped"),
                        "solver_budget_violation": step.metadata.get("budget_violation"),
                    },
                )
            )

    ensure_dir(output_path.parent)
    save_jsonl(rows, output_path)
    build_instance_summary(rows).to_csv(output_path.with_suffix(".summary.csv"), index=False)
    build_instance_group_summary(rows).to_csv(output_path.with_suffix(".instance_groups.csv"), index=False)
    build_complexity_summary(rows).to_csv(output_path.with_suffix(".complexity.csv"), index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs="*", default=[])
    parser.add_argument("--baselines", nargs="+", required=True)
    parser.add_argument("--eval-episodes-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    cfg = load_merged_config(*args.config) if args.config else {}
    set_global_seed(int(cfg["experiment"]["seed"]))
    episodes_dir = _resolve_path(args.eval_episodes_dir)
    output_dir = _resolve_path(args.output_dir)
    ensure_dir(output_dir)

    written_outputs: list[dict[str, str]] = []
    for baseline_name in args.baselines:
        output_path = output_dir / f"{baseline_name}_event_metrics.jsonl"
        _evaluate_baseline(
            cfg=cfg,
            baseline_name=str(baseline_name),
            episodes_dir=episodes_dir,
            output_path=output_path,
        )
        written_outputs.append(
            {
                "baseline": str(baseline_name),
                "output_path": str(output_path),
            }
        )

    save_json(
        {
            "eval_episodes_dir": str(episodes_dir),
            "output_dir": str(output_dir),
            "baselines": [str(name) for name in args.baselines],
            "written_outputs": written_outputs,
        },
        output_dir / "run_manifest.json",
    )


if __name__ == "__main__":
    main()
