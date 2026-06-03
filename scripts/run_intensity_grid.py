from __future__ import annotations

import argparse
import csv
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from _bootstrap import REPO_ROOT  # noqa: F401

from src.baselines.full_reopt import full_reoptimization_decision
from src.baselines.heuristic_rh import heuristic_rh_decision
from src.data.unified_parser import parse_instance
from src.env.dfjsp_env import DFJSPReschedulingEnv
from src.eval.event_logging import build_event_log_row
from src.eval.metrics import compute_mean_absolute_start_time_deviation
from src.events.serialization import deserialize_dynamic_event
from src.scheduling.incumbent_builder import load_incumbent_schedule
from src.scheduling.intensity_ladder import (
    INTENSITY_DEFINITIONS,
    forced_release_ops,
    repair_at_intensity,
)
from src.solver.base import RepairDecision
from src.solver.cp_repair_solver import CPRepairSolver
from src.utils.config import load_merged_config
from src.utils.io import ensure_dir, load_json, load_jsonl, save_json, save_jsonl
from src.utils.seed import set_global_seed


DEFAULT_BUDGETS = [0.25, 0.5, 1.0, 2.0, 5.0]
DEFAULT_INTENSITIES = ["L0", "L1", "L2", "L3"]
DEFAULT_DISTURBANCES = ["arrival_only", "breakdown_only", "mixed"]
DEFAULT_SCALES = [
    "mk6",
    "mk7",
    "mk8",
    "mk9",
    "mk10",
    "synthetic_30x10",
    "synthetic_50x15",
    "synthetic_100x20",
]
BASE_CONFIGS = [
    "configs/default.yaml",
    "configs/env/formal_dynamic_stronger_v2.yaml",
    "configs/solver/cp_repair_default.yaml",
]


@dataclass(frozen=True)
class ScaleSpec:
    label: str
    source_episodes_dir: str
    instance_ids: tuple[str, ...] = ()
    instance_prefix: str = ""


def _resolve_path(path_like: str | Path) -> Path:
    # Normalize Windows-style separators so episode paths stay cross-platform.
    path = Path(str(path_like).replace("\\", "/"))
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _format_budget(budget_sec: float) -> str:
    return f"{float(budget_sec):g}".replace(".", "p")


def _default_scale_specs() -> dict[str, ScaleSpec]:
    brandimarte = "outputs/episodes/brandimarte_heldout/episodes"
    synthetic_root = "outputs/episodes"
    return {
        "mk6": ScaleSpec("mk6", brandimarte, instance_ids=("mk6",)),
        "mk7": ScaleSpec("mk7", brandimarte, instance_ids=("mk7",)),
        "mk8": ScaleSpec("mk8", brandimarte, instance_ids=("mk8",)),
        "mk9": ScaleSpec("mk9", brandimarte, instance_ids=("mk9",)),
        "mk10": ScaleSpec("mk10", brandimarte, instance_ids=("mk10",)),
        "synthetic_30x10": ScaleSpec(
            "synthetic_30x10",
            f"{synthetic_root}/synthetic_30x10/episodes",
            instance_prefix="syn_30x10_",
        ),
        "synthetic_50x15": ScaleSpec(
            "synthetic_50x15",
            f"{synthetic_root}/synthetic_50x15/episodes",
            instance_prefix="syn_50x15_",
        ),
        "synthetic_100x20": ScaleSpec(
            "synthetic_100x20",
            f"{synthetic_root}/synthetic_100x20/episodes",
            instance_prefix="syn_100x20_",
        ),
    }


def _parse_filter(values: list[str] | None, default: list[str]) -> list[str]:
    if not values:
        return list(default)
    parsed: list[str] = []
    for value in values:
        parsed.extend(part.strip() for part in str(value).split(",") if part.strip())
    return parsed


def _build_cfg(budget_sec: float) -> dict[str, Any]:
    cfg = load_merged_config(*(_resolve_path(path) for path in BASE_CONFIGS))
    cfg.setdefault("experiment", {})["seed"] = 0
    solver_cfg = cfg.setdefault("solver", {})
    solver_cfg["time_limit_sec"] = float(budget_sec)
    solver_cfg["runtime_budget_guard_multiplier"] = 1.0
    solver_cfg["runtime_budget_guard_slack_sec"] = 0.0
    solver_cfg["enforce_event_wall_time_budget"] = True
    cfg.setdefault("eval", {})["enforce_event_wall_time_budget"] = True
    return cfg


def _write_run_config(
    output_root: Path,
    *,
    intensity: str,
    budget_sec: float,
    scale: str,
    disturbance: str,
    cfg: dict[str, Any],
) -> Path:
    config_dir = ensure_dir(output_root / "_configs")
    path = config_dir / f"{intensity}__b{_format_budget(budget_sec)}__{scale}__{disturbance}.json"
    save_json(
        {
            "base_configs": BASE_CONFIGS,
            "intensity": intensity,
            "budget_sec": float(budget_sec),
            "scale": scale,
            "disturbance": disturbance,
            "effective_config": cfg,
        },
        path,
    )
    return path


def _episode_matches_scale(episode_data: dict[str, Any], scale: ScaleSpec) -> bool:
    instance_id = str(episode_data.get("instance_id", ""))
    if scale.instance_ids:
        return instance_id in set(scale.instance_ids)
    if scale.instance_prefix:
        return instance_id.startswith(scale.instance_prefix)
    return True


def _iter_episode_payloads(scale: ScaleSpec) -> Iterable[tuple[Path, dict[str, Any]]]:
    source_dir = _resolve_path(scale.source_episodes_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"Frozen episode directory does not exist: {source_dir}")
    for path in sorted(source_dir.glob("*.json")):
        data = load_json(path)
        if not isinstance(data, dict) or "episode_id" not in data or "events" not in data:
            continue
        if _episode_matches_scale(data, scale):
            yield path, data


def _filter_events(events: list[dict[str, Any]], disturbance: str) -> list[dict[str, Any]]:
    if disturbance == "mixed":
        return list(events)
    if disturbance == "arrival_only":
        return [event for event in events if str(event.get("type")) == "job_arrival"]
    if disturbance == "breakdown_only":
        return [event for event in events if str(event.get("type")) == "machine_breakdown"]
    raise ValueError(f"Unknown disturbance regime: {disturbance}")


def _event_type_bucket(event_type: str, event_id: str) -> str:
    name = str(event_type).strip().lower()
    if "arrival" in name:
        return "arrival"
    if "breakdown" in name:
        return "breakdown"
    event_key = str(event_id).strip().lower()
    if event_key.startswith("arr_"):
        return "arrival"
    if event_key.startswith("bd_"):
        return "breakdown"
    return "other"


def _repair_status_from_step(*, feasible: bool, metadata: dict[str, Any]) -> str:
    if feasible:
        return "feasible"
    solver_status = str(metadata.get("solver_status", "")).strip().upper()
    if bool(metadata.get("budget_violation", False)) or solver_status in {"0", "UNKNOWN"}:
        return "timeout"
    return "infeasible"


def _copy_decision_with_budget(decision: RepairDecision, *, budget_sec: float, source: str) -> RepairDecision:
    metadata = dict(decision.metadata)
    metadata.update({"solver_time_limit_sec": float(budget_sec), "warm_start": True, "source": source})
    return RepairDecision(
        immutable_op_ids=list(decision.immutable_op_ids),
        kept_op_ids=list(decision.kept_op_ids),
        released_op_ids=list(decision.released_op_ids),
        metadata=metadata,
    )


def _gini(values: list[float]) -> float:
    clean = sorted(float(value) for value in values if float(value) >= 0.0)
    if not clean:
        return 0.0
    total = sum(clean)
    if total <= 0.0:
        return 0.0
    n = len(clean)
    weighted = sum((idx + 1) * value for idx, value in enumerate(clean))
    return float((2.0 * weighted) / (n * total) - (n + 1.0) / n)


def _structural_descriptors(env: DFJSPReschedulingEnv) -> dict[str, Any]:
    snapshot = env.state_snapshot
    if snapshot is None:
        raise RuntimeError("State snapshot is not available for structural descriptors.")
    immutable = set(snapshot.completed_op_ids) | set(snapshot.active_op_ids)
    pending_ops = [op_id for op_id in snapshot.unfinished_op_ids if op_id not in immutable]
    if pending_ops:
        flexibility = sum(len(env.instance.get_operation(op_id).eligible_machine_ids) for op_id in pending_ops) / len(pending_ops)
    else:
        flexibility = 0.0

    demand = {machine.machine_id: 0.0 for machine in env.instance.machines}
    for op_id in snapshot.window_op_ids:
        if op_id in immutable:
            continue
        for machine_id in env.instance.get_operation(op_id).eligible_machine_ids:
            demand[int(machine_id)] = demand.get(int(machine_id), 0.0) + 1.0

    downstream_lengths: list[float] = []
    for op_id in snapshot.directly_impacted_op_ids:
        operation = env.instance.get_operation(op_id)
        job = env.instance.get_job(operation.job_id)
        downstream = [
            candidate.op_global_id
            for candidate in job.operations[operation.op_index + 1 :]
            if candidate.op_global_id in snapshot.unfinished_op_ids
        ]
        downstream_lengths.append(float(len(downstream)))

    footprint = {
        "directly_impacted_ops": int(len(snapshot.directly_impacted_op_ids)),
        "affected_machines": int(len(snapshot.affected_machine_ids)),
    }
    return {
        "flexibility": float(flexibility),
        "contention": _gini(list(demand.values())),
        "propagation_depth": float(sum(downstream_lengths) / len(downstream_lengths)) if downstream_lengths else 0.0,
        "event_footprint": footprint,
        "event_footprint_ops": footprint["directly_impacted_ops"],
        "event_footprint_machines": footprint["affected_machines"],
    }


def _build_noop_row(
    *,
    method: str,
    intensity: str,
    budget_sec: float,
    scale: str,
    disturbance: str,
    episode_data: dict[str, Any],
    event_data: dict[str, Any],
    tau: float,
    objective_before,
    descriptors: dict[str, Any],
    forced_count: int,
) -> dict[str, Any]:
    return build_event_log_row(
        method=method,
        instance_id=str(episode_data["instance_id"]),
        seed=int(episode_data["seed"]),
        episode_id=str(episode_data["episode_id"]),
        event_id=str(event_data["event_id"]),
        tau=float(tau),
        budget_sec=float(budget_sec),
        window_size=0,
        forced_release_count=int(forced_count),
        motif_count=0,
        selected_motif_count=0,
        released_op_count=0,
        pred_gain_sum=None,
        inference_runtime_sec=0.0,
        selector_runtime_sec=0.0,
        solver_runtime_sec=0.0,
        makespan_after=objective_before.makespan,
        tardiness_after=objective_before.total_tardiness,
        instability_after=objective_before.instability,
        weighted_objective_after=objective_before.weighted_sum,
        changed_op_ratio=0.0,
        changed_machine_ratio=0.0,
        mean_abs_start_time_deviation=0.0,
        status="feasible",
        extra={
            "intensity_level": intensity,
            "scale": scale,
            "disturbance": disturbance,
            "event_type_bucket": _event_type_bucket(str(event_data["type"]), str(event_data["event_id"])),
            "objective_before": float(objective_before.weighted_sum),
            "reward_delta": 0.0,
            "released_op_ids": [],
            "forced_release_ops": [],
            "total_online_latency_sec": 0.0,
            **descriptors,
        },
    )


def evaluate_cell(
    *,
    cfg: dict[str, Any],
    mode: str,
    scale: ScaleSpec,
    disturbance: str,
    budget_sec: float,
    output_path: Path,
) -> tuple[int, float]:
    due_factor = float(cfg.get("data", {}).get("due_date_rule", {}).get("factor", 1.5))
    solver = CPRepairSolver(cfg["solver"])
    motif_cfg = dict(cfg.get("motifs", {}))
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for _, episode_data in _iter_episode_payloads(scale):
        selected_events = _filter_events(list(episode_data.get("events", [])), disturbance)
        if not selected_events:
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

        for fallback_id, event_data in enumerate(selected_events):
            event = deserialize_dynamic_event(event_data, fallback_event_id=fallback_id)
            env.apply_event(event)
            env.build_window()
            snapshot = env.state_snapshot
            if snapshot is None:
                raise RuntimeError("State snapshot is not available during intensity evaluation.")
            objective_before = env.compute_objective()
            descriptors = _structural_descriptors(env)
            forced_ops = forced_release_ops(snapshot)

            online_start = time.perf_counter()
            selector_start = time.perf_counter()
            release_counts: dict[str, int] = {}
            release_sets: dict[str, list[int]] = {}
            if mode in INTENSITY_DEFINITIONS:
                plan = repair_at_intensity(
                    incumbent=env.incumbent,
                    event=event,
                    budget_sec=budget_sec,
                    level=mode,
                    instance=env.instance,
                    snapshot=snapshot,
                    motif_cfg=motif_cfg,
                    instance_id=str(episode_data["instance_id"]),
                    seed=int(episode_data["seed"]),
                    episode_id=str(episode_data["episode_id"]),
                    event_id=str(event_data["event_id"]),
                )
                decision = plan.decision
                release_sets = dict(plan.released_op_ids_by_level)
                release_counts = {key: len(value) for key, value in release_sets.items()}
                intensity = plan.level
            elif mode == "heuristic_rh":
                baseline = heuristic_rh_decision(snapshot)
                decision = _copy_decision_with_budget(
                    baseline.decision,
                    budget_sec=budget_sec,
                    source="gate_baseline_heuristic_rh",
                )
                intensity = "baseline_heuristic_rh"
            elif mode == "full_reoptimization":
                baseline = full_reoptimization_decision(snapshot)
                decision = _copy_decision_with_budget(
                    baseline.decision,
                    budget_sec=budget_sec,
                    source="gate_baseline_full_reoptimization",
                )
                intensity = "baseline_full_reoptimization"
            else:
                raise ValueError(f"Unknown evaluation mode: {mode}")
            selector_runtime_sec = time.perf_counter() - selector_start

            step = env.step_reschedule(solver, decision)
            objective_after = step.objective
            total_online_latency_sec = time.perf_counter() - online_start
            status = _repair_status_from_step(feasible=bool(step.feasible), metadata=dict(step.metadata))
            event_type_bucket = _event_type_bucket(str(event_data["type"]), str(event_data["event_id"]))
            rows.append(
                build_event_log_row(
                    method=mode,
                    instance_id=str(episode_data["instance_id"]),
                    seed=int(episode_data["seed"]),
                    episode_id=str(episode_data["episode_id"]),
                    event_id=str(event_data["event_id"]),
                    tau=float(event.time),
                    budget_sec=float(budget_sec),
                    window_size=len(snapshot.window_op_ids),
                    forced_release_count=len(forced_ops),
                    motif_count=0,
                    selected_motif_count=max(0, len(decision.released_op_ids) - len(forced_ops)),
                    released_op_count=len(decision.released_op_ids),
                    pred_gain_sum=None,
                    inference_runtime_sec=0.0,
                    selector_runtime_sec=selector_runtime_sec,
                    solver_runtime_sec=float(step.runtime_sec),
                    makespan_after=objective_after.makespan,
                    tardiness_after=objective_after.total_tardiness,
                    instability_after=objective_after.instability,
                    weighted_objective_after=objective_after.weighted_sum,
                    changed_op_ratio=float(step.changed_op_ratio),
                    changed_machine_ratio=float(step.changed_machine_ratio),
                    mean_abs_start_time_deviation=compute_mean_absolute_start_time_deviation(env.incumbent),
                    status=status,
                    extra={
                        "intensity_level": intensity,
                        "scale": scale.label,
                        "disturbance": disturbance,
                        "event_type_bucket": event_type_bucket,
                        "objective_before": float(objective_before.weighted_sum),
                        "reward_delta": float(objective_before.weighted_sum - objective_after.weighted_sum),
                        "forced_release_ops": forced_ops,
                        "released_op_ids": list(decision.released_op_ids),
                        "released_op_ids_by_level": release_sets,
                        "released_op_count_by_level": release_counts,
                        "solver_time_limit_sec": float(decision.metadata.get("solver_time_limit_sec", budget_sec)),
                        "solver_status": str(step.metadata.get("solver_status", "")),
                        "solver_runtime_sec_raw": step.metadata.get("raw_wall_time_sec"),
                        "solver_wall_time_sec": step.metadata.get("solver_wall_time_sec"),
                        "solver_user_time_sec": step.metadata.get("solver_user_time_sec"),
                        "solver_runtime_budget_cap_sec": step.metadata.get("runtime_budget_cap_sec"),
                        "solver_runtime_accounting_source": step.metadata.get("runtime_accounting_source"),
                        "solver_runtime_clipped": step.metadata.get("runtime_clipped"),
                        "solver_budget_violation": step.metadata.get("budget_violation"),
                        "total_online_latency_sec": float(total_online_latency_sec),
                        **descriptors,
                    },
                )
            )

    if not rows:
        raise RuntimeError(f"No events evaluated for mode={mode}, scale={scale.label}, disturbance={disturbance}.")
    ensure_dir(output_path.parent)
    save_jsonl(rows, output_path)
    return len(rows), time.perf_counter() - started


def _row_key(row: dict[str, Any]) -> tuple[str, int, str, str]:
    return (str(row["instance_id"]), int(row["seed"]), str(row["episode_id"]), str(row["event_id"]))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _compare_gate_rows(candidate: list[dict[str, Any]], reference: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    candidate_map = {_row_key(row): row for row in candidate}
    reference_map = {_row_key(row): row for row in reference}
    failures: list[str] = []
    if set(candidate_map) != set(reference_map):
        missing = sorted(set(reference_map) - set(candidate_map))[:5]
        extra = sorted(set(candidate_map) - set(reference_map))[:5]
        failures.append(f"event-key mismatch: missing={missing}, extra={extra}")
        return False, failures
    for key in sorted(reference_map):
        cand = candidate_map[key]
        ref = reference_map[key]
        if list(cand.get("released_op_ids", [])) != list(ref.get("released_op_ids", [])):
            failures.append(f"{key}: released_op_ids differ")
        if abs(float(cand["weighted_objective_after"]) - float(ref["weighted_objective_after"])) > 1e-7:
            failures.append(f"{key}: weighted_objective_after differs")
        if abs(float(cand["reward_delta"]) - float(ref["reward_delta"])) > 1e-7:
            failures.append(f"{key}: reward_delta differs")
        if str(cand["status"]) != str(ref["status"]):
            failures.append(f"{key}: status differs")
        if len(failures) >= 10:
            break
    return not failures, failures


def _check_monotone_counts(level_rows: dict[str, list[dict[str, Any]]], sample_count: int = 20) -> tuple[bool, list[str]]:
    maps = {level: {_row_key(row): row for row in rows} for level, rows in level_rows.items()}
    common_keys = set.intersection(*(set(row_map) for row_map in maps.values()))
    failures: list[str] = []
    for key in sorted(common_keys)[:sample_count]:
        counts = [int(maps[level][key]["released_op_count"]) for level in DEFAULT_INTENSITIES]
        if counts != sorted(counts):
            failures.append(f"{key}: counts={counts}")
    return not failures, failures


def _check_monotone_counts_from_level_metadata(rows: list[dict[str, Any]], sample_count: int = 20) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for row in sorted(rows, key=_row_key)[:sample_count]:
        counts_by_level = dict(row.get("released_op_count_by_level", {}))
        counts = [int(counts_by_level.get(level, -1)) for level in DEFAULT_INTENSITIES]
        if any(count < 0 for count in counts):
            failures.append(f"{_row_key(row)}: missing released_op_count_by_level={counts_by_level}")
            continue
        if counts != sorted(counts):
            failures.append(f"{_row_key(row)}: counts={counts}")
    return not failures, failures


def _write_reproduction_gate(
    path: Path,
    *,
    passed: bool,
    l0_ok: bool,
    l3_ok: bool,
    monotone_ok: bool,
    stats: dict[str, dict[str, float]],
    failures: list[str],
    monotone_failures: list[str],
) -> None:
    lines = [
        "# Reproduction Gate",
        "",
        f"status: {'PASS' if passed else 'FAIL'}",
        "",
        "Scope: budget_sec=5.0, Brandimarte Mk8/Mk9/Mk10, frozen mixed traces.",
        "",
        "| mode | n | mean_reward_delta | feasible_rate | mean_released_op_count |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode, row in stats.items():
        lines.append(
            f"| {mode} | {int(row['n'])} | {row['mean_reward_delta']:.6f} | "
            f"{row['feasible_rate']:.6f} | {row['mean_released_op_count']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Checks:",
            f"- always-L0 release rule delegates to heuristic_rh per event: {'PASS' if l0_ok else 'FAIL'}",
            f"- always-L3 release rule delegates to full_reoptimization per event: {'PASS' if l3_ok else 'FAIL'}",
            f"- sampled 20-event monotone release counts L0 <= L1 <= L2 <= L3: {'PASS' if monotone_ok else 'FAIL'}",
        ]
    )
    if failures or monotone_failures:
        lines.append("")
        lines.append("Failure details:")
        for item in failures + monotone_failures:
            lines.append(f"- {item}")
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _audit_gate_release_sets(cfg: dict[str, Any], scale: ScaleSpec) -> tuple[bool, bool, bool, list[str], list[str], list[str]]:
    due_factor = float(cfg.get("data", {}).get("due_date_rule", {}).get("factor", 1.5))
    l0_failures: list[str] = []
    l3_failures: list[str] = []
    monotone_failures: list[str] = []
    sampled = 0
    for _, episode_data in _iter_episode_payloads(scale):
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
                raise RuntimeError("State snapshot is not available during reproduction-gate audit.")
            key = (str(episode_data["instance_id"]), int(episode_data["seed"]), str(episode_data["episode_id"]), str(event_data["event_id"]))
            l0_plan = repair_at_intensity(
                incumbent=env.incumbent,
                event=event,
                budget_sec=5.0,
                level="L0",
                instance=env.instance,
                snapshot=snapshot,
                instance_id=str(episode_data["instance_id"]),
                seed=int(episode_data["seed"]),
                episode_id=str(episode_data["episode_id"]),
                event_id=str(event_data["event_id"]),
            )
            l3_plan = repair_at_intensity(
                incumbent=env.incumbent,
                event=event,
                budget_sec=5.0,
                level="L3",
                instance=env.instance,
                snapshot=snapshot,
                instance_id=str(episode_data["instance_id"]),
                seed=int(episode_data["seed"]),
                episode_id=str(episode_data["episode_id"]),
                event_id=str(event_data["event_id"]),
            )
            heuristic_release = heuristic_rh_decision(snapshot).decision.released_op_ids
            full_release = full_reoptimization_decision(snapshot).decision.released_op_ids
            if list(l0_plan.decision.released_op_ids) != list(heuristic_release):
                l0_failures.append(f"{key}: L0 release differs from heuristic_rh")
            if list(l3_plan.decision.released_op_ids) != list(full_release):
                l3_failures.append(f"{key}: L3 release differs from full_reoptimization")
            if sampled < 20:
                counts = [len(l0_plan.released_op_ids_by_level[level]) for level in DEFAULT_INTENSITIES]
                if counts != sorted(counts):
                    monotone_failures.append(f"{key}: counts={counts}")
                sampled += 1
    return (
        not l0_failures,
        not l3_failures,
        not monotone_failures,
        l0_failures[:10],
        l3_failures[:10],
        monotone_failures[:10],
    )


def run_reproduction_gate(output_root: Path, *, skip_existing: bool = False) -> bool:
    gate_dir = ensure_dir(output_root / "reproduction_gate")
    gate_scale = ScaleSpec(
        "mk8_mk10_gate",
        "outputs/episodes/brandimarte_heldout/episodes",
        instance_ids=("mk8", "mk9", "mk10"),
    )
    budget_sec = 5.0
    cfg = _build_cfg(budget_sec)
    modes = ["L0", "L3"]
    paths: dict[str, Path] = {}
    for mode in modes:
        path = gate_dir / f"{mode}_event_metrics.jsonl"
        paths[mode] = path
        if skip_existing and path.exists():
            continue
        evaluate_cell(
            cfg=cfg,
            mode=mode,
            scale=gate_scale,
            disturbance="mixed",
            budget_sec=budget_sec,
            output_path=path,
        )

    loaded = {mode: load_jsonl(path) for mode, path in paths.items()}
    l0_ok, l3_ok, monotone_ok, l0_failures, l3_failures, monotone_failures = _audit_gate_release_sets(cfg, gate_scale)
    stats: dict[str, dict[str, float]] = {}
    for mode, rows in loaded.items():
        stats[mode] = {
            "n": float(len(rows)),
            "mean_reward_delta": _mean([float(row["reward_delta"]) for row in rows]),
            "feasible_rate": _mean([1.0 if str(row["status"]) == "feasible" else 0.0 for row in rows]),
            "mean_released_op_count": _mean([float(row["released_op_count"]) for row in rows]),
        }
    passed = bool(l0_ok and l3_ok and monotone_ok)
    failures: list[str] = []
    if not l0_ok:
        failures.extend(f"L0 vs heuristic_rh: {item}" for item in l0_failures)
    if not l3_ok:
        failures.extend(f"L3 vs full_reoptimization: {item}" for item in l3_failures)
    _write_reproduction_gate(
        output_root / "reproduction_gate.md",
        passed=passed,
        l0_ok=l0_ok,
        l3_ok=l3_ok,
        monotone_ok=monotone_ok,
        stats=stats,
        failures=failures,
        monotone_failures=monotone_failures,
    )
    return passed


def write_frontier_summary(output_root: Path, run_records: list[dict[str, Any]]) -> None:
    summary_rows: list[dict[str, Any]] = []
    for record in run_records:
        output_path = Path(record["output_path"])
        if not output_path.exists():
            continue
        rows = load_jsonl(output_path)
        if not rows:
            continue
        feasible = [1.0 if str(row.get("status")) == "feasible" else 0.0 for row in rows]
        summary_rows.append(
            {
                "intensity": record["intensity"],
                "budget_sec": float(record["budget_sec"]),
                "scale": record["scale"],
                "disturbance": record["disturbance"],
                "mean_reward_delta": _mean([float(row["reward_delta"]) for row in rows]),
                "feasible_rate": _mean(feasible),
                "mean_total_online_latency": _mean([float(row.get("total_online_latency_sec", 0.0)) for row in rows]),
                "mean_solver_runtime": _mean([float(row.get("solver_runtime_sec", 0.0)) for row in rows]),
                "n": len(rows),
            }
        )
    summary_rows.sort(key=lambda row: (str(row["scale"]), str(row["disturbance"]), float(row["budget_sec"]), str(row["intensity"])))
    path = output_root / "frontier_summary.csv"
    ensure_dir(path.parent)
    fieldnames = [
        "intensity",
        "budget_sec",
        "scale",
        "disturbance",
        "mean_reward_delta",
        "feasible_rate",
        "mean_total_online_latency",
        "mean_solver_runtime",
        "n",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def _write_manifest(
    output_root: Path,
    *,
    status: str,
    budgets: list[float],
    intensities: list[str],
    disturbances: list[str],
    scales: list[ScaleSpec],
    run_records: list[dict[str, Any]],
) -> None:
    save_json(
        {
            "status": status,
            "output_root": str(output_root),
            "intensity_ladder": INTENSITY_DEFINITIONS,
            "budgets": [float(value) for value in budgets],
            "disturbances": disturbances,
            "scales": [
                {
                    "label": scale.label,
                    "source_episodes_dir": str(_resolve_path(scale.source_episodes_dir)),
                    "instance_ids": list(scale.instance_ids),
                    "instance_prefix": scale.instance_prefix,
                }
                for scale in scales
            ],
            "seed_policy": "Frozen episode seed from each source episode; no regenerated traces.",
            "base_configs": BASE_CONFIGS,
            "runs": run_records,
        },
        output_root / "run_manifest.json",
    )


def _load_existing_run_records(output_root: Path) -> list[dict[str, Any]]:
    manifest_path = output_root / "run_manifest.json"
    if not manifest_path.exists():
        return []
    try:
        manifest = load_json(manifest_path)
    except Exception:
        return []
    runs = manifest.get("runs", [])
    if not isinstance(runs, list):
        return []
    return [dict(record) for record in runs if isinstance(record, dict) and record.get("output_path")]


def _upsert_run_record(records: list[dict[str, Any]], record: dict[str, Any]) -> None:
    key = (
        str(record.get("intensity", "")),
        float(record.get("budget_sec", 0.0)),
        str(record.get("scale", "")),
        str(record.get("disturbance", "")),
    )
    for index, existing in enumerate(records):
        existing_key = (
            str(existing.get("intensity", "")),
            float(existing.get("budget_sec", 0.0)),
            str(existing.get("scale", "")),
            str(existing.get("disturbance", "")),
        )
        if existing_key == key:
            records[index] = record
            return
    records.append(record)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fixed rescheduling-intensity frontier grid.")
    parser.add_argument("--output-root", default="outputs/intensity_grid")
    parser.add_argument("--budgets", nargs="*", type=float, default=None)
    parser.add_argument("--intensities", nargs="*", default=None)
    parser.add_argument("--disturbances", nargs="*", default=None)
    parser.add_argument("--scales", nargs="*", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-gate", action="store_true")
    parser.add_argument("--gate-only", action="store_true")
    parser.add_argument("--max-cells", type=int, default=0)
    args = parser.parse_args()

    output_root = ensure_dir(_resolve_path(args.output_root))
    set_global_seed(0)
    budgets = [float(value) for value in (args.budgets if args.budgets is not None else DEFAULT_BUDGETS)]
    intensities = _parse_filter(args.intensities, DEFAULT_INTENSITIES)
    disturbances = _parse_filter(args.disturbances, DEFAULT_DISTURBANCES)
    scale_labels = _parse_filter(args.scales, DEFAULT_SCALES)
    scale_specs = _default_scale_specs()
    scales = [scale_specs[label] for label in scale_labels]

    if not args.skip_gate:
        passed = run_reproduction_gate(output_root, skip_existing=bool(args.skip_existing))
        if not passed:
            _write_manifest(
                output_root,
                status="failed_reproduction_gate",
                budgets=budgets,
                intensities=intensities,
                disturbances=disturbances,
                scales=scales,
                run_records=[],
            )
            raise RuntimeError(f"Reproduction gate failed; see {output_root / 'reproduction_gate.md'}")
    if args.gate_only:
        _write_manifest(
            output_root,
            status="gate_only",
            budgets=budgets,
            intensities=intensities,
            disturbances=disturbances,
            scales=scales,
            run_records=[],
        )
        return

    run_records: list[dict[str, Any]] = _load_existing_run_records(output_root)
    cells_started = 0
    for scale in scales:
        scale_started = time.perf_counter()
        scale_event_rows = 0
        for disturbance in disturbances:
            for budget_sec in budgets:
                cfg = _build_cfg(budget_sec)
                for intensity in intensities:
                    if args.max_cells and cells_started >= int(args.max_cells):
                        break
                    output_path = output_root / f"{intensity}__b{_format_budget(budget_sec)}__{scale.label}__{disturbance}_event_metrics.jsonl"
                    config_path = _write_run_config(
                        output_root,
                        intensity=intensity,
                        budget_sec=budget_sec,
                        scale=scale.label,
                        disturbance=disturbance,
                        cfg=cfg,
                    )
                    command_record = {
                        "intensity": intensity,
                        "budget_sec": float(budget_sec),
                        "scale": scale.label,
                        "disturbance": disturbance,
                        "output_path": str(output_path),
                        "config_path": str(config_path),
                        "source_episodes_dir": str(_resolve_path(scale.source_episodes_dir)),
                    }
                    if args.skip_existing and output_path.exists():
                        row_count = len(load_jsonl(output_path))
                        elapsed_sec = 0.0
                    else:
                        row_count, elapsed_sec = evaluate_cell(
                            cfg=cfg,
                            mode=intensity,
                            scale=scale,
                            disturbance=disturbance,
                            budget_sec=budget_sec,
                            output_path=output_path,
                        )
                    scale_event_rows += int(row_count)
                    record = {
                        **command_record,
                        "event_count": int(row_count),
                        "elapsed_sec": float(elapsed_sec),
                    }
                    _upsert_run_record(run_records, record)
                    cells_started += 1
                    _write_manifest(
                        output_root,
                        status="partial",
                        budgets=budgets,
                        intensities=intensities,
                        disturbances=disturbances,
                        scales=scales,
                        run_records=run_records,
                    )
                    write_frontier_summary(output_root, run_records)
                if args.max_cells and cells_started >= int(args.max_cells):
                    break
            if args.max_cells and cells_started >= int(args.max_cells):
                break
        scale_elapsed = time.perf_counter() - scale_started
        print(
            f"[scale-complete] scale={scale.label} event_rows={scale_event_rows} elapsed_sec={scale_elapsed:.2f}",
            flush=True,
        )
        if args.max_cells and cells_started >= int(args.max_cells):
            break

    write_frontier_summary(output_root, run_records)
    _write_manifest(
        output_root,
        status="completed" if not args.max_cells or cells_started < int(args.max_cells) else "partial_max_cells",
        budgets=budgets,
        intensities=intensities,
        disturbances=disturbances,
        scales=scales,
        run_records=run_records,
    )


if __name__ == "__main__":
    main()
