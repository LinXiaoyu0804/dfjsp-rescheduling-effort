from __future__ import annotations

import argparse
import csv
import hashlib
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from _bootstrap import REPO_ROOT  # noqa: F401

from src.data.unified_parser import parse_instance
from src.env.dfjsp_env import DFJSPReschedulingEnv
from src.eval.event_logging import build_event_log_row
from src.eval.metrics import compute_mean_absolute_start_time_deviation
from src.eval.rho_boundary import INTENSITY_ORDER, build_event_outcomes, evaluate_honest_policy, write_rho_boundary_outputs
from src.events.generator import RHO_INTENSITY_PROFILE_DEFAULTS, generate_dynamic_events
from src.events.serialization import deserialize_dynamic_event, serialize_dynamic_event
from src.scheduling.incumbent_builder import load_incumbent_schedule
from src.scheduling.intensity_ladder import INTENSITY_DEFINITIONS, forced_release_ops, repair_at_intensity
from src.scheduling.rho import compute_rho_descriptors
from src.solver.cp_repair_solver import CPRepairSolver
from src.utils.config import load_merged_config
from src.utils.io import ensure_dir, load_json, load_jsonl, save_json, save_jsonl
from src.utils.seed import set_global_seed


DEFAULT_BUDGETS = [0.5, 1.0]
DEFAULT_REGIMES = ["R0", "R1", "R2", "R3", "R4", "R5"]
DEFAULT_SCALES = ["mk9", "mk10", "synthetic_50x15", "synthetic_100x20"]
BASE_CONFIGS = [
    "configs/default.yaml",
    "configs/env/formal_dynamic_stronger_v2.yaml",
    "configs/env/rho_boundary_profiles.yaml",
    "configs/solver/cp_repair_default.yaml",
]


@dataclass(frozen=True)
class ScaleSpec:
    label: str
    source_episodes_dir: str
    instance_ids: tuple[str, ...] = ()
    instance_prefix: str = ""
    seeds: tuple[int, ...] = ()


def _resolve_path(path_like: str | Path) -> Path:
    # Normalize Windows-style separators so episode paths stay cross-platform.
    path = Path(str(path_like).replace("\\", "/"))
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def _to_repo_relative(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _format_budget(budget_sec: float) -> str:
    return f"{float(budget_sec):g}".replace(".", "p")


def _default_scale_specs() -> dict[str, ScaleSpec]:
    brandimarte = "outputs/episodes/brandimarte_heldout/episodes"
    synthetic_root = "outputs/episodes"
    return {
        "mk9": ScaleSpec("mk9", brandimarte, instance_ids=("mk9",), seeds=tuple(range(10))),
        "mk10": ScaleSpec("mk10", brandimarte, instance_ids=("mk10",), seeds=tuple(range(10))),
        "synthetic_50x15": ScaleSpec(
            "synthetic_50x15",
            f"{synthetic_root}/synthetic_50x15/episodes",
            instance_prefix="syn_50x15_",
            seeds=tuple(range(5)),
        ),
        "synthetic_100x20": ScaleSpec(
            "synthetic_100x20",
            f"{synthetic_root}/synthetic_100x20/episodes",
            instance_prefix="syn_100x20_",
            seeds=tuple(range(5)),
        ),
    }


def _parse_filter(values: list[str] | None, default: list[str]) -> list[str]:
    if not values:
        return list(default)
    parsed: list[str] = []
    for value in values:
        parsed.extend(part.strip() for part in str(value).split(",") if part.strip())
    return parsed


def _build_cfg(*, budget_sec: float, regime: str | None = None) -> dict[str, Any]:
    cfg = load_merged_config(*(_resolve_path(path) for path in BASE_CONFIGS))
    cfg.setdefault("experiment", {})["seed"] = 0
    if regime is not None:
        cfg.setdefault("events", {}).setdefault("rho_intensity_profile", {})["label"] = regime
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
    regime: str,
    intensity: str,
    budget_sec: float,
    cfg: dict[str, Any],
) -> Path:
    path = output_root / "_configs" / f"{regime}__{intensity}__b{_format_budget(budget_sec)}.json"
    save_json(
        {
            "base_configs": BASE_CONFIGS,
            "regime": regime,
            "intensity": intensity,
            "budget_sec": float(budget_sec),
            "effective_config": cfg,
        },
        path,
    )
    return path


def _episode_matches_scale(episode_data: dict[str, Any], scale: ScaleSpec) -> bool:
    instance_id = str(episode_data.get("instance_id", ""))
    if scale.instance_ids and instance_id not in set(scale.instance_ids):
        return False
    if scale.instance_prefix and not instance_id.startswith(scale.instance_prefix):
        return False
    if scale.seeds and int(episode_data.get("seed", -1)) not in set(scale.seeds):
        return False
    return True


def _iter_source_episode_payloads(scale: ScaleSpec) -> Iterable[tuple[Path, dict[str, Any]]]:
    source_dir = _resolve_path(scale.source_episodes_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"Frozen episode directory does not exist: {source_dir}")
    for path in sorted(source_dir.glob("*.json")):
        if path.name == "episode_manifest.json":
            continue
        data = load_json(path)
        if not isinstance(data, dict) or "episode_id" not in data or "events" not in data:
            continue
        if _episode_matches_scale(data, scale):
            yield path, data


def _event_prefix(event_type: str) -> str:
    return {
        "job_arrival": "arr",
        "machine_breakdown": "bd",
        "processing_time_perturbation": "pt",
        "compound": "cmp",
    }[event_type]


def _serialize_generated_events(events: list[Any]) -> list[dict[str, Any]]:
    counters: dict[str, int] = {}
    serialized: list[dict[str, Any]] = []
    for event in events:
        prefix = _event_prefix(event.event_type)
        idx = counters.get(event.event_type, 0)
        counters[event.event_type] = idx + 1
        serialized.append(serialize_dynamic_event(event, event_id=f"{prefix}_{idx}"))
    return serialized


def generate_profile_episodes(
    *,
    regime: str,
    scales: list[ScaleSpec],
    output_root: Path,
    cfg: dict[str, Any],
    skip_existing: bool,
) -> Path:
    if regime == "R0":
        raise ValueError("R0 uses frozen standard source episodes and is not regenerated.")
    episode_dir = ensure_dir(output_root / "episodes" / regime)
    manifest_rows: list[dict[str, Any]] = []
    due_factor = float(cfg.get("data", {}).get("due_date_rule", {}).get("factor", 1.5))
    for scale in scales:
        for _, episode_data in _iter_source_episode_payloads(scale):
            episode_id = f"{episode_data['episode_id']}_{regime}"
            output_path = episode_dir / f"{episode_id}.json"
            if output_path.exists() and skip_existing:
                generated = load_json(output_path)
                events = generated.get("events", [])
            else:
                instance_path = _resolve_path(episode_data["instance_path"])
                incumbent_path = _resolve_path(episode_data["incumbent_ref"])
                if not instance_path.exists():
                    raise FileNotFoundError(f"Instance path does not exist: {instance_path}")
                if not incumbent_path.exists():
                    raise FileNotFoundError(f"Incumbent path does not exist: {incumbent_path}")
                instance = parse_instance(
                    instance_path,
                    family=cfg.get("data", {}).get("family", "fjsp"),
                    due_date_factor=due_factor,
                )
                incumbent = load_incumbent_schedule(instance, load_json(incumbent_path))
                events = _serialize_generated_events(
                    generate_dynamic_events(
                        instance,
                        cfg.get("events", {}),
                        seed=int(episode_data["seed"]),
                        incumbent=incumbent,
                    )
                )
                if not events:
                    raise RuntimeError(f"Profile {regime} generated no events for {episode_data['episode_id']}.")
                save_json(
                    {
                        "episode_id": episode_id,
                        "source_episode_id": str(episode_data["episode_id"]),
                        "regime": regime,
                        "instance_id": str(episode_data["instance_id"]),
                        "instance_path": str(episode_data["instance_path"]),
                        "seed": int(episode_data["seed"]),
                        "incumbent_ref": str(episode_data["incumbent_ref"]),
                        "events": events,
                    },
                    output_path,
                )
            subevents = [
                subevent
                for event in events
                for subevent in event.get("payload", {}).get("subevents", [])
            ]
            manifest_rows.append(
                {
                    "episode_id": episode_id,
                    "source_episode_id": str(episode_data["episode_id"]),
                    "regime": regime,
                    "scale": scale.label,
                    "instance_id": str(episode_data["instance_id"]),
                    "seed": int(episode_data["seed"]),
                    "episode_path": _to_repo_relative(output_path),
                    "num_events": int(len(events)),
                    "num_atomic_subevents": int(len(subevents)),
                    "num_arrivals": int(sum(1 for item in subevents if item.get("type") == "job_arrival")),
                    "num_breakdowns": int(sum(1 for item in subevents if item.get("type") == "machine_breakdown")),
                    "first_event_time": min((float(event["time"]) for event in events), default=None),
                    "last_event_time": max((float(event["time"]) for event in events), default=None),
                }
            )
    if not manifest_rows:
        raise RuntimeError(f"No episodes generated for regime={regime}.")
    save_json({"episodes": manifest_rows}, episode_dir / "episode_manifest.json")
    return episode_dir


def _iter_regime_episode_payloads(
    *,
    regime: str,
    scale: ScaleSpec,
    output_root: Path,
) -> Iterable[tuple[Path, dict[str, Any]]]:
    if regime == "R0":
        yield from _iter_source_episode_payloads(scale)
        return
    episode_dir = output_root / "episodes" / regime
    if not episode_dir.exists():
        raise FileNotFoundError(f"Generated episode directory does not exist: {episode_dir}")
    for path in sorted(episode_dir.glob("*.json")):
        if path.name == "episode_manifest.json":
            continue
        data = load_json(path)
        if _episode_matches_scale(data, scale):
            yield path, data


def _event_type_bucket(event_data: dict[str, Any]) -> str:
    event_type = str(event_data.get("type", event_data.get("event_type", ""))).lower()
    if event_type == "compound":
        subtypes = {
            str(subevent.get("type", subevent.get("event_type", ""))).lower()
            for subevent in event_data.get("payload", {}).get("subevents", [])
        }
        if subtypes == {"job_arrival"}:
            return "arrival"
        if subtypes == {"machine_breakdown"}:
            return "breakdown"
        return "compound"
    if "arrival" in event_type:
        return "arrival"
    if "breakdown" in event_type:
        return "breakdown"
    return "other"


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
    flexibility = (
        sum(len(env.instance.get_operation(op_id).eligible_machine_ids) for op_id in pending_ops) / len(pending_ops)
        if pending_ops
        else 0.0
    )
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


def _repair_status_from_step(*, feasible: bool, metadata: dict[str, Any]) -> str:
    if feasible:
        return "feasible"
    solver_status = str(metadata.get("solver_status", "")).strip().upper()
    if bool(metadata.get("budget_violation", False)) or solver_status in {"0", "UNKNOWN"}:
        return "timeout"
    return "infeasible"


def _build_noop_row(
    *,
    regime: str,
    intensity: str,
    budget_sec: float,
    scale: str,
    episode_data: dict[str, Any],
    event_data: dict[str, Any],
    event_time: float,
    objective_before: Any,
    descriptors: dict[str, Any],
    rho: dict[str, Any],
) -> dict[str, Any]:
    return build_event_log_row(
        method=intensity,
        instance_id=str(episode_data["instance_id"]),
        seed=int(episode_data["seed"]),
        episode_id=str(episode_data["episode_id"]),
        event_id=str(event_data["event_id"]),
        tau=float(event_time),
        budget_sec=float(budget_sec),
        window_size=0,
        forced_release_count=0,
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
            "regime": regime,
            "intensity_level": intensity,
            "scale": scale,
            "disturbance": "mixed",
            "event_type_bucket": _event_type_bucket(event_data),
            "makespan_before": float(objective_before.makespan),
            "tardiness_before": float(objective_before.total_tardiness),
            "instability_before": float(objective_before.instability),
            "objective_before": float(objective_before.weighted_sum),
            "reward_delta": 0.0,
            "released_op_ids": [],
            "forced_release_ops": [],
            "total_online_latency_sec": 0.0,
            "size": int(episode_data.get("size", 0)),
            **descriptors,
            **rho,
        },
    )


def evaluate_cell(
    *,
    cfg: dict[str, Any],
    regime: str,
    intensity: str,
    scales: list[ScaleSpec],
    budget_sec: float,
    output_path: Path,
) -> tuple[int, float]:
    due_factor = float(cfg.get("data", {}).get("due_date_rule", {}).get("factor", 1.5))
    solver = CPRepairSolver(cfg["solver"])
    motif_cfg = dict(cfg.get("motifs", {}))
    motif_cfg.setdefault("families", ["M3", "M4"])
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for scale in scales:
        for _, episode_data in _iter_regime_episode_payloads(regime=regime, scale=scale, output_root=output_path.parents[0]):
            events_data = list(episode_data.get("events", []))
            if not events_data:
                continue
            instance_path = _resolve_path(episode_data["instance_path"])
            incumbent_path = _resolve_path(episode_data["incumbent_ref"])
            if not instance_path.exists():
                raise FileNotFoundError(f"Instance path does not exist: {instance_path}")
            if not incumbent_path.exists():
                raise FileNotFoundError(f"Incumbent path does not exist: {incumbent_path}")
            instance = parse_instance(
                instance_path,
                family=cfg.get("data", {}).get("family", "fjsp"),
                due_date_factor=due_factor,
            )
            env = DFJSPReschedulingEnv(instance, cfg)
            env.reset()
            env.incumbent = load_incumbent_schedule(instance, load_json(incumbent_path))
            env.initial_instance = deepcopy(instance)
            env.instance = deepcopy(instance)

            for fallback_id, event_data in enumerate(events_data):
                event = deserialize_dynamic_event(event_data, fallback_event_id=fallback_id)
                env.apply_event(event)
                env.build_window()
                snapshot = env.state_snapshot
                if snapshot is None:
                    raise RuntimeError("State snapshot is not available during rho-boundary evaluation.")
                objective_before = env.compute_objective()
                descriptors = _structural_descriptors(env)
                rho = compute_rho_descriptors(
                    instance=env.instance,
                    incumbent=env.incumbent,
                    snapshot=snapshot,
                    makespan_before=float(objective_before.makespan),
                )
                forced_ops = forced_release_ops(snapshot)
                if not snapshot.window_op_ids:
                    row = _build_noop_row(
                        regime=regime,
                        intensity=intensity,
                        budget_sec=budget_sec,
                        scale=scale.label,
                        episode_data={**episode_data, "size": env.instance.num_operations},
                        event_data=event_data,
                        event_time=event.time,
                        objective_before=objective_before,
                        descriptors=descriptors,
                        rho=rho,
                    )
                    rows.append(row)
                    continue

                online_start = time.perf_counter()
                selector_start = time.perf_counter()
                plan = repair_at_intensity(
                    incumbent=env.incumbent,
                    event=event,
                    budget_sec=budget_sec,
                    level=intensity,
                    instance=env.instance,
                    snapshot=snapshot,
                    motif_cfg=motif_cfg,
                    instance_id=str(episode_data["instance_id"]),
                    seed=int(episode_data["seed"]),
                    episode_id=str(episode_data["episode_id"]),
                    event_id=str(event_data["event_id"]),
                    compute_all_release_sets=False,
                )
                decision = plan.decision
                selector_runtime_sec = time.perf_counter() - selector_start
                step = env.step_reschedule(solver, decision)
                objective_after = step.objective
                total_online_latency_sec = time.perf_counter() - online_start
                status = _repair_status_from_step(feasible=bool(step.feasible), metadata=dict(step.metadata))
                release_sets = dict(plan.released_op_ids_by_level)
                release_counts = {key: len(value) for key, value in release_sets.items()}
                rows.append(
                    build_event_log_row(
                        method=intensity,
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
                            "regime": regime,
                            "intensity_level": plan.level,
                            "scale": scale.label,
                            "disturbance": "mixed",
                            "event_type_bucket": _event_type_bucket(event_data),
                            "makespan_before": float(objective_before.makespan),
                            "tardiness_before": float(objective_before.total_tardiness),
                            "instability_before": float(objective_before.instability),
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
                            "size": int(env.instance.num_operations),
                            **descriptors,
                            **rho,
                        },
                    )
                )
    if not rows:
        raise RuntimeError(f"No events evaluated for regime={regime}, intensity={intensity}, budget={budget_sec}.")
    ensure_dir(output_path.parent)
    save_jsonl(rows, output_path)
    return len(rows), time.perf_counter() - started


def _build_replay_rho_lookup(*, cfg: dict[str, Any], scales: list[ScaleSpec]) -> dict[tuple[str, str, int, str, str], dict[str, Any]]:
    due_factor = float(cfg.get("data", {}).get("due_date_rule", {}).get("factor", 1.5))
    lookup: dict[tuple[str, str, int, str, str], dict[str, Any]] = {}
    for scale in scales:
        for _, episode_data in _iter_source_episode_payloads(scale):
            instance = parse_instance(
                _resolve_path(episode_data["instance_path"]),
                family=cfg.get("data", {}).get("family", "fjsp"),
                due_date_factor=due_factor,
            )
            env = DFJSPReschedulingEnv(instance, cfg)
            env.reset()
            env.incumbent = load_incumbent_schedule(instance, load_json(_resolve_path(episode_data["incumbent_ref"])))
            env.initial_instance = deepcopy(instance)
            env.instance = deepcopy(instance)
            for fallback_id, event_data in enumerate(episode_data.get("events", [])):
                event = deserialize_dynamic_event(event_data, fallback_event_id=fallback_id)
                env.apply_event(event)
                env.build_window()
                snapshot = env.state_snapshot
                if snapshot is None:
                    raise RuntimeError("State snapshot is missing while replaying R0 rho lookup.")
                objective_before = env.compute_objective()
                rho = compute_rho_descriptors(
                    instance=env.instance,
                    incumbent=env.incumbent,
                    snapshot=snapshot,
                    makespan_before=float(objective_before.makespan),
                )
                lookup[
                    (
                        scale.label,
                        str(episode_data["instance_id"]),
                        int(episode_data["seed"]),
                        str(episode_data["episode_id"]),
                        str(event_data["event_id"]),
                    )
                ] = rho
    return lookup


def materialize_r0_from_intensity_decomp(
    *,
    source_root: Path,
    output_root: Path,
    cfg: dict[str, Any],
    scales: list[ScaleSpec],
    budgets: list[float],
    intensities: list[str],
) -> list[dict[str, Any]]:
    if not source_root.exists():
        raise FileNotFoundError(f"R0 source decomposition root does not exist: {source_root}")
    rho_lookup = _build_replay_rho_lookup(cfg=cfg, scales=scales)
    records: list[dict[str, Any]] = []
    for budget_sec in budgets:
        budget_label = _format_budget(budget_sec)
        for intensity in intensities:
            rows: list[dict[str, Any]] = []
            for scale in scales:
                source_path = source_root / f"{intensity}__b{budget_label}__{scale.label}__mixed_event_metrics.jsonl"
                if not source_path.exists():
                    raise FileNotFoundError(f"Missing R0 source event metrics: {source_path}")
                for row in load_jsonl(source_path):
                    key = (
                        scale.label,
                        str(row["instance_id"]),
                        int(row["seed"]),
                        str(row["episode_id"]),
                        str(row["event_id"]),
                    )
                    rho = rho_lookup.get(key)
                    if rho is None:
                        raise KeyError(f"Missing replayed rho_t for R0 event key: {key}")
                    out = dict(row)
                    out["regime"] = "R0"
                    out["scale"] = scale.label
                    out["disturbance"] = "mixed"
                    out.update(rho)
                    rows.append(out)
            output_path = output_root / f"R0__{intensity}__b{budget_label}_event_metrics.jsonl"
            save_jsonl(rows, output_path)
            records.append(
                {
                    "regime": "R0",
                    "intensity": intensity,
                    "budget_sec": float(budget_sec),
                    "event_count": int(len(rows)),
                    "elapsed_sec": 0.0,
                    "output_path": str(output_path),
                    "config_path": None,
                    "source": str(source_root.resolve()),
                    "source_policy": "materialized_from_existing_intensity_grid_decomp_with_replayed_rho_t",
                }
            )
    return records


def _all_event_metric_rows(output_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(output_root.glob("*_event_metrics.jsonl")):
        rows.extend(load_jsonl(path))
    if not rows:
        raise FileNotFoundError(f"No event metrics found under {output_root}")
    return rows


def _rho_distribution(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    by_regime: dict[str, list[float]] = {}
    seen: set[tuple[str, str, str, int, str, str]] = set()
    for row in rows:
        key = (
            str(row.get("regime")),
            str(row.get("scale")),
            str(row.get("instance_id")),
            int(row.get("seed")),
            str(row.get("episode_id")),
            str(row.get("event_id")),
        )
        if key in seen:
            continue
        seen.add(key)
        value = float(row.get("rho_t", 0.0))
        by_regime.setdefault(str(row.get("regime")), []).append(value)
    stats: dict[str, dict[str, float]] = {}
    for regime, values in sorted(by_regime.items()):
        ordered = sorted(values)
        if not ordered:
            continue
        stats[regime] = {
            "n_events": float(len(ordered)),
            "min": float(ordered[0]),
            "median": float(ordered[len(ordered) // 2]),
            "max": float(ordered[-1]),
        }
    return stats


def _validate_high_rho(stats: dict[str, dict[str, float]]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    r0 = stats.get("R0")
    if not r0:
        return False, ["Missing R0 rho distribution."]
    r0_max = float(r0["max"])
    r0_median = float(r0["median"])
    for regime in ["R3", "R4", "R5"]:
        row = stats.get(regime)
        if not row:
            failures.append(f"Missing {regime} rho distribution.")
            continue
        if float(row["median"]) <= r0_median:
            failures.append(f"{regime} median rho_t={row['median']:.6g} does not exceed R0 median={r0_median:.6g}.")
        if float(row["max"]) <= r0_max:
            failures.append(f"{regime} max rho_t={row['max']:.6g} does not exceed R0 max={r0_max:.6g}.")
    return not failures, failures


def _reproduction_gate(output_root: Path, *, bootstrap_reps: int) -> tuple[bool, dict[str, Any], str]:
    rows = [row for row in _all_event_metric_rows(output_root) if str(row.get("regime")) == "R0"]
    if not rows:
        raise RuntimeError("R0 rows are missing; cannot run reproduction gate.")
    outcomes = build_event_outcomes(rows, gamma=0.2)
    policy_summary, _ = evaluate_honest_policy(outcomes, train_seed_parity=0)
    mean_headroom_percent = float(outcomes["oracle_headroom"].mean() * 100.0)
    frac_positive = float((outcomes["oracle_headroom"] > 0.0).mean())
    max_single_level_abs_gain_percent = max(
        abs(float(outcomes[f"gain_{level}"].mean() * 100.0)) for level in INTENSITY_ORDER[1:]
    )
    test_policy_gain_percent = float(policy_summary["test_mean_policy_gain"]) * 100.0
    passed = (
        0.015 <= mean_headroom_percent <= 0.06
        and abs(float(policy_summary["test_capture_fraction"])) <= 0.50
        and abs(test_policy_gain_percent) <= 0.01
        and max_single_level_abs_gain_percent <= 0.02
    )
    lines = [
        "# R0 Reproduction Gate",
        "",
        f"status: {'PASS' if passed else 'FAIL'}",
        "",
        "Scope: R0 standard mixed traces, selected rho-boundary instances, budgets 0.5s and 1.0s.",
        "",
        "| metric | value | gate |",
        "|---|---:|---|",
        f"| mean oracle headroom | {mean_headroom_percent:.6f}% | target around 0.03% |",
        f"| test honest capture | {100.0 * float(policy_summary['test_capture_fraction']):.6f}% | low-to-small, abs <= 50% |",
        f"| test mean policy gain | {test_policy_gain_percent:.8f}% | absolute gain <= 0.01% |",
        f"| max mean single non-L0 absolute gain | {max_single_level_abs_gain_percent:.8f}% | <= 0.02% |",
        f"| frac oracle headroom > 0 | {frac_positive:.6f} | diagnostic |",
        f"| n events after budget/intensity matching | {len(outcomes)} | diagnostic |",
        "",
        "Honest policy details:",
        f"- threshold: `{float(policy_summary['threshold']):.12g}`",
        f"- train events: `{int(policy_summary['n_train_events'])}`",
        f"- test events: `{int(policy_summary['n_test_events'])}`",
        f"- test upgrade rate: `{float(policy_summary['test_upgrade_rate']):.6f}`",
    ]
    path = output_root / "reproduction_gate.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return passed, policy_summary, "\n".join(lines)


def _write_frontier_summary(output_root: Path, run_records: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for record in run_records:
        path = Path(record["output_path"])
        if not path.exists():
            continue
        data = load_jsonl(path)
        if not data:
            continue
        rows.append(
            {
                "regime": record["regime"],
                "intensity": record["intensity"],
                "budget_sec": float(record["budget_sec"]),
                "event_count": len(data),
                "mean_rho_t": sum(float(row["rho_t"]) for row in data) / len(data),
                "mean_reward_delta": sum(float(row["reward_delta"]) for row in data) / len(data),
                "feasible_rate": sum(1.0 if str(row["status"]) == "feasible" else 0.0 for row in data) / len(data),
                "mean_solver_runtime": sum(float(row.get("solver_runtime_sec", 0.0)) for row in data) / len(data),
                "mean_total_online_latency": sum(float(row.get("total_online_latency_sec", 0.0)) for row in data) / len(data),
            }
        )
    path = output_root / "frontier_summary.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "regime",
            "intensity",
            "budget_sec",
            "event_count",
            "mean_rho_t",
            "mean_reward_delta",
            "feasible_rate",
            "mean_solver_runtime",
            "mean_total_online_latency",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["regime"], row["budget_sec"], row["intensity"])))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(
    output_root: Path,
    *,
    status: str,
    regimes: list[str],
    budgets: list[float],
    intensities: list[str],
    scales: list[ScaleSpec],
    run_records: list[dict[str, Any]],
    rho_distributions: dict[str, dict[str, float]] | None = None,
    gate: dict[str, Any] | None = None,
) -> None:
    config_paths = [_resolve_path(path) for path in BASE_CONFIGS]
    save_json(
        {
            "status": status,
            "output_root": str(output_root.resolve()),
            "base_configs": [str(path) for path in config_paths],
            "base_config_sha256": {str(path): _sha256(path) for path in config_paths},
            "checkpoint_path": None,
            "checkpoint_sha256": None,
            "intensity_ladder": INTENSITY_DEFINITIONS,
            "rho_intensity_profile_definitions": {"R0": "standard frozen traces", **RHO_INTENSITY_PROFILE_DEFAULTS},
            "regimes": regimes,
            "budgets": [float(value) for value in budgets],
            "intensities": intensities,
            "disturbance": "mixed",
            "seeds": {
                "mk9_mk10": list(range(10)),
                "synthetic_50x15_100x20": list(range(5)),
            },
            "instances": [
                {
                    "label": scale.label,
                    "source_episodes_dir": str(_resolve_path(scale.source_episodes_dir)),
                    "instance_ids": list(scale.instance_ids),
                    "instance_prefix": scale.instance_prefix,
                    "seeds": list(scale.seeds),
                }
                for scale in scales
            ],
            "event_trace_policy": "Each (regime, scale, instance, seed) episode trace is frozen and reused across all intensities and budgets.",
            "solver_budget_policy": "All intensity levels receive the same per-event CP-SAT wall-clock cap for a given budget cell.",
            "rho_definition": "rho_t = pending assigned processing mass in W_t / makespan_before; rho_t_foot = directly impacted pending assigned processing mass / makespan_before.",
            "rho_distributions": rho_distributions or {},
            "reproduction_gate": gate or {},
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
    return [dict(row) for row in manifest.get("runs", []) if isinstance(row, dict)]


def _upsert_run_record(records: list[dict[str, Any]], record: dict[str, Any]) -> None:
    key = (record["regime"], record["intensity"], float(record["budget_sec"]))
    for idx, existing in enumerate(records):
        if (existing.get("regime"), existing.get("intensity"), float(existing.get("budget_sec", 0.0))) == key:
            records[idx] = record
            return
    records.append(record)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rho_t boundary intensity-value experiment.")
    parser.add_argument("--output-root", default="outputs/rho_boundary")
    parser.add_argument("--regimes", nargs="*", default=None)
    parser.add_argument("--budgets", nargs="*", type=float, default=None)
    parser.add_argument("--intensities", nargs="*", default=None)
    parser.add_argument("--scales", nargs="*", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--gate-only", action="store_true")
    parser.add_argument("--skip-gate", action="store_true")
    parser.add_argument("--max-cells", type=int, default=0)
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--import-r0-from-decomp", default=None)
    args = parser.parse_args()

    output_root = ensure_dir(_resolve_path(args.output_root))
    set_global_seed(0)
    regimes = _parse_filter(args.regimes, DEFAULT_REGIMES)
    budgets = [float(value) for value in (args.budgets if args.budgets is not None else DEFAULT_BUDGETS)]
    intensities = _parse_filter(args.intensities, INTENSITY_ORDER)
    scale_labels = _parse_filter(args.scales, DEFAULT_SCALES)
    scale_specs = _default_scale_specs()
    scales = [scale_specs[label] for label in scale_labels]

    run_records = _load_existing_run_records(output_root)
    cells_started = 0
    gate_payload: dict[str, Any] = {}
    imported_r0 = False
    if args.import_r0_from_decomp and "R0" in regimes:
        r0_cfg = _build_cfg(budget_sec=budgets[0], regime="R0")
        imported_records = materialize_r0_from_intensity_decomp(
            source_root=_resolve_path(args.import_r0_from_decomp),
            output_root=output_root,
            cfg=r0_cfg,
            scales=scales,
            budgets=budgets,
            intensities=intensities,
        )
        for record in imported_records:
            _upsert_run_record(run_records, record)
        imported_r0 = True
        _write_frontier_summary(output_root, run_records)

    for regime in regimes:
        if regime == "R0" and imported_r0:
            if not args.skip_gate:
                passed, gate_summary, _ = _reproduction_gate(output_root, bootstrap_reps=int(args.bootstrap_reps))
                gate_payload = {"passed": bool(passed), **gate_summary}
                rho_stats = _rho_distribution(_all_event_metric_rows(output_root))
                _write_manifest(
                    output_root,
                    status="failed_reproduction_gate" if not passed else "passed_reproduction_gate",
                    regimes=regimes,
                    budgets=budgets,
                    intensities=intensities,
                    scales=scales,
                    run_records=run_records,
                    rho_distributions=rho_stats,
                    gate=gate_payload,
                )
                if not passed:
                    raise RuntimeError(f"R0 reproduction gate failed; see {output_root / 'reproduction_gate.md'}")
                if args.gate_only:
                    return
            continue
        if regime != "R0":
            cfg = _build_cfg(budget_sec=budgets[0], regime=regime)
            generate_profile_episodes(
                regime=regime,
                scales=scales,
                output_root=output_root,
                cfg=cfg,
                skip_existing=bool(args.skip_existing),
            )
        for budget_sec in budgets:
            cfg = _build_cfg(budget_sec=budget_sec, regime=regime)
            for intensity in intensities:
                if args.max_cells and cells_started >= int(args.max_cells):
                    break
                output_path = output_root / f"{regime}__{intensity}__b{_format_budget(budget_sec)}_event_metrics.jsonl"
                config_path = _write_run_config(
                    output_root,
                    regime=regime,
                    intensity=intensity,
                    budget_sec=budget_sec,
                    cfg=cfg,
                )
                if args.skip_existing and output_path.exists():
                    row_count = len(load_jsonl(output_path))
                    elapsed_sec = 0.0
                else:
                    row_count, elapsed_sec = evaluate_cell(
                        cfg=cfg,
                        regime=regime,
                        intensity=intensity,
                        scales=scales,
                        budget_sec=budget_sec,
                        output_path=output_path,
                    )
                record = {
                    "regime": regime,
                    "intensity": intensity,
                    "budget_sec": float(budget_sec),
                    "event_count": int(row_count),
                    "elapsed_sec": float(elapsed_sec),
                    "output_path": str(output_path),
                    "config_path": str(config_path),
                }
                _upsert_run_record(run_records, record)
                cells_started += 1
                rows_so_far = _all_event_metric_rows(output_root)
                rho_stats = _rho_distribution(rows_so_far)
                _write_frontier_summary(output_root, run_records)
                _write_manifest(
                    output_root,
                    status="partial",
                    regimes=regimes,
                    budgets=budgets,
                    intensities=intensities,
                    scales=scales,
                    run_records=run_records,
                    rho_distributions=rho_stats,
                    gate=gate_payload,
                )
            if args.max_cells and cells_started >= int(args.max_cells):
                break
        if regime == "R0" and not args.skip_gate:
            passed, gate_summary, _ = _reproduction_gate(output_root, bootstrap_reps=int(args.bootstrap_reps))
            gate_payload = {"passed": bool(passed), **gate_summary}
            rho_stats = _rho_distribution(_all_event_metric_rows(output_root))
            _write_manifest(
                output_root,
                status="failed_reproduction_gate" if not passed else "passed_reproduction_gate",
                regimes=regimes,
                budgets=budgets,
                intensities=intensities,
                scales=scales,
                run_records=run_records,
                rho_distributions=rho_stats,
                gate=gate_payload,
            )
            if not passed:
                raise RuntimeError(f"R0 reproduction gate failed; see {output_root / 'reproduction_gate.md'}")
            if args.gate_only:
                return
        if args.max_cells and cells_started >= int(args.max_cells):
            break

    all_rows = _all_event_metric_rows(output_root)
    rho_stats = _rho_distribution(all_rows)
    high_rho_ok, high_rho_failures = _validate_high_rho(rho_stats) if all(regime in rho_stats for regime in ["R0", "R3", "R4", "R5"]) else (True, [])
    if not high_rho_ok:
        _write_manifest(
            output_root,
            status="failed_high_rho_validation",
            regimes=regimes,
            budgets=budgets,
            intensities=intensities,
            scales=scales,
            run_records=run_records,
            rho_distributions=rho_stats,
            gate={**gate_payload, "high_rho_failures": high_rho_failures},
        )
        raise RuntimeError("High-rho validation failed: " + "; ".join(high_rho_failures))
    if not args.max_cells or cells_started < int(args.max_cells):
        write_rho_boundary_outputs(
            records=all_rows,
            output_root=output_root,
            gamma=0.2,
            bootstrap_reps=int(args.bootstrap_reps),
            train_seed_parity=0,
        )
    _write_frontier_summary(output_root, run_records)
    _write_manifest(
        output_root,
        status="completed" if not args.max_cells or cells_started < int(args.max_cells) else "partial_max_cells",
        regimes=regimes,
        budgets=budgets,
        intensities=intensities,
        scales=scales,
        run_records=run_records,
        rho_distributions=rho_stats,
        gate=gate_payload,
    )


if __name__ == "__main__":
    main()
