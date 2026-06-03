from __future__ import annotations

import argparse
import time
from copy import deepcopy
from pathlib import Path

import pandas as pd
import torch

from _bootstrap import REPO_ROOT  # noqa: F401

from src.data.teacher_trace_io import freeze_graph_tensors
from src.data.unified_parser import parse_instance
from src.env.dfjsp_env import DFJSPReschedulingEnv
from src.events.serialization import deserialize_dynamic_event
from src.scheduling.incumbent_builder import load_incumbent_schedule
from src.solver.base import RepairDecision, RepairSolverResult
from src.solver.teacher_labeler import build_teacher_trace_decision, labels_from_teacher_decision
from src.utils.config import load_merged_config
from src.utils.io import ensure_dir, load_json, save_jsonl


def _resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _to_repo_relative(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(resolved)


def _infer_forced_release_ops(snapshot) -> list[int]:
    immutable = set(snapshot.completed_op_ids) | set(snapshot.active_op_ids)
    return sorted(op_id for op_id in snapshot.directly_impacted_op_ids if op_id not in immutable)


def _build_noop_teacher_result(env) -> RepairSolverResult:
    objective = env.compute_objective()
    return RepairSolverResult(
        feasible=True,
        solver_status="no_releasable_ops",
        updated_operations={},
        objective_value=float(objective.weighted_sum),
        metadata={"teacher_stage": "noop"},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs="*", default=[])
    parser.add_argument("--episodes-dir", default="outputs/episodes")
    parser.add_argument("--output-dir", default="outputs/teacher/traces")
    parser.add_argument("--trace-jsonl", default=None)
    parser.add_argument("--manifest-csv", default=None)
    args = parser.parse_args()

    cfg = load_merged_config(*args.config) if args.config else {}
    due_factor = float(cfg.get("data", {}).get("due_date_rule", {}).get("factor", 1.5))

    episodes_dir = _resolve_path(args.episodes_dir)
    output_dir = ensure_dir(_resolve_path(args.output_dir))
    trace_jsonl = _resolve_path(args.trace_jsonl) if args.trace_jsonl else output_dir / "teacher_trace_rows.jsonl"
    manifest_csv = _resolve_path(args.manifest_csv) if args.manifest_csv else output_dir / "teacher_trace_manifest.csv"

    trace_rows: list[dict] = []
    manifest_rows: list[dict] = []

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

        episode_samples: list[dict] = []
        for fallback_id, event_data in enumerate(episode_data.get("events", [])):
            event = deserialize_dynamic_event(event_data, fallback_event_id=fallback_id)
            env.apply_event(event)
            env.build_window()
            snapshot = env.state_snapshot
            if snapshot is None:
                raise RuntimeError("State snapshot is not available after build_window().")

            graph = freeze_graph_tensors(env.export_state_for_policy())
            immutable_op_ids = sorted(set(snapshot.completed_op_ids + snapshot.active_op_ids))
            releasable_op_ids = [op_id for op_id in graph["op_ids"] if op_id not in immutable_op_ids]

            if releasable_op_ids:
                teacher_start = time.perf_counter()
                teacher_decision, teacher_result = build_teacher_trace_decision(
                    subproblem=env.export_subproblem_for_solver(decision=None),
                    solver_cfg=cfg["solver"],
                    immutable_op_ids=immutable_op_ids,
                    releasable_op_ids=releasable_op_ids,
                    directly_impacted_op_ids=snapshot.directly_impacted_op_ids,
                    shrink_cfg=cfg.get("teacher_trace_batch", {}).get("shrink", {}),
                )
                teacher_runtime_sec = time.perf_counter() - teacher_start
            else:
                teacher_decision = RepairDecision(
                    immutable_op_ids=immutable_op_ids,
                    kept_op_ids=[],
                    released_op_ids=[],
                    metadata={"teacher_stage": "noop"},
                )
                teacher_result = _build_noop_teacher_result(env)
                teacher_runtime_sec = 0.0

            keep_labels, release_labels = labels_from_teacher_decision(graph["op_ids"], teacher_decision)
            forced_release_ops = _infer_forced_release_ops(snapshot)

            sample_metadata = {
                "episode_id": str(episode_data["episode_id"]),
                "event_id": str(event_data["event_id"]),
                "instance_id": str(episode_data["instance_id"]),
                "instance_path": str(episode_data["instance_path"]),
                "incumbent_ref": str(episode_data["incumbent_ref"]),
                "seed": int(episode_data["seed"]),
                "event_time": float(event.time),
                "event_type": event.event_type,
                "num_window_ops": len(graph["op_ids"]),
                "forced_release_count": len(forced_release_ops),
                "teacher_release_count": len(teacher_decision.released_op_ids),
                "teacher_keep_count": len(teacher_decision.kept_op_ids),
                "teacher_objective": teacher_result.objective_value,
                "teacher_feasible": bool(teacher_result.feasible),
                "teacher_solver_status": str(teacher_result.solver_status),
                "teacher_runtime_sec": teacher_runtime_sec,
                "teacher_stage": str(teacher_decision.metadata.get("teacher_stage", "unknown")),
            }
            episode_samples.append(
                {
                    "graph_tensors": graph,
                    "keep_labels": keep_labels,
                    "release_labels": release_labels,
                    "metadata": sample_metadata,
                }
            )
            trace_rows.append(sample_metadata)

            if teacher_result.feasible and teacher_result.updated_operations:
                env.commit_solver_result(teacher_result)

        shard_path = output_dir / f"{episode_data['episode_id']}.pt"
        torch.save(episode_samples, shard_path)
        manifest_rows.append(
            {
                "episode_id": episode_data["episode_id"],
                "instance_id": episode_data["instance_id"],
                "seed": episode_data["seed"],
                "num_events": len(episode_samples),
                "teacher_shard_path": _to_repo_relative(shard_path),
            }
        )

    save_jsonl(trace_rows, trace_jsonl)
    pd.DataFrame(manifest_rows).sort_values(["instance_id", "seed"]).to_csv(manifest_csv, index=False)
    print(f"Wrote {len(trace_rows)} teacher trace rows to {trace_jsonl}")


if __name__ == "__main__":
    main()
