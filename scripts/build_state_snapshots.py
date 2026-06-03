from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import pandas as pd

from _bootstrap import REPO_ROOT  # noqa: F401

from src.data.unified_parser import parse_instance
from src.env.dfjsp_env import DFJSPReschedulingEnv
from src.events.serialization import deserialize_dynamic_event
from src.scheduling.incumbent_builder import load_incumbent_schedule
from src.utils.config import load_merged_config
from src.utils.io import ensure_dir, load_json, save_jsonl


def _resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _infer_forced_release_ops(snapshot) -> list[int]:
    immutable = set(snapshot.completed_op_ids) | set(snapshot.active_op_ids)
    return sorted(op_id for op_id in snapshot.directly_impacted_op_ids if op_id not in immutable)


def _serialize_incumbent_assignments(env, op_ids: list[int]) -> dict[str, dict[str, float | int | None]]:
    assignments: dict[str, dict[str, float | int | None]] = {}
    for op_id in sorted(set(op_ids)):
        schedule = env.incumbent.operations.get(op_id)
        if schedule is None:
            continue
        assignments[str(op_id)] = {
            "machine": None if schedule.machine_id is None else int(schedule.machine_id),
            "start": None if schedule.start_time is None else float(schedule.start_time),
            "end": None if schedule.end_time is None else float(schedule.end_time),
        }
    return assignments


def _serialize_machine_states(env, tau: float) -> dict[str, dict[str, object]]:
    machine_states: dict[str, dict[str, object]] = {}
    for machine_id, calendar in sorted(env.incumbent.machine_calendars.items()):
        down = any(float(start) <= tau < float(end) for start, end in calendar.breakdowns)
        machine_states[str(machine_id)] = {
            "down": down,
            "idle_from": float(calendar.available_time),
            "breakdowns": [[float(start), float(end)] for start, end in calendar.breakdowns],
        }
    return machine_states


def _build_snapshot_row(episode_data: dict, event_data: dict, env) -> dict:
    snapshot = env.state_snapshot
    if snapshot is None:
        raise RuntimeError("State snapshot is not available.")
    tau = float(snapshot.current_time)
    forced_release_ops = _infer_forced_release_ops(snapshot)
    relevant_ops = list(snapshot.window_op_ids) + list(snapshot.directly_impacted_op_ids)
    return {
        "episode_id": str(episode_data["episode_id"]),
        "instance_id": str(episode_data["instance_id"]),
        "instance_path": str(episode_data["instance_path"]),
        "seed": int(episode_data["seed"]),
        "incumbent_ref": str(episode_data["incumbent_ref"]),
        "event_id": str(event_data["event_id"]),
        "tau": tau,
        "completed_ops": sorted(snapshot.completed_op_ids),
        "active_ops": sorted(snapshot.active_op_ids),
        "unfinished_ops": sorted(snapshot.unfinished_op_ids),
        "window_ops": sorted(snapshot.window_op_ids),
        "forced_release_ops": forced_release_ops,
        "incumbent_assignments": _serialize_incumbent_assignments(env, relevant_ops),
        "machine_states": _serialize_machine_states(env, tau),
        "event_context": {
            "type": str(snapshot.triggering_event_type),
            "affected_ops": sorted(snapshot.directly_impacted_op_ids),
            "affected_machines": sorted(snapshot.affected_machine_ids),
            "payload": deepcopy(event_data["payload"]),
        },
        "window_size": len(snapshot.window_op_ids),
        "forced_release_count": len(forced_release_ops),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs="*", default=[])
    parser.add_argument("--episodes-dir", default="outputs/episodes")
    parser.add_argument("--output-path", default="outputs/states/state_snapshots.jsonl")
    parser.add_argument("--manifest-csv", default=None)
    args = parser.parse_args()

    cfg = load_merged_config(*args.config) if args.config else {}
    episodes_dir = _resolve_path(args.episodes_dir)
    output_path = _resolve_path(args.output_path)
    manifest_csv = _resolve_path(args.manifest_csv) if args.manifest_csv else output_path.with_name("state_snapshots_manifest.csv")

    rows: list[dict] = []
    manifest_rows: list[dict] = []
    for episode_path in sorted(episodes_dir.glob("*.json")):
        episode_data = load_json(episode_path)
        if not isinstance(episode_data, dict) or "episode_id" not in episode_data or "incumbent_ref" not in episode_data:
            continue
        incumbent_data = load_json(_resolve_path(episode_data["incumbent_ref"]))
        due_factor = float(cfg.get("data", {}).get("due_date_rule", {}).get("factor", 1.5))
        instance = parse_instance(
            _resolve_path(episode_data["instance_path"]),
            family=cfg.get("data", {}).get("family", "fjsp"),
            due_date_factor=due_factor,
        )
        env = DFJSPReschedulingEnv(instance, cfg)
        env.reset()
        env.incumbent = load_incumbent_schedule(instance, incumbent_data)
        env.initial_instance = deepcopy(instance)
        env.instance = deepcopy(instance)

        episode_count = 0
        for fallback_id, event_data in enumerate(episode_data.get("events", [])):
            event = deserialize_dynamic_event(event_data, fallback_event_id=fallback_id)
            env.apply_event(event)
            env.build_window()
            rows.append(_build_snapshot_row(episode_data, event_data, env))
            episode_count += 1

        manifest_rows.append(
            {
                "episode_id": episode_data["episode_id"],
                "instance_id": episode_data["instance_id"],
                "seed": episode_data["seed"],
                "num_events": episode_count,
                "episode_path": str(episode_path),
            }
        )

    ensure_dir(output_path.parent)
    save_jsonl(rows, output_path)
    pd.DataFrame(manifest_rows).sort_values(["instance_id", "seed"]).to_csv(manifest_csv, index=False)
    print(f"Wrote {len(rows)} snapshot rows to {output_path}")


if __name__ == "__main__":
    main()
