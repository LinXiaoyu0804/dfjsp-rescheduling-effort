from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from _bootstrap import REPO_ROOT  # noqa: F401

from src.data.unified_parser import parse_instance
from src.events.generator import generate_dynamic_events
from src.events.serialization import serialize_dynamic_event
from src.utils.config import load_merged_config, load_yaml
from src.utils.io import ensure_dir, load_json, save_json


def _resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _to_repo_relative(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _load_manifest(manifest_path: str | Path) -> dict[str, Any]:
    manifest = load_yaml(_resolve_path(manifest_path))
    if "benchmark_manifest" not in manifest:
        raise KeyError(f"Manifest {manifest_path} is missing 'benchmark_manifest'.")
    return manifest["benchmark_manifest"]


def _load_instance_specs(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(spec.get("tag", Path(spec["path"]).stem)): spec
        for spec in manifest["instances"]
    }


def _resolve_manifest_arg(cfg: dict[str, Any], manifest_arg: str | None) -> str:
    if manifest_arg:
        return manifest_arg
    protocol = cfg.get("protocol", {})
    if protocol.get("train_manifest"):
        return str(protocol["train_manifest"])
    raise ValueError("Please provide --manifest or set protocol.train_manifest in the config.")


def _event_prefix(event_type: str) -> str:
    return {
        "job_arrival": "arr",
        "machine_breakdown": "bd",
        "processing_time_perturbation": "pt",
        "compound": "cmp",
    }[event_type]


def _serialize_episode_events(events: list) -> list[dict[str, Any]]:
    counters: dict[str, int] = defaultdict(int)
    serialized: list[dict[str, Any]] = []
    for event in events:
        prefix = _event_prefix(event.event_type)
        event_id = f"{prefix}_{counters[event.event_type]}"
        counters[event.event_type] += 1
        serialized.append(serialize_dynamic_event(event, event_id=event_id))
    return serialized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs="*", default=[])
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--incumbent-dir", default="outputs/incumbents")
    parser.add_argument("--output-dir", default="outputs/episodes")
    parser.add_argument("--manifest-csv", default=None)
    parser.add_argument("--manifest-json", default=None)
    args = parser.parse_args()

    cfg = load_merged_config(*args.config) if args.config else {}
    manifest = _load_manifest(_resolve_manifest_arg(cfg, args.manifest))
    instance_specs = _load_instance_specs(manifest)
    incumbent_dir = _resolve_path(args.incumbent_dir)
    output_dir = ensure_dir(_resolve_path(args.output_dir))
    manifest_csv = _resolve_path(args.manifest_csv) if args.manifest_csv else output_dir / "episode_manifest.csv"
    manifest_json = _resolve_path(args.manifest_json) if args.manifest_json else output_dir / "episode_manifest.json"

    rows: list[dict[str, Any]] = []
    for incumbent_path in sorted(incumbent_dir.glob("*.json")):
        incumbent_data = load_json(incumbent_path)
        instance_id = str(incumbent_data["instance_id"])
        seed = int(incumbent_data["seed"])
        instance_spec = instance_specs.get(instance_id)
        if instance_spec is None:
            continue

        due_factor = float(cfg.get("data", {}).get("due_date_rule", {}).get("factor", 1.5))
        instance = parse_instance(
            _resolve_path(instance_spec["path"]),
            family=instance_spec.get("family", cfg.get("data", {}).get("family", "auto")),
            due_date_factor=due_factor,
        )
        events = generate_dynamic_events(instance, cfg.get("events", {}), seed=seed)
        serialized_events = _serialize_episode_events(events)
        episode_id = f"{instance_id}_seed{seed:02d}_ep001"
        episode_payload = {
            "episode_id": episode_id,
            "instance_id": instance_id,
            "instance_path": _to_repo_relative(_resolve_path(instance_spec["path"])),
            "seed": seed,
            "incumbent_ref": _to_repo_relative(incumbent_path),
            "events": serialized_events,
        }
        output_path = output_dir / f"{episode_id}.json"
        save_json(episode_payload, output_path)

        arrival_times = [event["time"] for event in serialized_events if event["type"] == "job_arrival"]
        breakdown_times = [event["time"] for event in serialized_events if event["type"] == "machine_breakdown"]
        rows.append(
            {
                "episode_id": episode_id,
                "instance_id": instance_id,
                "seed": seed,
                "incumbent_ref": _to_repo_relative(incumbent_path),
                "episode_path": _to_repo_relative(output_path),
                "num_events": len(serialized_events),
                "num_arrivals": len(arrival_times),
                "num_breakdowns": len(breakdown_times),
                "first_event_time": min((event["time"] for event in serialized_events), default=None),
                "last_event_time": max((event["time"] for event in serialized_events), default=None),
            }
        )

    manifest_df = pd.DataFrame(rows).sort_values(["instance_id", "seed"]).reset_index(drop=True)
    ensure_dir(manifest_csv.parent)
    ensure_dir(manifest_json.parent)
    manifest_df.to_csv(manifest_csv, index=False)
    save_json({"episodes": rows}, manifest_json)
    print(manifest_df.to_string(index=False))


if __name__ == "__main__":
    main()
