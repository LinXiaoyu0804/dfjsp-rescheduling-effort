from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from _bootstrap import REPO_ROOT  # noqa: F401

from src.data.unified_parser import parse_instance
from src.env.dfjsp_env import DFJSPReschedulingEnv
from src.events.generator import generate_dynamic_events
from src.utils.config import load_merged_config, load_yaml


def load_manifest(path: str | Path) -> dict[str, Any]:
    data = load_yaml(path)
    if "benchmark_manifest" not in data:
        raise KeyError(f"Manifest file {path} must contain 'benchmark_manifest'")
    return data["benchmark_manifest"]


def validate_instance(instance_spec: dict[str, Any], base_cfg: dict[str, Any], dry_env: bool) -> dict[str, Any]:
    path = Path(instance_spec["path"])
    row: dict[str, Any] = {
        "tag": instance_spec.get("tag", path.stem),
        "family": instance_spec.get("family", base_cfg["data"].get("family", "auto")),
        "path": str(path),
        "exists": path.exists(),
        "parse_ok": False,
        "env_ok": False,
        "num_jobs": None,
        "num_machines": None,
        "num_operations": None,
        "num_events": None,
        "note": "",
    }
    if not path.exists():
        row["note"] = "missing_file"
        return row

    try:
        due_factor = float(base_cfg["data"].get("due_date_rule", {}).get("factor", 1.5))
        instance = parse_instance(path, family=row["family"], due_date_factor=due_factor)
        row["parse_ok"] = True
        row["num_jobs"] = instance.num_jobs
        row["num_machines"] = instance.num_machines
        row["num_operations"] = instance.num_operations
    except Exception as exc:  # noqa: BLE001
        row["note"] = f"parse_failed: {type(exc).__name__}: {exc}"
        return row

    if dry_env:
        try:
            cfg = dict(base_cfg)
            cfg["data"] = dict(base_cfg["data"])
            cfg["data"]["instance_path"] = str(path)
            cfg["data"]["family"] = row["family"]
            env = DFJSPReschedulingEnv(instance, cfg)
            env.reset()
            events = generate_dynamic_events(instance, cfg["events"], seed=int(cfg["experiment"]["seed"]))
            row["num_events"] = len(events)
            if events:
                env.apply_event(events[0])
                env.build_window()
            row["env_ok"] = True
        except Exception as exc:  # noqa: BLE001
            row["note"] = f"env_failed: {type(exc).__name__}: {exc}"
            return row

    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs="+", required=True, help="Base configs used to construct env/event defaults.")
    parser.add_argument("--manifest", nargs="+", required=True, help="One or more benchmark manifest YAML files.")
    parser.add_argument("--output", required=False, help="Optional CSV output path.")
    parser.add_argument("--no-env-check", action="store_true", help="Skip env/event dry-run validation.")
    args = parser.parse_args()

    base_cfg = load_merged_config(*args.config)
    rows: list[dict[str, Any]] = []
    for manifest_path in args.manifest:
        manifest = load_manifest(manifest_path)
        for instance_spec in manifest["instances"]:
            row = validate_instance(instance_spec, base_cfg, dry_env=not args.no_env_check)
            row["manifest"] = manifest["name"]
            row["split"] = instance_spec.get("split", "")
            rows.append(row)

    df = pd.DataFrame(rows)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
