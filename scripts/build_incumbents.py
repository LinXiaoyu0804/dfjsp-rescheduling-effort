from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from _bootstrap import REPO_ROOT  # noqa: F401

from src.data.unified_parser import parse_instance
from src.scheduling.incumbent_builder import build_offline_incumbent, serialize_incumbent
from src.utils.config import load_merged_config, load_yaml
from src.utils.io import ensure_dir, save_json


DEFAULT_OFFLINE_BUDGET_SEC = 60.0


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


def _resolve_manifest_arg(cfg: dict[str, Any], manifest_arg: str | None) -> str:
    if manifest_arg:
        return manifest_arg
    protocol = cfg.get("protocol", {})
    if protocol.get("train_manifest"):
        return str(protocol["train_manifest"])
    raise ValueError("Please provide --manifest or set protocol.train_manifest in the config.")


def _resolve_instance_id(instance_spec: dict[str, Any]) -> str:
    return str(instance_spec.get("tag", Path(instance_spec["path"]).stem))


def build_one_incumbent(
    cfg: dict[str, Any],
    instance_spec: dict[str, Any],
    seed: int,
    output_dir: Path,
    offline_budget_sec: float,
) -> dict[str, Any]:
    instance_path = _resolve_path(instance_spec["path"])
    due_factor = float(cfg.get("data", {}).get("due_date_rule", {}).get("factor", 1.5))
    instance = parse_instance(
        instance_path,
        family=instance_spec.get("family", cfg.get("data", {}).get("family", "auto")),
        due_date_factor=due_factor,
    )

    solver_cfg = dict(cfg.get("solver", {}))
    objective_weights = dict(solver_cfg.get("objective_weights", {}))
    objective_weights["instability"] = 0.0
    incumbent, metadata = build_offline_incumbent(
        instance=instance,
        solver_cfg=solver_cfg,
        offline_budget_sec=offline_budget_sec,
        objective_weights=objective_weights,
    )

    instance_id = _resolve_instance_id(instance_spec)
    incumbent_payload = serialize_incumbent(
        instance_id=instance_id,
        instance_path=_to_repo_relative(instance_path),
        seed=seed,
        incumbent=incumbent,
        solver_name=str(solver_cfg.get("name", "cp_sat")),
        offline_budget_sec=offline_budget_sec,
        objective=metadata["objective"],
        solver_status=str(metadata["solver_status"]),
        solver_runtime_sec=float(metadata["solver_runtime_sec"]),
        extra_metadata={"split": instance_spec.get("split"), "family": instance_spec.get("family", instance.family)},
    )

    output_path = output_dir / f"{instance_id}_seed{seed:02d}.json"
    save_json(incumbent_payload, output_path)
    return {
        "instance_id": instance_id,
        "instance_path": _to_repo_relative(instance_path),
        "seed": int(seed),
        "solver": incumbent_payload["solver"],
        "offline_budget_sec": float(offline_budget_sec),
        "solver_status": incumbent_payload["solver_status"],
        "solver_runtime_sec": float(incumbent_payload["solver_runtime_sec"]),
        "fallback_used": bool(metadata["fallback_used"]),
        "makespan": float(incumbent_payload["objective"]["makespan"]),
        "tardiness": float(incumbent_payload["objective"]["tardiness"]),
        "weighted_objective": float(incumbent_payload["objective"]["weighted_objective"]),
        "schedule_hash": incumbent_payload["schedule_hash"],
        "output_path": _to_repo_relative(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs="*", default=[])
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output-dir", default="outputs/incumbents")
    parser.add_argument("--summary-path", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0])
    parser.add_argument("--offline-budget-sec", type=float, default=DEFAULT_OFFLINE_BUDGET_SEC)
    args = parser.parse_args()

    cfg = load_merged_config(*args.config) if args.config else {}
    manifest_path = _resolve_manifest_arg(cfg, args.manifest)
    manifest = _load_manifest(manifest_path)
    output_dir = ensure_dir(_resolve_path(args.output_dir))
    summary_path = _resolve_path(args.summary_path) if args.summary_path else output_dir / "incumbent_summary.csv"

    rows: list[dict[str, Any]] = []
    for instance_spec in manifest["instances"]:
        for seed in args.seeds:
            rows.append(
                build_one_incumbent(
                    cfg=cfg,
                    instance_spec=instance_spec,
                    seed=int(seed),
                    output_dir=output_dir,
                    offline_budget_sec=float(args.offline_budget_sec),
                )
            )

    summary_df = pd.DataFrame(rows).sort_values(["instance_id", "seed"]).reset_index(drop=True)
    ensure_dir(summary_path.parent)
    summary_df.to_csv(summary_path, index=False)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
