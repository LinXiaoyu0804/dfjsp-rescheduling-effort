from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path

from _bootstrap import REPO_ROOT  # noqa: F401

from src.data.unified_parser import parse_instance


@dataclass(frozen=True, slots=True)
class SyntheticGroupSpec:
    group_name: str
    tag_prefix: str
    num_jobs: int
    num_machines: int
    num_instances: int = 5
    ops_per_job_low: int = 5
    ops_per_job_high: int = 10


GROUP_SPECS = [
    SyntheticGroupSpec(group_name="Scale-S", tag_prefix="syn_30x10", num_jobs=30, num_machines=10),
    SyntheticGroupSpec(group_name="Scale-M", tag_prefix="syn_50x15", num_jobs=50, num_machines=15),
    SyntheticGroupSpec(group_name="Scale-L", tag_prefix="syn_100x20", num_jobs=100, num_machines=20),
]


def _build_job_row(num_machines: int, rng: random.Random, ops_low: int, ops_high: int) -> tuple[list[int], int, int]:
    num_ops = rng.randint(ops_low, ops_high)
    max_choice = max(2, math.ceil(num_machines / 3))
    row: list[int] = [num_ops]
    total_options = 0
    for _ in range(num_ops):
        num_options = rng.randint(2, max_choice)
        total_options += num_options
        row.append(num_options)
        machine_ids = sorted(rng.sample(range(1, num_machines + 1), num_options))
        for machine_id in machine_ids:
            proc_time = rng.randint(1, 99)
            row.extend([machine_id, proc_time])
    return row, num_ops, total_options


def _build_instance_text(spec: SyntheticGroupSpec, seed: int) -> tuple[str, int]:
    rng = random.Random(seed)
    rows: list[list[int]] = []
    total_ops = 0
    total_options = 0
    for _ in range(spec.num_jobs):
        row, job_ops, job_options = _build_job_row(
            num_machines=spec.num_machines,
            rng=rng,
            ops_low=spec.ops_per_job_low,
            ops_high=spec.ops_per_job_high,
        )
        rows.append(row)
        total_ops += job_ops
        total_options += job_options

    avg_flex = max(2, int(round(total_options / max(1, total_ops))))
    lines = [f"{spec.num_jobs}\t{spec.num_machines}\t{avg_flex}"]
    lines.extend(" ".join(str(token) for token in row) for row in rows)
    return "\n".join(lines) + "\n", total_ops


def _validate_instance(path: Path) -> tuple[int, int, int]:
    instance = parse_instance(path, family="fjsp")
    return instance.num_jobs, instance.num_machines, instance.num_operations


def generate_instances(output_root: Path, base_seed: int, overwrite: bool) -> list[dict[str, object]]:
    output_root.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []

    for group_idx, spec in enumerate(GROUP_SPECS):
        group_dir = output_root / spec.group_name
        group_dir.mkdir(parents=True, exist_ok=True)
        for instance_idx in range(1, spec.num_instances + 1):
            instance_seed = base_seed + group_idx * 10_000 + instance_idx
            stem = f"{spec.tag_prefix}_{instance_idx:02d}"
            path = group_dir / f"{stem}.fjs"
            if path.exists() and not overwrite:
                raise FileExistsError(f"Refusing to overwrite existing instance: {path}")

            text, generated_total_ops = _build_instance_text(spec, seed=instance_seed)
            path.write_text(text, encoding="utf-8")

            n_jobs, n_machines, total_ops = _validate_instance(path)
            print(
                f"[ok] {stem}: n_jobs={n_jobs}, n_machines={n_machines}, total_ops={total_ops}, "
                f"seed={instance_seed}"
            )
            summary_rows.append(
                {
                    "group_name": spec.group_name,
                    "instance_tag": stem,
                    "path": path.as_posix(),
                    "seed": instance_seed,
                    "n_jobs": n_jobs,
                    "n_machines": n_machines,
                    "total_ops": total_ops,
                    "generated_total_ops": generated_total_ops,
                }
            )

    return summary_rows


def write_summary(rows: list[dict[str, object]], output_root: Path) -> Path:
    summary_path = output_root / "synthetic_instance_summary.csv"
    if not rows:
        raise ValueError("No synthetic instances were generated.")
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic large-scale FJSP instances in project-native .fjs format.")
    parser.add_argument(
        "--output-root",
        default="data/raw/fjsp/synthetic_scaling",
        help="Directory where synthetic instances will be written.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=20260331,
        help="Base RNG seed used to derive per-instance seeds.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing synthetic instances if they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    rows = generate_instances(output_root=output_root, base_seed=int(args.base_seed), overwrite=bool(args.overwrite))
    summary_path = write_summary(rows, output_root)
    print(f"[done] Wrote {len(rows)} synthetic instances under {output_root.as_posix()}")
    print(f"[done] Summary saved to {summary_path.as_posix()}")


if __name__ == "__main__":
    main()
