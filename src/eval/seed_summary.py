from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


SUMMARY_STD_COLUMNS = [
    "makespan_std",
    "tardiness_std",
    "instability_std",
    "runtime_std",
    "changed_operation_ratio_std",
    "changed_machine_ratio_std",
]


def load_block_summary(result_dir: str | Path) -> pd.DataFrame:
    result_dir = Path(result_dir)
    path = result_dir / "summary_by_method_seed_instance.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing block summary: {path}")
    return pd.read_csv(path)


def summarize_by_method_blocks(block_df: pd.DataFrame) -> pd.DataFrame:
    summary = block_df.groupby("method", as_index=False).agg(
        makespan_mean=("makespan", "mean"),
        makespan_std=("makespan", "std"),
        tardiness_mean=("total_tardiness", "mean"),
        tardiness_std=("total_tardiness", "std"),
        instability_mean=("instability", "mean"),
        instability_std=("instability", "std"),
        runtime_mean=("runtime", "mean"),
        runtime_std=("runtime", "std"),
        changed_operation_ratio_mean=("changed_operation_ratio", "mean"),
        changed_operation_ratio_std=("changed_operation_ratio", "std"),
        changed_machine_ratio_mean=("changed_machine_ratio", "mean"),
        changed_machine_ratio_std=("changed_machine_ratio", "std"),
        feasibility_rate_mean=("feasibility_rate", "mean"),
        n_instances=("instance_tag", "nunique"),
        n_seed_blocks=("seed", "size"),
        n_seeds_present=("seed", "nunique"),
    )
    for col in SUMMARY_STD_COLUMNS:
        if col in summary.columns:
            summary[col] = summary[col].fillna(0.0)
    return summary.sort_values("makespan_mean").reset_index(drop=True)


def summarize_seed_coverage(block_df: pd.DataFrame, expected_seeds: Iterable[int]) -> pd.DataFrame:
    expected = sorted({int(seed) for seed in expected_seeds})
    rows: list[dict[str, object]] = []
    for method, method_df in block_df.groupby("method"):
        present = sorted(method_df["seed"].dropna().astype(int).unique().tolist())
        missing = [seed for seed in expected if seed not in present]
        rows.append(
            {
                "method": method,
                "available_seeds": ",".join(str(seed) for seed in present),
                "missing_seeds": ",".join(str(seed) for seed in missing),
                "n_seeds_present": len(present),
                "expected_seed_count": len(expected),
                "n_instances": int(method_df["instance_tag"].nunique()),
                "n_seed_blocks": int(len(method_df)),
            }
        )
    return pd.DataFrame(rows).sort_values("method").reset_index(drop=True)
