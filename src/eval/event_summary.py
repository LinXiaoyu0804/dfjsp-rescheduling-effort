from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd

WINDOW_SIZE_BUCKETS: tuple[tuple[float, str], ...] = (
    (80.0, "0-80"),
    (160.0, "81-160"),
    (240.0, "161-240"),
    (math.inf, "241+"),
)
FORCED_RELEASE_BUCKETS: tuple[tuple[float, str], ...] = (
    (0.0, "0"),
    (2.0, "1-2"),
    (5.0, "3-5"),
    (math.inf, "6+"),
)
MOTIF_COUNT_BUCKETS: tuple[tuple[float, str], ...] = (
    (15.0, "0-15"),
    (31.0, "16-31"),
    (63.0, "32-63"),
    (math.inf, "64+"),
)

_INSTANCE_GROUP_ORDER = {
    "mk1_mk5": 0,
    "mk6_mk7": 1,
    "mk8_mk10": 2,
    "synthetic_30x10": 3,
    "synthetic_50x15": 4,
    "synthetic_100x20": 5,
    "synthetic": 6,
    "other": 7,
}
_STRATIFIER_ORDER = {
    "window_size": 0,
    "forced_release_count": 1,
    "motif_count": 2,
}
_BUCKET_ORDER = {
    "window_size": {label: index for index, (_, label) in enumerate(WINDOW_SIZE_BUCKETS)},
    "forced_release_count": {label: index for index, (_, label) in enumerate(FORCED_RELEASE_BUCKETS)},
    "motif_count": {label: index for index, (_, label) in enumerate(MOTIF_COUNT_BUCKETS)},
}


def _bucketize_value(value: float | int | None, buckets: tuple[tuple[float, str], ...]) -> str:
    if value is None or pd.isna(value):
        return "unknown"
    numeric = float(value)
    for upper, label in buckets:
        if numeric <= upper:
            return label
    return buckets[-1][1]


def _infer_instance_group(instance_id: str) -> str:
    lowered = str(instance_id).strip().lower()
    synthetic_match = re.search(r"syn_(\d+)x(\d+)", lowered)
    if synthetic_match:
        return f"synthetic_{synthetic_match.group(1)}x{synthetic_match.group(2)}"
    match = re.search(r"mk(\d+)", lowered)
    if match:
        mk_id = int(match.group(1))
        if mk_id <= 5:
            return "mk1_mk5"
        if mk_id <= 7:
            return "mk6_mk7"
        if mk_id <= 10:
            return "mk8_mk10"
    if "synthetic" in lowered:
        return "synthetic"
    return "other"


def _ensure_numeric(df: pd.DataFrame, column: str, default: float = 0.0) -> None:
    if column not in df:
        df[column] = default
        return
    df[column] = pd.to_numeric(df[column], errors="coerce").fillna(default)


def _ensure_string(df: pd.DataFrame, column: str, default: str = "") -> None:
    if column not in df:
        df[column] = default
        return
    df[column] = df[column].fillna(default).astype(str)


def normalize_event_metrics(rows: list[dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    df = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "method",
                "instance_id",
                "event_id",
                "status",
                "selection_source",
                "operator_selection_mode",
                "window_size",
                "forced_release_count",
                "motif_count",
                "selected_motif_count",
                "released_op_count",
                "reward_delta",
                "weighted_objective_after",
                "solver_runtime_sec",
                "changed_op_ratio",
                "changed_machine_ratio",
                "operator_release_match_score",
                "operator_cost_match_score",
                "operator_gain_density_match_score",
                "instance_group",
                "window_size_bin",
                "forced_release_bin",
                "motif_count_bin",
                "selection_active",
                "is_feasible",
                "positive_reward",
                "alns_used",
                "relaxed_used",
            ]
        )

    for column in (
        "window_size",
        "forced_release_count",
        "motif_count",
        "selected_motif_count",
        "released_op_count",
        "reward_delta",
        "weighted_objective_after",
        "solver_runtime_sec",
        "changed_op_ratio",
        "changed_machine_ratio",
        "operator_release_match_score",
        "operator_cost_match_score",
        "operator_gain_density_match_score",
        "makespan_after",
        "tardiness_after",
    ):
        _ensure_numeric(df, column)

    for column in ("method", "instance_id", "event_id", "status", "selection_source", "operator_selection_mode"):
        _ensure_string(df, column)

    df["instance_group"] = df["instance_id"].map(_infer_instance_group)
    df["window_size_bin"] = df["window_size"].map(lambda value: _bucketize_value(value, WINDOW_SIZE_BUCKETS))
    df["forced_release_bin"] = df["forced_release_count"].map(
        lambda value: _bucketize_value(value, FORCED_RELEASE_BUCKETS)
    )
    df["motif_count_bin"] = df["motif_count"].map(lambda value: _bucketize_value(value, MOTIF_COUNT_BUCKETS))
    df["selection_active"] = df["selected_motif_count"] > 0
    df["is_feasible"] = df["status"].str.lower().eq("feasible")
    df["positive_reward"] = df["reward_delta"] > 1e-9
    df["alns_used"] = df["selection_source"].str.lower().eq("alns_lite")
    df["relaxed_used"] = df["operator_selection_mode"].str.lower().str.contains("relaxed", regex=False)
    return df


def build_instance_summary(rows: list[dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    df = normalize_event_metrics(rows)
    columns = [
        "method",
        "instance_id",
        "num_events",
        "mean_motif_count",
        "mean_selected_motif_count",
        "mean_released_op_count",
        "mean_makespan",
        "mean_tardiness",
        "mean_changed_op_ratio",
        "mean_changed_machine_ratio",
        "feasible_rate",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)

    summary = (
        df.groupby(["method", "instance_id"], as_index=False)
        .agg(
            num_events=("event_id", "count"),
            mean_motif_count=("motif_count", "mean"),
            mean_selected_motif_count=("selected_motif_count", "mean"),
            mean_released_op_count=("released_op_count", "mean"),
            mean_makespan=("makespan_after", "mean"),
            mean_tardiness=("tardiness_after", "mean"),
            mean_changed_op_ratio=("changed_op_ratio", "mean"),
            mean_changed_machine_ratio=("changed_machine_ratio", "mean"),
            feasible_rate=("is_feasible", "mean"),
        )
        .sort_values(["method", "instance_id"])
        .reset_index(drop=True)
    )
    return summary[columns]


def _build_rich_group_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    columns = group_cols + [
        "num_events",
        "mean_window_size",
        "mean_forced_release_count",
        "mean_motif_count",
        "selection_rate",
        "mean_selected_motif_count",
        "mean_released_op_count",
        "mean_reward_delta",
        "positive_reward_rate",
        "mean_weighted_objective_after",
        "mean_solver_runtime_sec",
        "mean_changed_op_ratio",
        "mean_changed_machine_ratio",
        "feasible_rate",
        "alns_usage_rate",
        "relaxed_usage_rate",
        "mean_operator_release_match_score",
        "mean_operator_cost_match_score",
        "mean_operator_gain_density_match_score",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)

    summary = (
        df.groupby(group_cols, as_index=False)
        .agg(
            num_events=("event_id", "count"),
            mean_window_size=("window_size", "mean"),
            mean_forced_release_count=("forced_release_count", "mean"),
            mean_motif_count=("motif_count", "mean"),
            selection_rate=("selection_active", "mean"),
            mean_selected_motif_count=("selected_motif_count", "mean"),
            mean_released_op_count=("released_op_count", "mean"),
            mean_reward_delta=("reward_delta", "mean"),
            positive_reward_rate=("positive_reward", "mean"),
            mean_weighted_objective_after=("weighted_objective_after", "mean"),
            mean_solver_runtime_sec=("solver_runtime_sec", "mean"),
            mean_changed_op_ratio=("changed_op_ratio", "mean"),
            mean_changed_machine_ratio=("changed_machine_ratio", "mean"),
            feasible_rate=("is_feasible", "mean"),
            alns_usage_rate=("alns_used", "mean"),
            relaxed_usage_rate=("relaxed_used", "mean"),
            mean_operator_release_match_score=("operator_release_match_score", "mean"),
            mean_operator_cost_match_score=("operator_cost_match_score", "mean"),
            mean_operator_gain_density_match_score=("operator_gain_density_match_score", "mean"),
        )
        .reset_index(drop=True)
    )
    return summary[columns]


def build_instance_group_summary(rows: list[dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    df = normalize_event_metrics(rows)
    summary = _build_rich_group_summary(df, ["method", "instance_group"])
    if summary.empty:
        return summary
    summary["_group_order"] = summary["instance_group"].map(lambda value: _INSTANCE_GROUP_ORDER.get(value, 99))
    summary = summary.sort_values(["method", "_group_order"]).drop(columns="_group_order").reset_index(drop=True)
    return summary


def build_complexity_summary(rows: list[dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    df = normalize_event_metrics(rows)
    columns = [
        "method",
        "instance_group",
        "stratifier",
        "bucket",
        "num_events",
        "mean_window_size",
        "mean_forced_release_count",
        "mean_motif_count",
        "selection_rate",
        "mean_selected_motif_count",
        "mean_released_op_count",
        "mean_reward_delta",
        "positive_reward_rate",
        "mean_weighted_objective_after",
        "mean_solver_runtime_sec",
        "mean_changed_op_ratio",
        "mean_changed_machine_ratio",
        "feasible_rate",
        "alns_usage_rate",
        "relaxed_usage_rate",
        "mean_operator_release_match_score",
        "mean_operator_cost_match_score",
        "mean_operator_gain_density_match_score",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)

    frames: list[pd.DataFrame] = []
    for stratifier, bucket_column in (
        ("window_size", "window_size_bin"),
        ("forced_release_count", "forced_release_bin"),
        ("motif_count", "motif_count_bin"),
    ):
        part = _build_rich_group_summary(df, ["method", "instance_group", bucket_column]).rename(
            columns={bucket_column: "bucket"}
        )
        part.insert(2, "stratifier", stratifier)
        frames.append(part)

    combined = pd.concat(frames, ignore_index=True)
    combined["_group_order"] = combined["instance_group"].map(lambda value: _INSTANCE_GROUP_ORDER.get(value, 99))
    combined["_stratifier_order"] = combined["stratifier"].map(lambda value: _STRATIFIER_ORDER.get(value, 99))
    combined["_bucket_order"] = combined.apply(
        lambda row: _BUCKET_ORDER.get(str(row["stratifier"]), {}).get(str(row["bucket"]), 99),
        axis=1,
    )
    combined = (
        combined.sort_values(["method", "_group_order", "_stratifier_order", "_bucket_order", "bucket"])
        .drop(columns=["_group_order", "_stratifier_order", "_bucket_order"])
        .reset_index(drop=True)
    )
    return combined[columns]
