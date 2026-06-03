from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.eval.intensity_sensitivity import bootstrap_mean_ci
from src.utils.io import ensure_dir, save_json


INTENSITY_ORDER = ["L0", "L1", "L2", "L3"]
EVENT_KEY = ["regime", "budget_sec", "scale", "instance_id", "seed", "episode_id", "event_id"]
FEATURE_COLUMNS = [
    "rho_t",
    "budget_sec",
    "flexibility",
    "contention",
    "propagation_depth",
    "event_footprint_ops",
    "window_size",
    "size",
    "event_type_arrival",
    "event_type_breakdown",
    "event_type_compound",
    "event_type_other",
]


@dataclass(frozen=True)
class ThresholdResult:
    epsilon: float
    rho_t: float | None
    label: str


def _as_float_series(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([default] * len(frame), index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default).astype(float)


def _event_type_flags(values: pd.Series) -> pd.DataFrame:
    normalized = values.fillna("other").astype(str).str.lower()
    return pd.DataFrame(
        {
            "event_type_arrival": (normalized == "arrival").astype(float),
            "event_type_breakdown": (normalized == "breakdown").astype(float),
            "event_type_compound": (normalized == "compound").astype(float),
            "event_type_other": (~normalized.isin(["arrival", "breakdown", "compound"])).astype(float),
        },
        index=values.index,
    )


def _cheapest_best_level(values: dict[str, float], *, tolerance: float = 1e-12) -> str:
    best_value = min(float(values[level]) for level in INTENSITY_ORDER)
    for level in INTENSITY_ORDER:
        if float(values[level]) <= best_value + tolerance:
            return level
    raise RuntimeError("unreachable cheapest-best selection")


def build_event_outcomes(records: Iterable[dict[str, Any]], *, gamma: float = 0.2) -> pd.DataFrame:
    frame = pd.DataFrame(list(records))
    if frame.empty:
        raise ValueError("No rho-boundary event records were supplied.")
    if "intensity_level" not in frame.columns:
        raise KeyError("Event records must include intensity_level.")
    frame = frame[frame["intensity_level"].isin(INTENSITY_ORDER)].copy()
    for key in EVENT_KEY:
        if key not in frame.columns:
            raise KeyError(f"Event records are missing key column: {key}")
    frame["J_after"] = (
        _as_float_series(frame, "makespan_after")
        + _as_float_series(frame, "tardiness_after")
        + float(gamma) * _as_float_series(frame, "instability_after")
    )
    frame["feasible_flag"] = (frame["status"].astype(str) == "feasible").astype(float)
    objective = frame.pivot_table(index=EVENT_KEY, columns="intensity_level", values="J_after", aggfunc="first")
    missing = [level for level in INTENSITY_ORDER if level not in objective.columns]
    if missing:
        raise ValueError(f"Missing intensity columns in outcome pivot: {missing}")
    objective = objective.dropna(subset=INTENSITY_ORDER).reset_index()

    l0_rows = frame[frame["intensity_level"] == "L0"].copy()
    attrs = l0_rows.drop_duplicates(subset=EVENT_KEY, keep="first")
    attr_cols = [
        *EVENT_KEY,
        "rho_t",
        "rho_t_foot",
        "flexibility",
        "contention",
        "propagation_depth",
        "event_footprint_ops",
        "event_footprint_machines",
        "window_size",
        "size",
        "event_type_bucket",
        "makespan_before",
        "tardiness_before",
        "instability_before",
    ]
    attrs = attrs[[col for col in attr_cols if col in attrs.columns]]
    outcomes = objective.merge(attrs, on=EVENT_KEY, how="left", validate="one_to_one")
    feasibility = (
        frame.groupby(EVENT_KEY, as_index=False)
        .agg(
            all_intensity_feasible=("feasible_flag", "min"),
            mean_intensity_feasible=("feasible_flag", "mean"),
        )
    )
    outcomes = outcomes.merge(feasibility, on=EVENT_KEY, how="left", validate="one_to_one")

    for level in INTENSITY_ORDER:
        outcomes[f"J_{level}"] = pd.to_numeric(outcomes[level], errors="coerce")
    j0 = outcomes["J_L0"].astype(float).to_numpy()
    best_levels: list[str] = []
    oracle_gain: list[float] = []
    for row in outcomes.itertuples(index=False):
        values = {level: float(getattr(row, level)) for level in INTENSITY_ORDER}
        best = _cheapest_best_level(values)
        best_levels.append(best)
        denom = max(1e-12, float(values["L0"]))
        oracle_gain.append(max(0.0, (float(values["L0"]) - float(values[best])) / denom))
    outcomes["oracle_best_intensity"] = best_levels
    outcomes["oracle_headroom"] = oracle_gain
    for level in INTENSITY_ORDER[1:]:
        outcomes[f"gain_{level}"] = (outcomes["J_L0"].astype(float) - outcomes[f"J_{level}"].astype(float)) / np.maximum(
            1e-12,
            j0,
        )
    outcomes["gain_L0"] = 0.0

    flags = _event_type_flags(outcomes.get("event_type_bucket", pd.Series(["other"] * len(outcomes))))
    for column in flags.columns:
        outcomes[column] = flags[column]
    for column in FEATURE_COLUMNS:
        if column not in outcomes.columns:
            outcomes[column] = 0.0
        outcomes[column] = pd.to_numeric(outcomes[column], errors="coerce").fillna(0.0).astype(float)
    return outcomes


def _prediction_best(predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    best_idx = np.argmax(predictions, axis=1) + 1
    best_gain = predictions[np.arange(predictions.shape[0]), best_idx - 1]
    return best_idx.astype(int), best_gain.astype(float)


def _tune_threshold(
    *,
    predicted_gain: np.ndarray,
    predicted_level_idx: np.ndarray,
    gain_matrix: np.ndarray,
) -> tuple[float, float, float]:
    candidates = [float("-inf"), float("inf"), 0.0]
    candidates.extend(float(value) for value in np.unique(predicted_gain[np.isfinite(predicted_gain)]))
    best: tuple[float, float, float] | None = None
    for threshold in sorted(set(candidates)):
        selected = np.where(predicted_gain > threshold, predicted_level_idx, 0)
        gains = gain_matrix[np.arange(gain_matrix.shape[0]), selected]
        mean_gain = float(np.mean(gains)) if gains.size else float("-inf")
        upgrade_rate = float(np.mean(selected != 0)) if selected.size else 0.0
        score = (mean_gain, -upgrade_rate, threshold)
        if best is None or score > (best[1], -best[2], best[0]):
            best = (float(threshold), mean_gain, upgrade_rate)
    if best is None:
        raise RuntimeError("No threshold candidates were evaluated.")
    return best


def evaluate_honest_policy(
    outcomes: pd.DataFrame,
    *,
    train_seed_parity: int = 0,
    random_state: int = 0,
) -> tuple[dict[str, Any], pd.DataFrame]:
    from sklearn.ensemble import GradientBoostingRegressor

    dataset = outcomes.copy()
    train_mask = (dataset["seed"].astype(int) % 2).to_numpy() == int(train_seed_parity)
    test_mask = ~train_mask
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError("Both train and test seed-parity splits must contain events.")
    x_all = dataset[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = dataset[[f"gain_{level}" for level in INTENSITY_ORDER[1:]]].to_numpy(dtype=float)

    predictions = np.zeros_like(y)
    for idx, level in enumerate(INTENSITY_ORDER[1:]):
        model = GradientBoostingRegressor(random_state=random_state)
        model.fit(x_all[train_mask], y[train_mask, idx])
        predictions[:, idx] = model.predict(x_all)

    predicted_idx, predicted_gain = _prediction_best(predictions)
    gain_matrix = dataset[["gain_L0", "gain_L1", "gain_L2", "gain_L3"]].to_numpy(dtype=float)
    threshold, train_mean_gain, train_upgrade_rate = _tune_threshold(
        predicted_gain=predicted_gain[train_mask],
        predicted_level_idx=predicted_idx[train_mask],
        gain_matrix=gain_matrix[train_mask],
    )
    selected_idx = np.where(predicted_gain > threshold, predicted_idx, 0)
    selected_gain = gain_matrix[np.arange(gain_matrix.shape[0]), selected_idx]
    selected_level = [INTENSITY_ORDER[int(idx)] for idx in selected_idx]

    event_eval = dataset[EVENT_KEY + ["rho_t", "oracle_headroom", "all_intensity_feasible"]].copy()
    event_eval["split"] = np.where(train_mask, "train", "test")
    event_eval["predicted_gain"] = predicted_gain
    event_eval["predicted_level"] = [INTENSITY_ORDER[int(idx)] for idx in predicted_idx]
    event_eval["selected_level"] = selected_level
    event_eval["policy_gain"] = selected_gain
    event_eval["oracle_gain"] = dataset["oracle_headroom"].astype(float).to_numpy()
    event_eval["threshold"] = float(threshold)

    test_policy_gain = event_eval.loc[test_mask, "policy_gain"].astype(float).sum()
    test_oracle_gain = event_eval.loc[test_mask, "oracle_gain"].astype(float).sum()
    summary = {
        "train_seed_parity": int(train_seed_parity),
        "test_seed_parity": 1 - int(train_seed_parity),
        "n_train_events": int(train_mask.sum()),
        "n_test_events": int(test_mask.sum()),
        "threshold": float(threshold),
        "train_mean_policy_gain": float(train_mean_gain),
        "train_upgrade_rate": float(train_upgrade_rate),
        "test_mean_policy_gain": float(event_eval.loc[test_mask, "policy_gain"].astype(float).mean()),
        "test_mean_oracle_gain": float(event_eval.loc[test_mask, "oracle_gain"].astype(float).mean()),
        "test_capture_fraction": float(test_policy_gain / test_oracle_gain) if test_oracle_gain > 0.0 else float("nan"),
        "test_upgrade_rate": float((event_eval.loc[test_mask, "selected_level"] != "L0").mean()),
    }
    return summary, event_eval


def _bootstrap_ratio_ci(
    numerator: Iterable[float],
    denominator: Iterable[float],
    *,
    seed: int,
    n_boot: int,
) -> tuple[float, float, float]:
    num = np.asarray([float(value) for value in numerator], dtype=float)
    den = np.asarray([float(value) for value in denominator], dtype=float)
    mask = np.isfinite(num) & np.isfinite(den)
    num = num[mask]
    den = den[mask]
    if num.size == 0 or den.sum() <= 0.0:
        return float("nan"), float("nan"), float("nan")
    ratio = float(num.sum() / den.sum())
    if num.size == 1:
        return ratio, ratio, ratio
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, num.size, size=(int(n_boot), num.size))
    den_boot = den[idx].sum(axis=1)
    num_boot = num[idx].sum(axis=1)
    valid = den_boot > 0.0
    if not valid.any():
        return ratio, float("nan"), float("nan")
    boot = num_boot[valid] / den_boot[valid]
    return ratio, float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def add_rho_quantile_bins(outcomes: pd.DataFrame, *, q: int = 6) -> pd.DataFrame:
    data = outcomes.copy()
    finite = np.isfinite(data["rho_t"].astype(float).to_numpy())
    if finite.sum() == 0:
        raise ValueError("No finite rho_t values are available for binning.")
    bins = pd.qcut(data.loc[finite, "rho_t"].astype(float), q=min(q, finite.sum()), duplicates="drop")
    data["rho_bin"] = "missing"
    data.loc[finite, "rho_bin"] = bins.astype(str)
    order = {label: idx for idx, label in enumerate(pd.Series(bins.astype(str)).drop_duplicates().tolist())}
    data["rho_bin_order"] = data["rho_bin"].map(order).fillna(-1).astype(int)
    return data


def summarize_groups(
    outcomes: pd.DataFrame,
    policy_eval: pd.DataFrame,
    *,
    group_col: str,
    summary_type: str,
    bootstrap_reps: int = 5000,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    policy_test = policy_eval[policy_eval["split"] == "test"].copy()
    for group_value, subset in outcomes.groupby(group_col, sort=True):
        subset = subset.copy()
        test_subset = policy_test[policy_test[group_col] == group_value] if group_col in policy_test.columns else policy_test.iloc[0:0]
        headroom = subset["oracle_headroom"].astype(float)
        mean_h, low_h, high_h = bootstrap_mean_ci(headroom * 100.0, seed=101, n_boot=bootstrap_reps)
        frac_01, frac_01_low, frac_01_high = bootstrap_mean_ci(
            (headroom > 0.001).astype(float), seed=102, n_boot=bootstrap_reps
        )
        frac_1, frac_1_low, frac_1_high = bootstrap_mean_ci(
            (headroom > 0.01).astype(float), seed=103, n_boot=bootstrap_reps
        )
        feas, feas_low, feas_high = bootstrap_mean_ci(
            subset["all_intensity_feasible"].astype(float), seed=104, n_boot=bootstrap_reps
        )
        capture, capture_low, capture_high = _bootstrap_ratio_ci(
            test_subset.get("policy_gain", pd.Series(dtype=float)),
            test_subset.get("oracle_gain", pd.Series(dtype=float)),
            seed=105,
            n_boot=bootstrap_reps,
        )
        rho_values = subset["rho_t"].astype(float)
        rows.append(
            {
                "summary_type": summary_type,
                "group": str(group_value),
                "n_events": int(len(subset)),
                "n_test_events": int(len(test_subset)),
                "rho_min": float(rho_values.min()),
                "rho_median": float(rho_values.median()),
                "rho_max": float(rho_values.max()),
                "mean_oracle_headroom_percent": mean_h,
                "mean_oracle_headroom_percent_ci95_low": low_h,
                "mean_oracle_headroom_percent_ci95_high": high_h,
                "frac_headroom_gt_0p1pct": frac_01,
                "frac_headroom_gt_0p1pct_ci95_low": frac_01_low,
                "frac_headroom_gt_0p1pct_ci95_high": frac_01_high,
                "frac_headroom_gt_1pct": frac_1,
                "frac_headroom_gt_1pct_ci95_low": frac_1_low,
                "frac_headroom_gt_1pct_ci95_high": frac_1_high,
                "capture_percent": capture * 100.0 if math.isfinite(capture) else float("nan"),
                "capture_percent_ci95_low": capture_low * 100.0 if math.isfinite(capture_low) else float("nan"),
                "capture_percent_ci95_high": capture_high * 100.0 if math.isfinite(capture_high) else float("nan"),
                "feasible_rate": feas,
                "feasible_rate_ci95_low": feas_low,
                "feasible_rate_ci95_high": feas_high,
            }
        )
    result = pd.DataFrame(rows)
    if summary_type == "rho_bin" and not result.empty:
        result = result.sort_values("rho_median")
    return result


def locate_headroom_thresholds(rho_summary: pd.DataFrame, epsilons: Iterable[float]) -> list[ThresholdResult]:
    ordered = rho_summary.sort_values("rho_median")
    thresholds: list[ThresholdResult] = []
    for epsilon in epsilons:
        epsilon_percent = float(epsilon) * 100.0
        hit = ordered[ordered["mean_oracle_headroom_percent"].astype(float) > epsilon_percent]
        if hit.empty:
            thresholds.append(ThresholdResult(float(epsilon), None, "not_observed"))
        else:
            thresholds.append(ThresholdResult(float(epsilon), float(hit.iloc[0]["rho_median"]), str(hit.iloc[0]["group"])))
    return thresholds


def locate_capture_threshold(rho_summary: pd.DataFrame) -> ThresholdResult:
    ordered = rho_summary.sort_values("rho_median")
    capture = pd.to_numeric(ordered["capture_percent"], errors="coerce")
    low = pd.to_numeric(ordered["capture_percent_ci95_low"], errors="coerce")
    hit = ordered[(capture > 50.0) & (low > 0.0)]
    if hit.empty:
        return ThresholdResult(0.5, None, "not_observed")
    return ThresholdResult(0.5, float(hit.iloc[0]["rho_median"]), str(hit.iloc[0]["group"]))


def write_rho_boundary_outputs(
    *,
    records: Iterable[dict[str, Any]],
    output_root: Path,
    gamma: float = 0.2,
    bootstrap_reps: int = 5000,
    train_seed_parity: int = 0,
) -> dict[str, Any]:
    ensure_dir(output_root)
    outcomes = add_rho_quantile_bins(build_event_outcomes(records, gamma=gamma))
    policy_summary, policy_eval = evaluate_honest_policy(outcomes, train_seed_parity=train_seed_parity)
    policy_eval = policy_eval.merge(outcomes[EVENT_KEY + ["rho_bin"]], on=EVENT_KEY, how="left", validate="one_to_one")

    rho_summary = summarize_groups(
        outcomes,
        policy_eval,
        group_col="rho_bin",
        summary_type="rho_bin",
        bootstrap_reps=bootstrap_reps,
    )
    regime_summary = summarize_groups(
        outcomes,
        policy_eval,
        group_col="regime",
        summary_type="regime",
        bootstrap_reps=bootstrap_reps,
    )
    summary = pd.concat([rho_summary, regime_summary], ignore_index=True)
    summary.to_csv(output_root / "rho_boundary_summary.csv", index=False)
    outcomes.to_csv(output_root / "rho_boundary_event_outcomes.csv", index=False)
    policy_eval.to_csv(output_root / "rho_boundary_policy_predictions.csv", index=False)

    headroom_thresholds = locate_headroom_thresholds(rho_summary, [0.001, 0.01])
    capture_threshold = locate_capture_threshold(rho_summary)
    threshold_payload = {
        "gamma": float(gamma),
        "headroom_thresholds": [
            {
                "epsilon_fraction": item.epsilon,
                "epsilon_percent": item.epsilon * 100.0,
                "rho_t": item.rho_t,
                "bin": item.label,
            }
            for item in headroom_thresholds
        ],
        "capture_threshold": {
            "criterion": "capture_percent > 50 and bootstrap_ci95_low > 0",
            "rho_t": capture_threshold.rho_t,
            "bin": capture_threshold.label,
        },
        "honest_policy": policy_summary,
    }
    save_json(threshold_payload, output_root / "rho_boundary_thresholds.json")
    _write_figdata_and_figures(output_root=output_root, rho_summary=rho_summary, thresholds=threshold_payload)
    return {
        "outcomes": outcomes,
        "policy_eval": policy_eval,
        "summary": summary,
        "thresholds": threshold_payload,
    }


def _write_figdata_and_figures(*, output_root: Path, rho_summary: pd.DataFrame, thresholds: dict[str, Any]) -> None:
    figdata = rho_summary.sort_values("rho_median").copy()
    figdata.to_csv(output_root / "figdata_headroom_vs_rho.csv", index=False)
    figdata.to_csv(output_root / "figdata_capture_vs_rho.csv", index=False)

    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.bbox": "tight",
        }
    )
    x = figdata["rho_median"].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    y = figdata["mean_oracle_headroom_percent"].astype(float).to_numpy()
    yerr = np.vstack(
        [
            y - figdata["mean_oracle_headroom_percent_ci95_low"].astype(float).to_numpy(),
            figdata["mean_oracle_headroom_percent_ci95_high"].astype(float).to_numpy() - y,
        ]
    )
    ax.errorbar(x, y, yerr=yerr, marker="o", linewidth=1.8, capsize=3, color="#2563eb")
    ax.axhline(0.1, color="#f59e0b", linestyle="--", linewidth=1.0, label="epsilon 0.1%")
    ax.axhline(1.0, color="#ef4444", linestyle="--", linewidth=1.0, label="epsilon 1%")
    for item in thresholds["headroom_thresholds"]:
        if item["rho_t"] is not None:
            ax.axvline(float(item["rho_t"]), color="#6b7280", linestyle=":", linewidth=0.9)
    ax.set_xlabel("rho_t")
    ax.set_ylabel("Oracle headroom (%)")
    ax.set_title("Oracle Headroom vs rho_t")
    ax.legend(frameon=False)
    fig.savefig(output_root / "headroom_vs_rho.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    y = figdata["capture_percent"].astype(float).to_numpy()
    y_low = figdata["capture_percent_ci95_low"].astype(float).to_numpy()
    y_high = figdata["capture_percent_ci95_high"].astype(float).to_numpy()
    yerr = np.vstack([y - y_low, y_high - y])
    ax.errorbar(x, y, yerr=yerr, marker="o", linewidth=1.8, capsize=3, color="#10b981")
    ax.axhline(0.0, color="#6b7280", linewidth=0.8)
    ax.axhline(50.0, color="#ef4444", linestyle="--", linewidth=1.0, label="50% capture")
    capture_rho = thresholds["capture_threshold"]["rho_t"]
    if capture_rho is not None:
        ax.axvline(float(capture_rho), color="#6b7280", linestyle=":", linewidth=0.9)
    ax.set_xlabel("rho_t")
    ax.set_ylabel("Honest capture (%)")
    ax.set_title("Honest Capture vs rho_t")
    ax.legend(frameon=False)
    fig.savefig(output_root / "capture_vs_rho.png")
    plt.close(fig)
