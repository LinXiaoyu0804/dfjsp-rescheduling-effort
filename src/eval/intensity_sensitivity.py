from __future__ import annotations

import json
import math
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.data.unified_parser import parse_instance
from src.env.dfjsp_env import DFJSPReschedulingEnv
from src.events.serialization import deserialize_dynamic_event
from src.scheduling.incumbent_builder import load_incumbent_schedule
from src.utils.io import ensure_dir, load_json, save_json, save_jsonl


INTENSITY_ORDER = ["L0", "L1", "L2", "L3"]
GAMMA_FACTORS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
EVENT_KEY = ["budget_sec", "scale", "disturbance", "instance_id", "seed", "episode_id", "event_id"]
FEATURE_COLUMNS = [
    "budget_sec",
    "size",
    "flexibility",
    "contention",
    "footprint_ops",
    "footprint_machines",
    "propagation_depth",
    "window_size",
    "forced_release_count",
    "event_type_arrival",
    "event_type_breakdown",
    "event_type_other",
]


@dataclass(frozen=True)
class ScaleSpec:
    label: str
    source_episodes_dir: Path
    instance_ids: tuple[str, ...]
    instance_prefix: str


def normalized_reward(
    *,
    makespan_before: float,
    tardiness_before: float,
    instability_before: float,
    makespan_after: float,
    tardiness_after: float,
    instability_after: float,
    alpha: float,
    beta: float,
    gamma: float,
) -> float:
    objective_before = alpha * makespan_before + beta * tardiness_before + gamma * instability_before
    objective_after = alpha * makespan_after + beta * tardiness_after + gamma * instability_after
    return float((objective_before - objective_after) / max(1.0, abs(objective_before)))


def choose_cheapest_best(reward_by_intensity: dict[str, float], *, tolerance: float = 1e-12) -> str:
    best_reward = max(float(reward_by_intensity[level]) for level in INTENSITY_ORDER)
    for level in INTENSITY_ORDER:
        if float(reward_by_intensity[level]) >= best_reward - tolerance:
            return level
    raise RuntimeError("unreachable intensity tie resolution")


def bootstrap_mean_ci(values: Iterable[float], *, seed: int = 0, n_boot: int = 5000) -> tuple[float, float, float]:
    arr = np.asarray([float(value) for value in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(arr.mean())
    if arr.size == 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    samples = rng.choice(arr, size=(int(n_boot), arr.size), replace=True)
    boot = samples.mean(axis=1)
    return mean, float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def paired_wilcoxon_p(a: Iterable[float], b: Iterable[float]) -> tuple[float | None, float | None, str]:
    a_arr = np.asarray([float(value) for value in a], dtype=float)
    b_arr = np.asarray([float(value) for value in b], dtype=float)
    mask = np.isfinite(a_arr) & np.isfinite(b_arr)
    a_arr = a_arr[mask]
    b_arr = b_arr[mask]
    if a_arr.size < 2:
        return None, None, "insufficient_pairs"
    if np.allclose(a_arr - b_arr, 0.0):
        return 0.0, 1.0, "identical_pairs"
    stat, p_value = wilcoxon(a_arr, b_arr, zero_method="pratt", alternative="two-sided")
    return float(stat), float(p_value), ""


def tune_upgrade_threshold(
    predicted_gain: np.ndarray,
    predicted_level_idx: np.ndarray,
    reward_matrix: np.ndarray,
) -> tuple[float, float, float]:
    if predicted_gain.ndim != 1 or predicted_level_idx.ndim != 1:
        raise ValueError("predicted_gain and predicted_level_idx must be 1-D arrays")
    if reward_matrix.shape[0] != predicted_gain.shape[0] or reward_matrix.shape[1] != len(INTENSITY_ORDER):
        raise ValueError("reward_matrix shape is inconsistent with intensity order")
    candidates = [float("-inf"), float("inf")]
    candidates.extend(float(value) for value in np.unique(predicted_gain[np.isfinite(predicted_gain)]))
    candidates.append(0.0)
    best: tuple[float, float, float] | None = None
    for threshold in sorted(set(candidates)):
        selected_idx = np.where(predicted_gain > threshold, predicted_level_idx, 0)
        selected_reward = reward_matrix[np.arange(reward_matrix.shape[0]), selected_idx]
        mean_reward = float(np.mean(selected_reward)) if selected_reward.size else float("-inf")
        upgrade_rate = float(np.mean(selected_idx != 0)) if selected_idx.size else 0.0
        # Conservative tie break: same train reward prefers fewer upgrades, then higher threshold.
        score = (mean_reward, -upgrade_rate, threshold)
        if best is None or score > (best[1], -best[2], best[0]):
            best = (float(threshold), mean_reward, upgrade_rate)
    if best is None:
        raise RuntimeError("No threshold candidates were evaluated.")
    return best


def _resolve_path(path_like: str | Path, *, root: Path) -> Path:
    # Normalize Windows-style separators so episode paths stay cross-platform.
    path = Path(str(path_like).replace("\\", "/"))
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def _event_type_bucket(event_type: str, event_id: str) -> str:
    event_type = str(event_type).lower()
    event_id = str(event_id).lower()
    if "arrival" in event_type or event_id.startswith("arr_"):
        return "arrival"
    if "breakdown" in event_type or event_id.startswith("bd_"):
        return "breakdown"
    return "other"


def _filter_events(events: list[dict[str, Any]], disturbance: str) -> list[dict[str, Any]]:
    if disturbance == "mixed":
        return list(events)
    if disturbance == "arrival_only":
        return [event for event in events if str(event.get("type")) == "job_arrival"]
    if disturbance == "breakdown_only":
        return [event for event in events if str(event.get("type")) == "machine_breakdown"]
    raise ValueError(f"Unknown disturbance regime: {disturbance}")


def _episode_matches_scale(episode_data: dict[str, Any], scale: ScaleSpec) -> bool:
    instance_id = str(episode_data.get("instance_id", ""))
    if scale.instance_ids:
        return instance_id in set(scale.instance_ids)
    if scale.instance_prefix:
        return instance_id.startswith(scale.instance_prefix)
    return True


def _iter_episode_payloads(scale: ScaleSpec) -> Iterable[tuple[Path, dict[str, Any]]]:
    if not scale.source_episodes_dir.exists():
        raise FileNotFoundError(f"Frozen episode directory does not exist: {scale.source_episodes_dir}")
    for path in sorted(scale.source_episodes_dir.glob("*.json")):
        data = load_json(path)
        if not isinstance(data, dict) or "episode_id" not in data or "events" not in data:
            continue
        if _episode_matches_scale(data, scale):
            yield path, data


def _load_source_records(input_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(input_root.glob("*_event_metrics.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row["_source_file"] = path.name
                row["intensity"] = str(row.get("intensity_level") or row.get("method"))
                if row["intensity"] in INTENSITY_ORDER:
                    records.append(row)
    if not records:
        raise FileNotFoundError(f"No intensity event-metric rows found under {input_root}")
    return records


def _load_manifest(input_root: Path) -> dict[str, Any]:
    manifest_path = input_root / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing intensity-grid manifest: {manifest_path.resolve()}")
    return load_json(manifest_path)


def _load_effective_config(input_root: Path) -> dict[str, Any]:
    config_paths = sorted((input_root / "_configs").glob("*.json"))
    if not config_paths:
        raise FileNotFoundError(f"Missing intensity-grid configs under {(input_root / '_configs').resolve()}")
    data = load_json(config_paths[0])
    cfg = data.get("effective_config")
    if not isinstance(cfg, dict):
        raise ValueError(f"Config file lacks effective_config: {config_paths[0].resolve()}")
    return cfg


def _scale_specs_from_manifest(manifest: dict[str, Any], *, repo_root: Path) -> dict[str, ScaleSpec]:
    specs: dict[str, ScaleSpec] = {}
    for item in manifest.get("scales", []):
        label = str(item["label"])
        specs[label] = ScaleSpec(
            label=label,
            source_episodes_dir=_resolve_path(item["source_episodes_dir"], root=repo_root),
            instance_ids=tuple(str(value) for value in item.get("instance_ids", [])),
            instance_prefix=str(item.get("instance_prefix", "")),
        )
    if not specs:
        raise ValueError("No scale specs were found in intensity-grid manifest.")
    return specs


def build_before_component_lookup(
    *,
    records: list[dict[str, Any]],
    manifest: dict[str, Any],
    cfg: dict[str, Any],
    repo_root: Path,
) -> dict[tuple[str, str, str, int, str, str], dict[str, Any]]:
    scale_specs = _scale_specs_from_manifest(manifest, repo_root=repo_root)
    needed = {
        (str(row["scale"]), str(row["disturbance"]))
        for row in records
        if str(row.get("scale")) in scale_specs and str(row.get("disturbance"))
    }
    due_factor = float(cfg.get("data", {}).get("due_date_rule", {}).get("factor", 1.5))
    family = str(cfg.get("data", {}).get("family", "fjsp"))
    lookup: dict[tuple[str, str, str, int, str, str], dict[str, Any]] = {}

    for scale_label, disturbance in sorted(needed):
        scale = scale_specs[scale_label]
        for _, episode_data in _iter_episode_payloads(scale):
            instance_path = _resolve_path(episode_data["instance_path"], root=repo_root)
            incumbent_path = _resolve_path(episode_data["incumbent_ref"], root=repo_root)
            if not instance_path.exists():
                raise FileNotFoundError(f"Instance path does not exist: {instance_path}")
            if not incumbent_path.exists():
                raise FileNotFoundError(f"Incumbent path does not exist: {incumbent_path}")
            instance = parse_instance(instance_path, family=family, due_date_factor=due_factor)
            env = DFJSPReschedulingEnv(instance, cfg)
            env.reset()
            incumbent_data = load_json(incumbent_path)
            env.incumbent = load_incumbent_schedule(instance, incumbent_data)
            env.initial_instance = deepcopy(instance)
            env.instance = deepcopy(instance)

            selected_events = _filter_events(list(episode_data.get("events", [])), disturbance)
            for fallback_id, event_data in enumerate(selected_events):
                event = deserialize_dynamic_event(event_data, fallback_event_id=fallback_id)
                env.apply_event(event)
                env.build_window()
                objective = env.compute_objective()
                key = (
                    scale_label,
                    disturbance,
                    str(episode_data["instance_id"]),
                    int(episode_data["seed"]),
                    str(episode_data["episode_id"]),
                    str(event_data["event_id"]),
                )
                lookup[key] = {
                    "makespan_before": float(objective.makespan),
                    "tardiness_before": float(objective.total_tardiness),
                    "instability_before": 0.0,
                    "incumbent_instability_raw_before": float(objective.instability),
                    "objective_before_replay_weighted": float(objective.weighted_sum),
                    "before_component_source": "cached_incumbent_replay_no_repair",
                    "before_component_source_instance_path": str(instance_path),
                    "before_component_source_incumbent_path": str(incumbent_path),
                    "size": int(env.instance.num_operations),
                }
    return lookup


def augment_records_with_before_components(
    records: list[dict[str, Any]],
    lookup: dict[tuple[str, str, str, int, str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    augmented: list[dict[str, Any]] = []
    missing: list[tuple[str, str, str, int, str, str]] = []
    for row in records:
        key = (
            str(row["scale"]),
            str(row["disturbance"]),
            str(row["instance_id"]),
            int(row["seed"]),
            str(row["episode_id"]),
            str(row["event_id"]),
        )
        before = lookup.get(key)
        if before is None:
            missing.append(key)
            continue
        out = dict(row)
        out.update(before)
        out["objective_before_existing_raw"] = float(row.get("objective_before", float("nan")))
        out["reward_delta_existing_raw"] = float(row.get("reward_delta", float("nan")))
        augmented.append(out)
    if missing:
        preview = ", ".join(str(key) for key in missing[:5])
        raise KeyError(f"Missing recomputed before components for {len(missing)} rows; first keys: {preview}")
    return augmented


def write_decomposition_outputs(
    *,
    augmented: list[dict[str, Any]],
    decomp_root: Path,
    input_root: Path,
    manifest: dict[str, Any],
    cfg: dict[str, Any],
) -> pd.DataFrame:
    ensure_dir(decomp_root)
    by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in augmented:
        by_file[str(row["_source_file"])].append({key: value for key, value in row.items() if key != "_source_file"})
    for filename, rows in sorted(by_file.items()):
        save_jsonl(rows, decomp_root / filename)

    frame = pd.DataFrame(augmented)
    compact_cols = [
        "intensity",
        "budget_sec",
        "scale",
        "disturbance",
        "instance_id",
        "seed",
        "episode_id",
        "event_id",
        "event_type_bucket",
        "size",
        "makespan_before",
        "tardiness_before",
        "instability_before",
        "makespan_after",
        "tardiness_after",
        "instability_after",
        "objective_before_existing_raw",
        "objective_before_replay_weighted",
        "reward_delta_existing_raw",
        "status",
        "before_component_source",
    ]
    frame[[col for col in compact_cols if col in frame.columns]].to_csv(
        decomp_root / "event_components_compact.csv", index=False
    )

    validation_rows = []
    l0 = frame[frame["intensity"] == "L0"].copy()
    if not l0.empty:
        l0["objective_before_replay_abs_diff"] = (
            l0["objective_before_existing_raw"].astype(float) - l0["objective_before_replay_weighted"].astype(float)
        ).abs()
        validation = (
            l0.groupby(["scale", "disturbance", "budget_sec"], as_index=False)
            .agg(
                n=("objective_before_replay_abs_diff", "size"),
                max_abs_diff=("objective_before_replay_abs_diff", "max"),
                mean_abs_diff=("objective_before_replay_abs_diff", "mean"),
            )
            .sort_values(["scale", "disturbance", "budget_sec"])
        )
        validation_rows = validation.to_dict(orient="records")
        validation.to_csv(decomp_root / "replay_validation.csv", index=False)

    save_json(
        {
            "status": "completed",
            "source_intensity_grid": str(input_root.resolve()),
            "output_root": str(decomp_root.resolve()),
            "before_component_source": "cached_incumbent_replay_no_repair",
            "replayed_without_cp_sat_repair": True,
            "instability_before_policy": "set_to_zero_for_relative_incumbent_reward",
            "existing_reward_delta_definition": "raw objective_before.weighted_sum - weighted_objective_after",
            "sensitivity_reward_definition": "(J_before - J_after) / max(1, abs(J_before))",
            "env_objective_weights": cfg.get("env", {}).get("objective_weights", {}),
            "source_manifest_status": manifest.get("status"),
            "validation": validation_rows,
        },
        decomp_root / "run_manifest.json",
    )
    return frame


def _prepare_event_features(pivot: pd.DataFrame) -> pd.DataFrame:
    event_rows = pivot[pivot["intensity"] == "L0"].copy()
    if event_rows.empty:
        raise ValueError("No L0 rows are available for event features.")
    event_rows["footprint_ops"] = pd.to_numeric(event_rows["event_footprint_ops"], errors="coerce").fillna(0.0)
    event_rows["footprint_machines"] = pd.to_numeric(event_rows["event_footprint_machines"], errors="coerce").fillna(0.0)
    event_type = event_rows["event_type_bucket"].fillna("other").astype(str).str.lower()
    event_rows["event_type_arrival"] = (event_type == "arrival").astype(float)
    event_rows["event_type_breakdown"] = (event_type == "breakdown").astype(float)
    event_rows["event_type_other"] = (~event_type.isin(["arrival", "breakdown"])).astype(float)
    for col in FEATURE_COLUMNS:
        if col not in event_rows.columns:
            event_rows[col] = 0.0
        event_rows[col] = pd.to_numeric(event_rows[col], errors="coerce").fillna(0.0)
    return event_rows


def _reward_pivot(frame: pd.DataFrame) -> pd.DataFrame:
    pivot = frame.pivot_table(index=EVENT_KEY, columns="intensity", values="reward_reweighted", aggfunc="first")
    missing_cols = [level for level in INTENSITY_ORDER if level not in pivot.columns]
    if missing_cols:
        raise ValueError(f"Missing intensity columns for reward pivot: {missing_cols}")
    pivot = pivot.dropna(subset=INTENSITY_ORDER).reset_index()
    return pivot


def _prediction_best(predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    best_idx = []
    best_gain = []
    for row in predictions:
        best_col = int(np.argmax(row))
        best_idx.append(best_col + 1)
        best_gain.append(float(row[best_col]))
    return np.asarray(best_idx, dtype=int), np.asarray(best_gain, dtype=float)


def _fit_gain_models(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_predict: np.ndarray,
    random_state: int,
) -> tuple[list[tuple[str, Any]], np.ndarray]:
    from sklearn.ensemble import GradientBoostingRegressor

    models = []
    pred = []
    for idx, level in enumerate(INTENSITY_ORDER[1:]):
        model = GradientBoostingRegressor(random_state=random_state)
        model.fit(x_train, y_train[:, idx])
        models.append((level, model))
        pred.append(model.predict(x_predict))
    return models, np.vstack(pred).T


def _select_rewards(reward_matrix: np.ndarray, selected_idx: np.ndarray) -> np.ndarray:
    return reward_matrix[np.arange(reward_matrix.shape[0]), selected_idx]


def evaluate_honest_policy(
    *,
    gamma_frame: pd.DataFrame,
    train_seed_parity: int = 0,
    random_state: int = 0,
) -> tuple[dict[str, Any], pd.DataFrame]:
    rewards = _reward_pivot(gamma_frame)
    features = _prepare_event_features(gamma_frame)
    dataset = features.merge(rewards, on=EVENT_KEY, how="inner", validate="one_to_one")
    if dataset.empty:
        raise ValueError("No matched event rows are available for policy evaluation.")

    reward_matrix = dataset[INTENSITY_ORDER].to_numpy(dtype=float)
    x_all = dataset[FEATURE_COLUMNS].to_numpy(dtype=float)
    train_mask = (dataset["seed"].astype(int) % 2).to_numpy() == int(train_seed_parity)
    test_mask = ~train_mask
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError("Seed-parity split produced an empty train or test partition.")

    y_train = reward_matrix[train_mask, 1:] - reward_matrix[train_mask, 0:1]
    _, predictions = _fit_gain_models(
        x_train=x_all[train_mask],
        y_train=y_train,
        x_predict=x_all,
        random_state=random_state,
    )
    best_idx, best_gain = _prediction_best(predictions)

    train_indices = np.flatnonzero(train_mask)
    train_groups = dataset.loc[train_mask, "seed"].astype(int).to_numpy()
    unique_groups = np.unique(train_groups)
    oof_predictions = np.full((train_indices.shape[0], len(INTENSITY_ORDER) - 1), np.nan, dtype=float)
    if unique_groups.shape[0] >= 2:
        for group in unique_groups:
            fold_train_mask = train_mask & (dataset["seed"].astype(int).to_numpy() != int(group))
            fold_val_mask = train_mask & (dataset["seed"].astype(int).to_numpy() == int(group))
            if fold_train_mask.sum() == 0 or fold_val_mask.sum() == 0:
                continue
            _, fold_pred = _fit_gain_models(
                x_train=x_all[fold_train_mask],
                y_train=reward_matrix[fold_train_mask, 1:] - reward_matrix[fold_train_mask, 0:1],
                x_predict=x_all[fold_val_mask],
                random_state=random_state,
            )
            local_positions = np.flatnonzero(fold_val_mask)[
                np.isin(np.flatnonzero(fold_val_mask), train_indices)
            ]
            for global_idx, pred_row in zip(local_positions, fold_pred):
                oof_predictions[np.where(train_indices == global_idx)[0][0], :] = pred_row
    if not np.isfinite(oof_predictions).all():
        oof_predictions = predictions[train_mask]
    threshold_best_idx, threshold_best_gain = _prediction_best(oof_predictions)

    threshold, train_reward, train_upgrade_rate = tune_upgrade_threshold(
        predicted_gain=threshold_best_gain,
        predicted_level_idx=threshold_best_idx,
        reward_matrix=reward_matrix[train_mask],
    )
    selected_idx = np.where(best_gain > threshold, best_idx, 0)
    policy_rewards = _select_rewards(reward_matrix, selected_idx)

    oracle_idx = []
    for row in reward_matrix:
        reward_by_intensity = {level: float(row[pos]) for pos, level in enumerate(INTENSITY_ORDER)}
        oracle_idx.append(INTENSITY_ORDER.index(choose_cheapest_best(reward_by_intensity)))
    oracle_idx_arr = np.asarray(oracle_idx, dtype=int)
    oracle_rewards = _select_rewards(reward_matrix, oracle_idx_arr)

    event_eval_cols = list(dict.fromkeys(EVENT_KEY + ["event_type_bucket"] + FEATURE_COLUMNS))
    event_eval = dataset[event_eval_cols].copy()
    event_eval["split"] = np.where(train_mask, "train", "test")
    event_eval["always_l0_reward"] = reward_matrix[:, 0]
    event_eval["policy_reward"] = policy_rewards
    event_eval["oracle_reward"] = oracle_rewards
    event_eval["policy_selected_intensity"] = [INTENSITY_ORDER[idx] for idx in selected_idx]
    event_eval["oracle_selected_intensity"] = [INTENSITY_ORDER[idx] for idx in oracle_idx_arr]
    event_eval["policy_predicted_gain"] = best_gain
    event_eval["policy_threshold"] = threshold

    test = event_eval[event_eval["split"] == "test"]
    l0_mean = float(test["always_l0_reward"].mean())
    policy_mean = float(test["policy_reward"].mean())
    oracle_mean = float(test["oracle_reward"].mean())
    denom = oracle_mean - l0_mean
    capture = float((policy_mean - l0_mean) / denom) if abs(denom) > 1e-12 else float("nan")
    summary = {
        "train_seed_parity": int(train_seed_parity),
        "test_seed_parity": int(1 - train_seed_parity),
        "n_train_events": int(train_mask.sum()),
        "n_test_events": int(test_mask.sum()),
        "threshold": float(threshold),
        "train_policy_reward": float(train_reward),
        "train_upgrade_rate": float(train_upgrade_rate),
        "test_always_l0_reward": l0_mean,
        "test_policy_reward": policy_mean,
        "test_oracle_reward": oracle_mean,
        "policy_minus_l0": float(policy_mean - l0_mean),
        "oracle_minus_l0": float(denom),
        "capture_fraction": capture,
        "capture_percent": capture * 100.0 if math.isfinite(capture) else float("nan"),
        "test_policy_upgrade_rate": float((test["policy_selected_intensity"] != "L0").mean()),
    }
    return summary, event_eval


def _gamma_frame(base: pd.DataFrame, *, alpha: float, beta: float, gamma: float) -> pd.DataFrame:
    frame = base.copy()
    frame["reward_reweighted"] = [
        normalized_reward(
            makespan_before=float(row.makespan_before),
            tardiness_before=float(row.tardiness_before),
            instability_before=float(row.instability_before),
            makespan_after=float(row.makespan_after),
            tardiness_after=float(row.tardiness_after),
            instability_after=float(row.instability_after),
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )
        for row in frame.itertuples(index=False)
    ]
    frame["weighted_objective_before_reweighted"] = (
        alpha * frame["makespan_before"].astype(float)
        + beta * frame["tardiness_before"].astype(float)
        + gamma * frame["instability_before"].astype(float)
    )
    frame["weighted_objective_after_reweighted"] = (
        alpha * frame["makespan_after"].astype(float)
        + beta * frame["tardiness_after"].astype(float)
        + gamma * frame["instability_after"].astype(float)
    )
    return frame


def _capture_crossing(summary: pd.DataFrame) -> tuple[float | None, float | None]:
    ordered = summary.sort_values("gamma").reset_index(drop=True)
    prev = None
    for row in ordered.itertuples(index=False):
        value = float(row.capture_fraction)
        if not math.isfinite(value):
            prev = row
            continue
        if value == 0.0:
            return float(row.gamma), float(row.gamma_factor)
        if prev is not None:
            prev_value = float(prev.capture_fraction)
            if math.isfinite(prev_value) and prev_value < 0.0 <= value:
                g0, g1 = float(prev.gamma), float(row.gamma)
                f0, f1 = float(prev.gamma_factor), float(row.gamma_factor)
                if value == prev_value:
                    return g1, f1
                ratio = (0.0 - prev_value) / (value - prev_value)
                return g0 + ratio * (g1 - g0), f0 + ratio * (f1 - f0)
        prev = row
    return None, None


def run_sensitivity_analysis(
    *,
    augmented: list[dict[str, Any]],
    output_root: Path,
    alpha: float,
    beta: float,
    baseline_gamma: float,
    train_seed_parity: int = 0,
    bootstrap_reps: int = 5000,
) -> dict[str, Any]:
    ensure_dir(output_root)
    base = pd.DataFrame(augmented)
    base = base[base["intensity"].isin(INTENSITY_ORDER)].copy()
    for col in [
        "budget_sec",
        "seed",
        "size",
        "makespan_before",
        "tardiness_before",
        "instability_before",
        "makespan_after",
        "tardiness_after",
        "instability_after",
        "flexibility",
        "contention",
        "propagation_depth",
        "window_size",
        "forced_release_count",
        "event_footprint_ops",
        "event_footprint_machines",
    ]:
        base[col] = pd.to_numeric(base[col], errors="coerce")

    summary_rows: list[dict[str, Any]] = []
    event_eval_by_gamma: dict[float, pd.DataFrame] = {}
    baseline_gamma_frame: pd.DataFrame | None = None

    for factor in GAMMA_FACTORS:
        gamma = baseline_gamma * factor
        gamma_data = _gamma_frame(base, alpha=alpha, beta=beta, gamma=gamma)
        if abs(factor - 1.0) < 1e-12:
            baseline_gamma_frame = gamma_data.copy()
        summary, event_eval = evaluate_honest_policy(
            gamma_frame=gamma_data,
            train_seed_parity=train_seed_parity,
            random_state=0,
        )
        summary.update(
            {
                "gamma_factor": float(factor),
                "gamma": float(gamma),
                "alpha": float(alpha),
                "beta": float(beta),
                "baseline_gamma": float(baseline_gamma),
            }
        )
        summary_rows.append(summary)
        event_eval["gamma_factor"] = float(factor)
        event_eval["gamma"] = float(gamma)
        event_eval_by_gamma[float(factor)] = event_eval

    summary_df = pd.DataFrame(summary_rows).sort_values("gamma_factor")
    crossing_gamma, crossing_factor = _capture_crossing(summary_df)
    summary_df["capture_negative_to_positive_threshold_gamma"] = crossing_gamma
    summary_df["capture_negative_to_positive_threshold_factor"] = crossing_factor
    summary_df.to_csv(output_root / "sensitivity_summary.csv", index=False)

    all_events = pd.concat(event_eval_by_gamma.values(), ignore_index=True)
    all_events.to_csv(output_root / "sensitivity_event_predictions.csv", index=False)

    if baseline_gamma_frame is None:
        raise RuntimeError("Baseline gamma factor 1.0 was not evaluated.")
    baseline_events = event_eval_by_gamma[1.0]
    baseline_test = baseline_events[baseline_events["split"] == "test"].copy()
    _write_stat_tests(
        output_root=output_root,
        baseline_test=baseline_test,
        baseline_gamma=baseline_gamma,
        bootstrap_reps=bootstrap_reps,
    )
    figdata = _write_figdata(
        output_root=output_root,
        baseline_gamma_frame=baseline_gamma_frame,
        baseline_events=baseline_events,
        summary_df=summary_df,
        bootstrap_reps=bootstrap_reps,
    )
    _write_figures(output_root=output_root, figdata=figdata, summary_df=summary_df)
    return {
        "summary": summary_df,
        "baseline_events": baseline_events,
        "baseline_gamma_frame": baseline_gamma_frame,
        "capture_crossing_gamma": crossing_gamma,
        "capture_crossing_factor": crossing_factor,
    }


def _write_stat_tests(
    *,
    output_root: Path,
    baseline_test: pd.DataFrame,
    baseline_gamma: float,
    bootstrap_reps: int,
) -> None:
    comparisons = [
        ("Policy - Always-L0", "policy_reward", "always_l0_reward"),
        ("Policy - Oracle", "policy_reward", "oracle_reward"),
    ]
    lines = [
        "# Baseline Gamma Statistical Tests",
        "",
        f"Baseline gamma: `{baseline_gamma:.12g}`.",
        "",
        "| comparison | n | mean paired diff | 95% bootstrap CI | Wilcoxon p | note |",
        "|---|---:|---:|---:|---:|---|",
    ]
    rows = []
    for label, a_col, b_col in comparisons:
        diff = baseline_test[a_col].astype(float).to_numpy() - baseline_test[b_col].astype(float).to_numpy()
        mean, low, high = bootstrap_mean_ci(diff, seed=17, n_boot=bootstrap_reps)
        stat, p_value, note = paired_wilcoxon_p(baseline_test[a_col], baseline_test[b_col])
        rows.append(
            {
                "comparison": label,
                "n": int(len(diff)),
                "mean_paired_diff": mean,
                "ci95_low": low,
                "ci95_high": high,
                "wilcoxon_statistic": stat,
                "wilcoxon_p": p_value,
                "note": note,
            }
        )
        p_text = "" if p_value is None else f"{p_value:.6g}"
        lines.append(
            f"| {label} | {len(diff)} | {mean:.8f} | [{low:.8f}, {high:.8f}] | {p_text} | {note} |"
        )
    pd.DataFrame(rows).to_csv(output_root / "stat_tests.csv", index=False)
    lines.extend(
        [
            "",
            "The paired bootstrap resamples test events with replacement. Wilcoxon is two-sided with Pratt zero handling.",
        ]
    )
    (output_root / "stat_tests.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_figdata(
    *,
    output_root: Path,
    baseline_gamma_frame: pd.DataFrame,
    baseline_events: pd.DataFrame,
    summary_df: pd.DataFrame,
    bootstrap_reps: int,
) -> dict[str, pd.DataFrame]:
    f1_rows = []
    for (budget, intensity), subset in baseline_gamma_frame.groupby(["budget_sec", "intensity"], sort=True):
        mean, low, high = bootstrap_mean_ci(subset["reward_reweighted"], seed=101, n_boot=bootstrap_reps)
        f1_rows.append(
            {
                "budget_sec": float(budget),
                "intensity": intensity,
                "intensity_order": INTENSITY_ORDER.index(str(intensity)),
                "mean_reward": mean,
                "ci95_low": low,
                "ci95_high": high,
                "n": int(len(subset)),
            }
        )
    f1 = pd.DataFrame(f1_rows).sort_values(["budget_sec", "intensity_order"])
    f1.to_csv(output_root / "figdata_F1_intensity_quality_frontier.csv", index=False)

    budget_fixed = 5.0
    f2_base = baseline_gamma_frame[np.isclose(baseline_gamma_frame["budget_sec"].astype(float), budget_fixed)].copy()
    f2_means = (
        f2_base.groupby(["scale", "disturbance", "intensity"], as_index=False)["reward_reweighted"]
        .mean()
        .rename(columns={"reward_reweighted": "mean_reward"})
    )
    f2_rows = []
    for (scale, disturbance), subset in f2_means.groupby(["scale", "disturbance"], sort=True):
        rewards = {row["intensity"]: float(row["mean_reward"]) for _, row in subset.iterrows()}
        for level in INTENSITY_ORDER:
            rewards.setdefault(level, float("-inf"))
        best = choose_cheapest_best(rewards)
        row = {
            "budget_sec": budget_fixed,
            "scale": scale,
            "disturbance": disturbance,
            "best_intensity": best,
            "best_intensity_order": INTENSITY_ORDER.index(best),
        }
        row.update({f"mean_reward_{level}": rewards[level] for level in INTENSITY_ORDER})
        f2_rows.append(row)
    f2 = pd.DataFrame(f2_rows).sort_values(["scale", "disturbance"])
    f2.to_csv(output_root / "figdata_F2_heterogeneity_heatmap.csv", index=False)

    baseline_test = baseline_events[baseline_events["split"] == "test"].copy()
    f3_rows = []
    method_cols = [
        ("Always-L0", "always_l0_reward"),
        ("Policy", "policy_reward"),
        ("Oracle", "oracle_reward"),
    ]
    for (budget, method), subset in [
        (key, group)
        for budget, group_budget in baseline_test.groupby("budget_sec", sort=True)
        for key, group in [((budget, label), group_budget) for label, _ in method_cols]
    ]:
        col = dict(method_cols)[method]
        mean, low, high = bootstrap_mean_ci(subset[col], seed=202, n_boot=bootstrap_reps)
        f3_rows.append(
            {
                "budget_sec": float(budget),
                "method": method,
                "mean_reward": mean,
                "ci95_low": low,
                "ci95_high": high,
                "n": int(len(subset)),
            }
        )
    f3 = pd.DataFrame(f3_rows)
    method_order = {label: idx for idx, (label, _) in enumerate(method_cols)}
    f3["method_order"] = f3["method"].map(method_order)
    f3 = f3.sort_values(["budget_sec", "method_order"])
    f3.to_csv(output_root / "figdata_F3_value_gap_bars.csv", index=False)

    f4 = summary_df[
        [
            "gamma_factor",
            "gamma",
            "test_always_l0_reward",
            "test_policy_reward",
            "test_oracle_reward",
            "capture_fraction",
            "capture_percent",
            "capture_negative_to_positive_threshold_gamma",
            "capture_negative_to_positive_threshold_factor",
        ]
    ].copy()
    f4.to_csv(output_root / "figdata_F4_gamma_sensitivity.csv", index=False)
    return {"F1": f1, "F2": f2, "F3": f3, "F4": f4}


def _write_figures(*, output_root: Path, figdata: dict[str, pd.DataFrame], summary_df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.bbox": "tight",
        }
    )
    palette = ["#2563eb", "#10b981", "#f59e0b", "#7c3aed", "#ef4444"]

    f1 = figdata["F1"]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for idx, (budget, subset) in enumerate(f1.groupby("budget_sec", sort=True)):
        subset = subset.sort_values("intensity_order")
        yerr = np.vstack(
            [
                subset["mean_reward"].to_numpy() - subset["ci95_low"].to_numpy(),
                subset["ci95_high"].to_numpy() - subset["mean_reward"].to_numpy(),
            ]
        )
        ax.errorbar(
            subset["intensity_order"],
            subset["mean_reward"],
            yerr=yerr,
            marker="o",
            capsize=3,
            linewidth=1.8,
            label=f"{budget:g}s",
            color=palette[idx % len(palette)],
        )
    ax.axhline(0.0, color="#6b7280", linewidth=0.8)
    ax.set_xticks(range(len(INTENSITY_ORDER)), INTENSITY_ORDER)
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Mean normalized reward")
    ax.set_title("F1 Intensity-Quality Frontier")
    ax.legend(title="Budget", frameon=False, ncols=3)
    _save_figure(fig, output_root / "F1_intensity_quality_frontier")

    f2 = figdata["F2"]
    scales = list(dict.fromkeys(f2["scale"].tolist()))
    disturbances = ["arrival_only", "breakdown_only", "mixed"]
    matrix = np.full((len(scales), len(disturbances)), np.nan)
    labels = [["" for _ in disturbances] for _ in scales]
    for row in f2.itertuples(index=False):
        if row.disturbance not in disturbances:
            continue
        i = scales.index(row.scale)
        j = disturbances.index(row.disturbance)
        matrix[i, j] = float(row.best_intensity_order)
        labels[i][j] = str(row.best_intensity)
    fig, ax = plt.subplots(figsize=(6.7, 4.8))
    cmap = ListedColormap(["#d1d5db", "#60a5fa", "#34d399", "#fbbf24"])
    im = ax.imshow(matrix, vmin=0, vmax=3, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(disturbances)), ["arrival", "breakdown", "mixed"])
    ax.set_yticks(range(len(scales)), scales)
    ax.set_xlabel("Disturbance")
    ax.set_ylabel("Scale")
    ax.set_title("F2 Best Fixed Intensity at 5s")
    for i in range(len(scales)):
        for j in range(len(disturbances)):
            ax.text(j, i, labels[i][j], ha="center", va="center", color="#111827", fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, ticks=range(4))
    cbar.ax.set_yticklabels(INTENSITY_ORDER)
    _save_figure(fig, output_root / "F2_heterogeneity_heatmap")

    f3 = figdata["F3"]
    budgets = sorted(f3["budget_sec"].unique())
    methods = ["Always-L0", "Policy", "Oracle"]
    x = np.arange(len(budgets))
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    method_colors = {"Always-L0": "#6b7280", "Policy": "#2563eb", "Oracle": "#10b981"}
    for idx, method in enumerate(methods):
        subset = f3[f3["method"] == method].set_index("budget_sec").loc[budgets]
        positions = x + (idx - 1) * width
        yerr = np.vstack(
            [
                subset["mean_reward"].to_numpy() - subset["ci95_low"].to_numpy(),
                subset["ci95_high"].to_numpy() - subset["mean_reward"].to_numpy(),
            ]
        )
        ax.bar(
            positions,
            subset["mean_reward"],
            width=width,
            label=method,
            color=method_colors[method],
            yerr=yerr,
            capsize=3,
            linewidth=0,
        )
    ax.axhline(0.0, color="#6b7280", linewidth=0.8)
    ax.set_xticks(x, [f"{budget:g}s" for budget in budgets])
    ax.set_xlabel("Budget")
    ax.set_ylabel("Mean normalized reward")
    ax.set_title("F3 Realizable Value Gap")
    ax.legend(frameon=False)
    _save_figure(fig, output_root / "F3_value_gap_bars")

    f4 = figdata["F4"].sort_values("gamma")
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(f4["gamma"], f4["capture_percent"], marker="o", linewidth=1.9, color="#7c3aed")
    ax.axhline(0.0, color="#6b7280", linewidth=0.8)
    crossing = summary_df["capture_negative_to_positive_threshold_gamma"].dropna()
    if not crossing.empty:
        threshold = float(crossing.iloc[0])
        ax.axvline(threshold, color="#ef4444", linestyle="--", linewidth=1.2)
        ax.text(threshold, ax.get_ylim()[1], f" threshold {threshold:.3g}", va="top", ha="left", color="#ef4444")
    ax.set_xlabel("Instability weight gamma")
    ax.set_ylabel("Capture (%)")
    ax.set_title("F4 Gamma Sensitivity")
    _save_figure(fig, output_root / "F4_gamma_sensitivity")
    plt.close("all")


def _save_figure(fig: Any, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"))
    fig.clf()


def write_readmes(
    *,
    decomp_root: Path,
    sensitivity_root: Path,
    source_root: Path,
    alpha: float,
    beta: float,
    baseline_gamma: float,
    crossing_gamma: float | None,
    crossing_factor: float | None,
) -> None:
    decomp_readme = f"""# Intensity Grid Decomposition

Source: `{source_root.resolve()}`.

Before components were recomputed from cached `incumbent_ref` schedules and frozen episode JSON traces by replaying `DFJSPReschedulingEnv.apply_event()` / `build_window()` only. No CP-SAT repair was invoked, and no intensity-grid cell was re-solved.

Fields added to each event row:

- `makespan_before`
- `tardiness_before`
- `instability_before` fixed to `0.0` for relative-to-incumbent reward reweighting
- `incumbent_instability_raw_before` retained only for auditing the original environment objective
- `objective_before_replay_weighted` using the original environment weights
- `size` as the post-event operation count

The existing source `reward_delta` is raw weighted-objective difference, not normalized: `objective_before.weighted_sum - weighted_objective_after`. The sensitivity outputs use `(J_before - J_after) / max(1, abs(J_before))`, with `J = alpha*Cmax + beta*sumT + gamma*I`.
"""
    (decomp_root / "README.md").write_text(decomp_readme, encoding="utf-8")

    threshold_text = (
        "not observed"
        if crossing_gamma is None
        else f"gamma={crossing_gamma:.12g} (factor={crossing_factor:.12g})"
    )
    sensitivity_readme = f"""# Objective Weight Sensitivity

Source decomposition: `{decomp_root.resolve()}`.

Baseline weights: `alpha={alpha:.12g}`, `beta={beta:.12g}`, `gamma={baseline_gamma:.12g}`. Gamma factors scanned: `{GAMMA_FACTORS}`.

Honest policy split: even seeds for training, odd seeds for testing. The upgrade threshold is selected from leave-one-training-seed-out predictions only; test rows are not used for fitting or threshold tuning.

Capture negative-to-positive threshold: {threshold_text}.

Main artifacts:

- `sensitivity_summary.csv`
- `stat_tests.md` and `stat_tests.csv`
- `F1_intensity_quality_frontier.svg/png`
- `F2_heterogeneity_heatmap.svg/png`
- `F3_value_gap_bars.svg/png`
- `F4_gamma_sensitivity.svg/png`
- `figdata_F1_intensity_quality_frontier.csv`
- `figdata_F2_heterogeneity_heatmap.csv`
- `figdata_F3_value_gap_bars.csv`
- `figdata_F4_gamma_sensitivity.csv`
"""
    (sensitivity_root / "README.md").write_text(sensitivity_readme, encoding="utf-8")


def run_full_pipeline(
    *,
    input_root: Path,
    decomp_root: Path,
    sensitivity_root: Path,
    repo_root: Path,
    train_seed_parity: int = 0,
    bootstrap_reps: int = 5000,
) -> dict[str, Any]:
    input_root = input_root.resolve()
    decomp_root = decomp_root.resolve()
    sensitivity_root = sensitivity_root.resolve()
    records = _load_source_records(input_root)
    manifest = _load_manifest(input_root)
    cfg = _load_effective_config(input_root)
    weights = cfg.get("env", {}).get("objective_weights", {})
    alpha = float(weights.get("makespan", 1.0))
    beta = float(weights.get("tardiness", 1.0))
    baseline_gamma = float(weights.get("instability", 0.0))

    lookup = build_before_component_lookup(records=records, manifest=manifest, cfg=cfg, repo_root=repo_root)
    augmented = augment_records_with_before_components(records, lookup)
    write_decomposition_outputs(
        augmented=augmented,
        decomp_root=decomp_root,
        input_root=input_root,
        manifest=manifest,
        cfg=cfg,
    )
    analysis = run_sensitivity_analysis(
        augmented=augmented,
        output_root=sensitivity_root,
        alpha=alpha,
        beta=beta,
        baseline_gamma=baseline_gamma,
        train_seed_parity=train_seed_parity,
        bootstrap_reps=bootstrap_reps,
    )
    write_readmes(
        decomp_root=decomp_root,
        sensitivity_root=sensitivity_root,
        source_root=input_root,
        alpha=alpha,
        beta=beta,
        baseline_gamma=baseline_gamma,
        crossing_gamma=analysis["capture_crossing_gamma"],
        crossing_factor=analysis["capture_crossing_factor"],
    )
    return analysis
