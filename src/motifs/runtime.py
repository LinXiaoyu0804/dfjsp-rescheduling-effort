from __future__ import annotations

from dataclasses import dataclass, field
from math import log, sqrt
from typing import Any

from src.events.serialization import serialize_event_payload
from src.motifs.grouped_probing import motif_release_support


def _forced_release_ops(snapshot) -> list[int]:
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


def build_runtime_snapshot_row(
    instance_id: str,
    seed: int,
    episode_id: str,
    event_id: str,
    env,
) -> dict[str, Any]:
    snapshot = env.state_snapshot
    event = env.current_event
    if snapshot is None or event is None:
        raise RuntimeError("State snapshot or event is not available.")
    relevant_ops = list(snapshot.window_op_ids) + list(snapshot.directly_impacted_op_ids)
    return {
        "episode_id": str(episode_id),
        "event_id": str(event_id),
        "instance_id": str(instance_id),
        "seed": int(seed),
        "tau": float(snapshot.current_time),
        "completed_ops": sorted(snapshot.completed_op_ids),
        "active_ops": sorted(snapshot.active_op_ids),
        "unfinished_ops": sorted(snapshot.unfinished_op_ids),
        "window_ops": sorted(snapshot.window_op_ids),
        "forced_release_ops": _forced_release_ops(snapshot),
        "incumbent_assignments": _serialize_incumbent_assignments(env, relevant_ops),
        "event_context": {
            "type": str(snapshot.triggering_event_type),
            "affected_ops": sorted(snapshot.directly_impacted_op_ids),
            "affected_machines": sorted(snapshot.affected_machine_ids),
            "payload": serialize_event_payload(event),
        },
    }


@dataclass(slots=True)
class MotifSelectionDecision:
    selected_motif_ids: list[str]
    released_op_ids: list[int]
    predicted_gain_sum: float
    predicted_cost_sum: float
    predicted_inst_sum: float
    motif_scores: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RuntimeMotifCandidate:
    motif_id: str
    family: str
    release_ops: list[int]
    q_prob: float
    gain: float
    inst: float
    cost: float
    utility: float
    score: float


def compute_release_cap(
    window_size: int,
    budget_sec: float,
    hard_cap: int = 80,
    eta0: float = 2.0,
    eta1: float = 0.15,
    eta2: float = 4.0,
) -> int:
    if window_size <= 0:
        return 0
    cap = int(round(eta0 + eta1 * window_size + eta2 * budget_sec))
    return max(1, min(int(hard_cap), cap, int(window_size)))


def _selector_gate_metadata(source: str, forced_release: set[int]) -> MotifSelectionDecision:
    return MotifSelectionDecision(
        selected_motif_ids=[],
        released_op_ids=sorted(forced_release),
        predicted_gain_sum=0.0,
        predicted_cost_sum=0.0,
        predicted_inst_sum=0.0,
        motif_scores={},
        metadata={
            "selection_source": source,
        },
    )


def _event_type_bucket(snapshot_row: dict[str, Any]) -> str:
    event_context = snapshot_row.get("event_context", {})
    event_type = str(event_context.get("type", "")).strip().lower()
    if "arrival" in event_type:
        return "arrival"
    if "breakdown" in event_type:
        return "breakdown"
    event_id = str(snapshot_row.get("event_id", "")).strip().lower()
    if event_id.startswith("arr_"):
        return "arrival"
    if event_id.startswith("bd_"):
        return "breakdown"
    return "other"


def _runtime_problem_size(snapshot_row: dict[str, Any]) -> int:
    explicit = snapshot_row.get("problem_size")
    if explicit is not None:
        return max(0, int(explicit))
    completed = len(snapshot_row.get("completed_ops", []))
    active = len(snapshot_row.get("active_ops", []))
    unfinished = len(snapshot_row.get("unfinished_ops", []))
    return max(0, completed + active + unfinished)


def _resolve_controller_runtime_profile(
    controller_cfg: dict[str, Any],
    snapshot_row: dict[str, Any],
) -> dict[str, Any]:
    problem_size = _runtime_problem_size(snapshot_row)
    profile_name = "default"
    family_release_fractions = [
        float(value)
        for value in controller_cfg.get("family_release_fractions", [0.4, 0.7, 1.0])
    ]
    family_budget_fractions = [
        float(value)
        for value in controller_cfg.get("family_budget_fractions", [0.35, 0.55, 0.8])
    ]
    release_expansion_fill_ratio = max(0.0, min(1.0, float(controller_cfg.get("release_expansion_fill_ratio", 1.0))))

    small_instance_total_ops_max = int(controller_cfg.get("small_instance_total_ops_max", 0))
    if small_instance_total_ops_max > 0 and problem_size > 0 and problem_size <= small_instance_total_ops_max:
        profile_name = "small_instance"
        family_release_fractions = [
            float(value)
            for value in controller_cfg.get("small_instance_family_release_fractions", family_release_fractions)
        ]
        family_budget_fractions = [
            float(value)
            for value in controller_cfg.get("small_instance_family_budget_fractions", family_budget_fractions)
        ]
        release_expansion_fill_ratio = max(
            0.0,
            min(
                1.0,
                float(
                    controller_cfg.get(
                        "small_instance_release_expansion_fill_ratio",
                        release_expansion_fill_ratio,
                    )
                ),
            ),
        )

    return {
        "problem_size": problem_size,
        "profile_name": profile_name,
        "family_release_fractions": family_release_fractions,
        "family_budget_fractions": family_budget_fractions,
        "release_expansion_fill_ratio": release_expansion_fill_ratio,
    }


def _passes_selector_activation_gate(snapshot_row: dict[str, Any], selector_cfg: dict[str, Any]) -> bool:
    min_window_size = int(selector_cfg.get("min_window_size", 0))
    min_motif_count = int(selector_cfg.get("min_motif_count", 0))
    min_forced_release_count = int(selector_cfg.get("min_forced_release_count", 0))
    window_size = len(snapshot_row.get("window_ops", []))
    forced_release_count = len(snapshot_row.get("forced_release_ops", []))
    motif_count = int(selector_cfg.get("_runtime_motif_count", 0))
    return (
        window_size >= min_window_size
        and motif_count >= min_motif_count
        and forced_release_count >= min_forced_release_count
    )


def _build_runtime_candidates(
    motif_rows: list[dict[str, Any]],
    motif_predictions: list[dict[str, float]],
    instance,
    snapshot_row: dict[str, Any],
    selector_cfg: dict[str, Any],
) -> list[RuntimeMotifCandidate]:
    eps = float(selector_cfg.get("eps", 1e-6))
    lambda_d = float(selector_cfg.get("lambda_d", 1.0))
    gain_scale = float(selector_cfg.get("gain_scale", 1.0))
    inst_scale = float(selector_cfg.get("inst_scale", 1.0))
    cost_scale = float(selector_cfg.get("cost_scale", 1.0))
    backward_depth = int(selector_cfg.get("precedence_closure_backward_depth", 1))
    forward_depth = int(selector_cfg.get("precedence_closure_forward_depth", 0))

    candidates: list[RuntimeMotifCandidate] = []
    for row, pred in zip(motif_rows, motif_predictions):
        if "release_ops" in row:
            release_ops = sorted(set(int(op_id) for op_id in row.get("release_ops", [])))
        else:
            if instance is None:
                raise RuntimeError("Runtime candidate construction needs either precomputed release_ops or an instance.")
            release_support = motif_release_support(
                row,
                instance=instance,
                snapshot_row=snapshot_row,
                teacher_release_ops=set(),
                backward_depth=backward_depth,
                forward_depth=forward_depth,
            )
            release_ops = release_support["induced_release_ops"]
        calibrated_gain = gain_scale * float(pred["gain"])
        calibrated_inst = inst_scale * float(pred["inst"])
        calibrated_cost = max(eps, cost_scale * float(pred["cost"]))
        utility = float(pred["q_prob"]) * calibrated_gain - lambda_d * calibrated_inst
        score = utility / calibrated_cost
        candidates.append(
            RuntimeMotifCandidate(
                motif_id=str(row["motif_id"]),
                family=str(row.get("family", "M1")),
                release_ops=release_ops,
                q_prob=float(pred["q_prob"]),
                gain=calibrated_gain,
                inst=calibrated_inst,
                cost=calibrated_cost,
                utility=utility,
                score=score,
            )
        )
    return candidates


def _select_from_candidates(
    candidates: list[RuntimeMotifCandidate],
    forced_release: set[int],
    release_cap: int,
    q_threshold: float,
    budget_fraction_limit: float,
    top_k_fallback: int,
    fallback_min_q_prob: float,
) -> MotifSelectionDecision:
    ranked = sorted(
        candidates,
        key=lambda item: (-item.score, -item.utility, item.motif_id),
    )

    selected_ids: list[str] = []
    selected_release = set(forced_release)
    predicted_gain_sum = 0.0
    predicted_cost_sum = 0.0
    predicted_inst_sum = 0.0
    motif_scores: dict[str, float] = {}

    for candidate in ranked:
        motif_scores[candidate.motif_id] = candidate.score
        if candidate.q_prob < q_threshold or candidate.utility <= 0.0:
            continue
        candidate_release = set(candidate.release_ops)
        if not (candidate_release - selected_release):
            continue
        proposed_release = selected_release | candidate_release
        proposed_cost = predicted_cost_sum + candidate.cost
        if len(proposed_release) > release_cap:
            continue
        if selected_ids and proposed_cost > budget_fraction_limit:
            continue
        selected_ids.append(candidate.motif_id)
        selected_release = proposed_release
        predicted_gain_sum += candidate.gain
        predicted_cost_sum = proposed_cost
        predicted_inst_sum += candidate.inst

    if not selected_ids and top_k_fallback > 0:
        fallback_count = 0
        for candidate in ranked:
            if candidate.q_prob < fallback_min_q_prob or candidate.utility <= 0.0:
                continue
            candidate_release = set(candidate.release_ops)
            if not (candidate_release - selected_release):
                continue
            proposed_release = selected_release | candidate_release
            proposed_cost = predicted_cost_sum + candidate.cost
            if len(proposed_release) > release_cap:
                continue
            if selected_ids and proposed_cost > budget_fraction_limit:
                continue
            selected_ids.append(candidate.motif_id)
            selected_release = proposed_release
            predicted_gain_sum += candidate.gain
            predicted_cost_sum = proposed_cost
            predicted_inst_sum += candidate.inst
            fallback_count += 1
            if fallback_count >= top_k_fallback:
                break

    return MotifSelectionDecision(
        selected_motif_ids=selected_ids,
        released_op_ids=sorted(selected_release),
        predicted_gain_sum=predicted_gain_sum,
        predicted_cost_sum=predicted_cost_sum,
        predicted_inst_sum=predicted_inst_sum,
        motif_scores=motif_scores,
    )


def _expand_selection_release(
    selection: MotifSelectionDecision,
    candidates: list[RuntimeMotifCandidate],
    forced_release: set[int],
    release_cap: int,
    target_release_count: int,
    *,
    min_q_prob: float,
) -> tuple[MotifSelectionDecision, int]:
    selected_ids = set(selection.selected_motif_ids)
    selected_release = set(selection.released_op_ids)
    if not selected_ids or len(selected_release) >= target_release_count:
        return selection, 0

    added_motifs = 0
    ranked = sorted(candidates, key=lambda item: (-item.score, -item.utility, item.motif_id))
    for candidate in ranked:
        if candidate.motif_id in selected_ids:
            continue
        if candidate.q_prob < min_q_prob or candidate.utility <= 0.0:
            continue
        candidate_release = set(candidate.release_ops)
        new_ops = candidate_release - selected_release
        if not new_ops:
            continue
        proposed_release = selected_release | candidate_release
        if len(proposed_release) > release_cap:
            continue

        selection.selected_motif_ids.append(candidate.motif_id)
        selected_ids.add(candidate.motif_id)
        selected_release = proposed_release
        selection.released_op_ids = sorted(selected_release)
        selection.predicted_gain_sum += candidate.gain
        selection.predicted_cost_sum += candidate.cost
        selection.predicted_inst_sum += candidate.inst
        added_motifs += 1
        if len(selected_release) >= target_release_count:
            break

    return selection, added_motifs


def _selection_to_shortlist_entry(
    selection: MotifSelectionDecision,
    arm: dict[str, Any],
    *,
    event_type_bucket: str,
) -> dict[str, Any]:
    return {
        "selected_motif_ids": list(selection.selected_motif_ids),
        "released_op_ids": list(selection.released_op_ids),
        "predicted_gain_sum": float(selection.predicted_gain_sum),
        "predicted_cost_sum": float(selection.predicted_cost_sum),
        "predicted_inst_sum": float(selection.predicted_inst_sum),
        "motif_scores": dict(selection.motif_scores),
        "controller_mode": "alns_lite",
        "selection_source": "alns_lite",
        "selected_operator_key": arm["operator_key"],
        "selected_operator_family": arm["family"],
        "operator_prior_score": float(arm["prior_score"]),
        "operator_total_score": float(arm["controller_score"]),
        "operator_release_fraction": float(arm["release_fraction"]),
        "operator_budget_fraction": float(arm["budget_fraction"]),
        "operator_family_reward_ema_before": float(arm["family_reward_ema"]),
        "operator_family_pulls_before": int(arm["family_pulls"]),
        "operator_family_bonus": float(arm["family_bonus"]),
        "operator_family_prior_bias": float(arm["family_prior_bias"]),
        "operator_family_q_threshold_scale": float(arm["family_q_threshold_scale"]),
        "operator_family_fallback_min_q_prob_scale": float(arm["family_fallback_min_q_prob_scale"]),
        "operator_family_target_release_ops": float(arm["family_target_release_ops"]),
        "operator_family_target_cost": float(arm["family_target_cost"]),
        "operator_family_target_gain_per_release": float(arm["family_target_gain_per_release"]),
        "operator_release_match_score": float(arm["release_match_score"]),
        "operator_cost_match_score": float(arm["cost_match_score"]),
        "operator_gain_density_match_score": float(arm["gain_density_match_score"]),
        "operator_selection_mode": str(arm["selection_mode"]),
        "release_cap": int(arm["release_cap"]),
        "event_type_bucket": event_type_bucket,
        "release_expansion_added_motifs": int(arm["release_expansion_added_motifs"]),
        "release_expansion_target_count": int(arm["release_expansion_target_count"]),
    }


def _clone_selection(selection: MotifSelectionDecision) -> MotifSelectionDecision:
    return MotifSelectionDecision(
        selected_motif_ids=list(selection.selected_motif_ids),
        released_op_ids=list(selection.released_op_ids),
        predicted_gain_sum=float(selection.predicted_gain_sum),
        predicted_cost_sum=float(selection.predicted_cost_sum),
        predicted_inst_sum=float(selection.predicted_inst_sum),
        motif_scores=dict(selection.motif_scores),
        metadata=dict(selection.metadata),
    )


def _should_emit_intensified_candidate(
    snapshot_row: dict[str, Any],
    controller_cfg: dict[str, Any],
    *,
    current_release_count: int,
    target_release_count: int,
) -> bool:
    if target_release_count <= current_release_count:
        return False
    window_size = len(snapshot_row.get("window_ops", []))
    forced_release_count = len(snapshot_row.get("forced_release_ops", []))
    motif_count = int(snapshot_row.get("motif_count", 0))
    if window_size < int(controller_cfg.get("acceptance_intensify_min_window_size", 0)):
        return False
    if forced_release_count < int(controller_cfg.get("acceptance_intensify_min_forced_release_count", 0)):
        return False
    if motif_count < int(controller_cfg.get("acceptance_intensify_min_motif_count", 0)):
        return False
    return True


def select_pairwise_motifs(
    motif_rows: list[dict[str, Any]],
    motif_predictions: list[dict[str, float]],
    instance,
    snapshot_row: dict[str, Any],
    selector_cfg: dict[str, Any],
    budget_sec: float,
) -> MotifSelectionDecision:
    forced_release = set(int(op_id) for op_id in snapshot_row.get("forced_release_ops", []))
    selector_cfg = dict(selector_cfg)
    selector_cfg["_runtime_motif_count"] = len(motif_rows)
    if not _passes_selector_activation_gate(snapshot_row, selector_cfg):
        selection = _selector_gate_metadata("gated_off", forced_release)
        selection.metadata.update(
            {
                "controller_mode": "pairwise",
                "release_cap": len(forced_release),
            }
        )
        return selection
    release_cap = compute_release_cap(
        window_size=len(snapshot_row.get("window_ops", [])),
        budget_sec=budget_sec,
        hard_cap=int(selector_cfg.get("release_cap_max", 80)),
        eta0=float(selector_cfg.get("release_cap_eta0", 2.0)),
        eta1=float(selector_cfg.get("release_cap_eta1", 0.15)),
        eta2=float(selector_cfg.get("release_cap_eta2", 4.0)),
    )
    q_threshold = float(selector_cfg.get("q_threshold", 0.5))
    budget_fraction_limit = float(selector_cfg.get("budget_fraction_limit", 0.8))
    top_k_fallback = int(selector_cfg.get("top_k_fallback", 0))
    fallback_min_q_prob = float(selector_cfg.get("fallback_min_q_prob", 0.0))
    candidates = _build_runtime_candidates(
        motif_rows=motif_rows,
        motif_predictions=motif_predictions,
        instance=instance,
        snapshot_row=snapshot_row,
        selector_cfg=selector_cfg,
    )
    selection = _select_from_candidates(
        candidates=candidates,
        forced_release=forced_release,
        release_cap=release_cap,
        q_threshold=q_threshold,
        budget_fraction_limit=budget_fraction_limit,
        top_k_fallback=top_k_fallback,
        fallback_min_q_prob=fallback_min_q_prob,
    )
    selection.metadata.update(
        {
            "controller_mode": "pairwise",
            "selection_source": "pairwise",
            "release_cap": release_cap,
        }
    )
    return selection


def init_alns_lite_controller_state(controller_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    calibration = {}
    if controller_cfg is not None:
        calibration = dict(controller_cfg.get("initial_state", {}))
    family_stats_cfg = calibration.get("family_stats", {})
    family_stats: dict[str, dict[str, float | int]] = {}
    if isinstance(family_stats_cfg, dict):
        for family, stats in family_stats_cfg.items():
            if not isinstance(stats, dict):
                continue
            family_stats[str(family)] = {
                "pulls": int(stats.get("pulls", 0)),
                "reward_ema": float(stats.get("reward_ema", 0.0)),
                "reward_sum": float(stats.get("reward_sum", 0.0)),
                "positive_updates": int(stats.get("positive_updates", 0)),
            }
    return {
        "total_updates": int(calibration.get("total_updates", 0)),
        "family_stats": family_stats,
    }


def _scaled_release_cap(base_cap: int, forced_release_count: int, release_fraction: float) -> int:
    additional_capacity = max(1, int(round(max(1, base_cap - forced_release_count) * max(0.0, float(release_fraction)))))
    return min(base_cap, forced_release_count + additional_capacity)


def _compute_operator_prior_score(
    selection: MotifSelectionDecision,
    forced_release_count: int,
    release_cap: int,
    lambda_d: float,
    eps: float,
    release_coverage_weight: float,
    score_transform: str,
    score_cap: float,
) -> float:
    raw_density = (
        selection.predicted_gain_sum - lambda_d * selection.predicted_inst_sum
    ) / max(eps, selection.predicted_cost_sum)
    density = max(0.0, raw_density)
    normalized_density = density
    if score_transform == "log1p":
        normalized_density = log(1.0 + density)
    elif score_transform == "sqrt":
        normalized_density = sqrt(density)
    elif score_transform == "clip":
        normalized_density = density
    normalized_density = min(float(score_cap), normalized_density)
    released_delta = max(0, len(selection.released_op_ids) - forced_release_count)
    coverage = released_delta / max(1, release_cap - forced_release_count)
    return normalized_density + release_coverage_weight * coverage


def _match_score(observed: float, target: float) -> float:
    if target <= 0.0:
        return 0.0
    return max(0.0, 1.0 - abs(float(observed) - float(target)) / max(1.0, float(target)))


def _resolve_family_calibration(
    controller_cfg: dict[str, Any],
    family: str,
    event_type_bucket: str,
) -> dict[str, Any]:
    family_cfg: dict[str, Any] = {}
    family_calibration = controller_cfg.get("family_calibration", {})
    if isinstance(family_calibration, dict):
        base_cfg = family_calibration.get(family, {})
        if isinstance(base_cfg, dict):
            family_cfg.update(base_cfg)
    event_type_family_calibration = controller_cfg.get("event_type_family_calibration", {})
    if isinstance(event_type_family_calibration, dict):
        bucket_cfg = event_type_family_calibration.get(event_type_bucket, {})
        if isinstance(bucket_cfg, dict):
            event_specific_cfg = bucket_cfg.get(family, {})
            if isinstance(event_specific_cfg, dict):
                family_cfg.update(event_specific_cfg)
    return family_cfg


def select_alns_lite_motifs(
    motif_rows: list[dict[str, Any]],
    motif_predictions: list[dict[str, float]],
    instance,
    snapshot_row: dict[str, Any],
    selector_cfg: dict[str, Any],
    controller_cfg: dict[str, Any],
    controller_state: dict[str, Any] | None,
    budget_sec: float,
) -> MotifSelectionDecision:
    state = controller_state if controller_state is not None else init_alns_lite_controller_state(controller_cfg)
    forced_release = set(int(op_id) for op_id in snapshot_row.get("forced_release_ops", []))
    selector_cfg = dict(selector_cfg)
    selector_cfg["_runtime_motif_count"] = len(motif_rows)
    if not _passes_selector_activation_gate(snapshot_row, selector_cfg):
        selection = _selector_gate_metadata("gated_off", forced_release)
        selection.metadata.update(
            {
                "controller_mode": "alns_lite",
                "release_cap": len(forced_release),
                "selected_operator_key": None,
                "selected_operator_family": None,
            }
        )
        return selection
    release_cap = compute_release_cap(
        window_size=len(snapshot_row.get("window_ops", [])),
        budget_sec=budget_sec,
        hard_cap=int(selector_cfg.get("release_cap_max", 80)),
        eta0=float(selector_cfg.get("release_cap_eta0", 2.0)),
        eta1=float(selector_cfg.get("release_cap_eta1", 0.15)),
        eta2=float(selector_cfg.get("release_cap_eta2", 4.0)),
    )
    q_threshold = float(selector_cfg.get("q_threshold", 0.5))
    budget_fraction_limit = float(selector_cfg.get("budget_fraction_limit", 0.8))
    top_k_fallback = int(selector_cfg.get("top_k_fallback", 0))
    fallback_min_q_prob = float(selector_cfg.get("fallback_min_q_prob", 0.0))
    lambda_d = float(selector_cfg.get("lambda_d", 1.0))
    eps = float(selector_cfg.get("eps", 1e-6))

    candidates = _build_runtime_candidates(
        motif_rows=motif_rows,
        motif_predictions=motif_predictions,
        instance=instance,
        snapshot_row=snapshot_row,
        selector_cfg=selector_cfg,
    )
    pairwise_fallback = select_pairwise_motifs(
        motif_rows=motif_rows,
        motif_predictions=motif_predictions,
        instance=instance,
        snapshot_row=snapshot_row,
        selector_cfg=selector_cfg,
        budget_sec=budget_sec,
    )
    if not candidates:
        pairwise_fallback.metadata.update(
            {
                "controller_mode": "alns_lite",
                "selection_source": "pairwise_fallback",
                "selected_operator_key": None,
                "selected_operator_family": None,
                "event_type_bucket": _event_type_bucket(snapshot_row),
                "problem_size": problem_size,
                "controller_profile": controller_profile,
            }
        )
        return pairwise_fallback

    prior_weight = float(controller_cfg.get("prior_weight", 1.0))
    adaptive_weight = float(controller_cfg.get("adaptive_weight", 0.35))
    exploration_weight = float(controller_cfg.get("exploration_weight", 0.15))
    release_coverage_weight = float(controller_cfg.get("release_coverage_weight", 0.1))
    runtime_profile = _resolve_controller_runtime_profile(controller_cfg, snapshot_row)
    family_release_fractions = runtime_profile["family_release_fractions"]
    family_budget_fractions = runtime_profile["family_budget_fractions"]
    min_controller_score = float(controller_cfg.get("min_controller_score", 0.0))
    fallback_to_pairwise = bool(controller_cfg.get("fallback_to_pairwise", True))
    allow_relaxed_family_fallback = bool(controller_cfg.get("allow_relaxed_family_fallback", False))
    relaxed_q_threshold_scale = max(0.0, float(controller_cfg.get("relaxed_q_threshold_scale", 0.5)))
    relaxed_fallback_scale = max(0.0, float(controller_cfg.get("relaxed_fallback_min_q_prob_scale", 0.25)))
    relaxed_top_k_fallback = max(
        top_k_fallback,
        int(controller_cfg.get("relaxed_top_k_fallback", max(1, top_k_fallback))),
    )
    release_match_weight = float(controller_cfg.get("release_match_weight", 0.0))
    cost_match_weight = float(controller_cfg.get("cost_match_weight", 0.0))
    gain_density_match_weight = float(controller_cfg.get("gain_density_match_weight", 0.0))
    prior_score_transform = str(controller_cfg.get("prior_score_transform", "log1p")).lower()
    prior_score_cap = float(controller_cfg.get("prior_score_cap", 24.0))
    expand_release_to_operator_cap = bool(controller_cfg.get("expand_release_to_operator_cap", False))
    release_expansion_fill_ratio = runtime_profile["release_expansion_fill_ratio"]
    release_expansion_min_q_prob = max(0.0, float(controller_cfg.get("release_expansion_min_q_prob", 0.0)))
    acceptance_candidate_top_k = max(1, int(controller_cfg.get("acceptance_candidate_top_k", 1)))
    event_type_bucket = _event_type_bucket(snapshot_row)
    problem_size = int(runtime_profile["problem_size"])
    controller_profile = str(runtime_profile["profile_name"])

    families = sorted({candidate.family for candidate in candidates})
    family_stats = state.setdefault("family_stats", {})
    total_updates = int(state.get("total_updates", 0))
    operator_arms: list[dict[str, Any]] = []

    for family in families:
        family_candidates = [candidate for candidate in candidates if candidate.family == family]
        if not family_candidates:
            continue
        stats = family_stats.get(family, {})
        family_cfg = _resolve_family_calibration(
            controller_cfg=controller_cfg,
            family=family,
            event_type_bucket=event_type_bucket,
        )
        family_reward_ema = float(stats.get("reward_ema", 0.0))
        family_pulls = int(stats.get("pulls", 0))
        family_bonus = exploration_weight * sqrt(log(total_updates + 2.0) / (family_pulls + 1.0))
        family_prior_bias = float(family_cfg.get("prior_bias", 0.0))
        family_q_threshold_scale = max(0.0, float(family_cfg.get("q_threshold_scale", 1.0)))
        family_fallback_scale = max(0.0, float(family_cfg.get("fallback_min_q_prob_scale", 1.0)))
        family_target_release_ops = float(
            family_cfg.get("positive_mean_release_ops", family_cfg.get("mean_release_ops", 0.0))
        )
        family_target_cost = float(family_cfg.get("positive_mean_cost", 0.0))
        family_target_gain_per_release = float(family_cfg.get("positive_mean_gain_per_release", 0.0))

        for release_fraction in family_release_fractions:
            arm_release_cap = _scaled_release_cap(
                base_cap=release_cap,
                forced_release_count=len(forced_release),
                release_fraction=release_fraction,
            )
            for budget_fraction in family_budget_fractions:
                arm_budget_limit = min(budget_fraction_limit, float(budget_fraction))
                selection_mode = "standard"
                selection = _select_from_candidates(
                    candidates=family_candidates,
                    forced_release=forced_release,
                    release_cap=arm_release_cap,
                    q_threshold=q_threshold * family_q_threshold_scale,
                    budget_fraction_limit=arm_budget_limit,
                    top_k_fallback=top_k_fallback,
                    fallback_min_q_prob=fallback_min_q_prob * family_fallback_scale,
                )
                if (
                    not selection.selected_motif_ids
                    and allow_relaxed_family_fallback
                    and (family_prior_bias > 0.0 or family_reward_ema > 0.0)
                ):
                    relaxed_selection = _select_from_candidates(
                        candidates=family_candidates,
                        forced_release=forced_release,
                        release_cap=arm_release_cap,
                        q_threshold=q_threshold * family_q_threshold_scale * relaxed_q_threshold_scale,
                        budget_fraction_limit=arm_budget_limit,
                        top_k_fallback=relaxed_top_k_fallback,
                        fallback_min_q_prob=fallback_min_q_prob * family_fallback_scale * relaxed_fallback_scale,
                    )
                    if relaxed_selection.selected_motif_ids:
                        selection = relaxed_selection
                        selection_mode = "relaxed"
                if not selection.selected_motif_ids:
                    continue
                expansion_added_motifs = 0
                expansion_target_release_count = len(selection.released_op_ids)
                if expand_release_to_operator_cap and release_expansion_fill_ratio > 0.0:
                    expansion_target_release_count = min(
                        arm_release_cap,
                        len(forced_release)
                        + int(round((arm_release_cap - len(forced_release)) * release_expansion_fill_ratio)),
                    )
                    selection, expansion_added_motifs = _expand_selection_release(
                        selection=selection,
                        candidates=family_candidates,
                        forced_release=forced_release,
                        release_cap=arm_release_cap,
                        target_release_count=expansion_target_release_count,
                        min_q_prob=release_expansion_min_q_prob,
                    )
                prior_score = _compute_operator_prior_score(
                    selection=selection,
                    forced_release_count=len(forced_release),
                    release_cap=arm_release_cap,
                    lambda_d=lambda_d,
                    eps=eps,
                    release_coverage_weight=release_coverage_weight,
                    score_transform=prior_score_transform,
                    score_cap=prior_score_cap,
                )
                selected_delta_release = max(0, len(selection.released_op_ids) - len(forced_release))
                release_match_score = _match_score(selected_delta_release, family_target_release_ops)
                cost_match_score = _match_score(selection.predicted_cost_sum, family_target_cost)
                gain_per_release = selection.predicted_gain_sum / max(1.0, float(selected_delta_release))
                gain_density_match_score = _match_score(gain_per_release, family_target_gain_per_release)
                controller_score = (
                    prior_weight * prior_score
                    + adaptive_weight * family_reward_ema
                    + family_bonus
                    + family_prior_bias
                    + release_match_weight * release_match_score
                    + cost_match_weight * cost_match_score
                    + gain_density_match_weight * gain_density_match_score
                )
                operator_key = (
                    f"{family}|release={release_fraction:.2f}|budget={arm_budget_limit:.2f}"
                )
                operator_arms.append(
                    {
                        "operator_key": operator_key,
                        "family": family,
                        "release_fraction": float(release_fraction),
                        "budget_fraction": float(arm_budget_limit),
                        "selection": selection,
                        "prior_score": float(prior_score),
                        "controller_score": float(controller_score),
                        "family_reward_ema": family_reward_ema,
                        "family_pulls": family_pulls,
                        "family_bonus": family_bonus,
                        "family_prior_bias": family_prior_bias,
                        "family_q_threshold_scale": family_q_threshold_scale,
                        "family_fallback_min_q_prob_scale": family_fallback_scale,
                        "family_target_release_ops": family_target_release_ops,
                        "family_target_cost": family_target_cost,
                        "family_target_gain_per_release": family_target_gain_per_release,
                        "release_match_score": release_match_score,
                        "cost_match_score": cost_match_score,
                        "gain_density_match_score": gain_density_match_score,
                        "selection_mode": selection_mode,
                        "release_cap": arm_release_cap,
                        "event_type_bucket": event_type_bucket,
                        "release_expansion_added_motifs": expansion_added_motifs,
                        "release_expansion_target_count": expansion_target_release_count,
                    }
                )

    if not operator_arms:
        if fallback_to_pairwise:
            pairwise_fallback.metadata.update(
                {
                    "controller_mode": "alns_lite",
                    "selection_source": "pairwise_fallback",
                    "selected_operator_key": None,
                    "selected_operator_family": None,
                    "event_type_bucket": event_type_bucket,
                    "problem_size": problem_size,
                    "controller_profile": controller_profile,
                }
            )
            return pairwise_fallback
        return MotifSelectionDecision(
            selected_motif_ids=[],
            released_op_ids=sorted(forced_release),
            predicted_gain_sum=0.0,
            predicted_cost_sum=0.0,
            predicted_inst_sum=0.0,
            motif_scores={candidate.motif_id: candidate.score for candidate in candidates},
            metadata={
                "controller_mode": "alns_lite",
                "selection_source": "empty",
                "selected_operator_key": None,
                "selected_operator_family": None,
                "release_cap": release_cap,
                "event_type_bucket": event_type_bucket,
                "problem_size": problem_size,
                "controller_profile": controller_profile,
            },
        )

    ranked_arms = sorted(
        operator_arms,
        key=lambda item: (
            item["controller_score"],
            item["prior_score"],
            len(item["selection"].selected_motif_ids),
            item["operator_key"],
        ),
        reverse=True,
    )
    best_arm = ranked_arms[0]
    if best_arm["controller_score"] < min_controller_score and fallback_to_pairwise:
        pairwise_fallback.metadata.update(
            {
                "controller_mode": "alns_lite",
                "selection_source": "pairwise_fallback",
                "selected_operator_key": None,
                "selected_operator_family": None,
                "event_type_bucket": event_type_bucket,
                "problem_size": problem_size,
                "controller_profile": controller_profile,
            }
        )
        return pairwise_fallback

    selection = best_arm["selection"]
    selection.metadata.update(
        {
            "controller_mode": "alns_lite",
            "selection_source": "alns_lite",
            "selected_operator_key": best_arm["operator_key"],
            "selected_operator_family": best_arm["family"],
            "operator_prior_score": best_arm["prior_score"],
            "operator_total_score": best_arm["controller_score"],
            "operator_release_fraction": best_arm["release_fraction"],
            "operator_budget_fraction": best_arm["budget_fraction"],
            "operator_family_reward_ema_before": best_arm["family_reward_ema"],
            "operator_family_pulls_before": best_arm["family_pulls"],
            "operator_family_bonus": best_arm["family_bonus"],
            "operator_family_prior_bias": best_arm["family_prior_bias"],
            "operator_family_q_threshold_scale": best_arm["family_q_threshold_scale"],
            "operator_family_fallback_min_q_prob_scale": best_arm["family_fallback_min_q_prob_scale"],
            "operator_family_target_release_ops": best_arm["family_target_release_ops"],
            "operator_family_target_cost": best_arm["family_target_cost"],
            "operator_family_target_gain_per_release": best_arm["family_target_gain_per_release"],
            "operator_release_match_score": best_arm["release_match_score"],
            "operator_cost_match_score": best_arm["cost_match_score"],
            "operator_gain_density_match_score": best_arm["gain_density_match_score"],
            "operator_selection_mode": best_arm["selection_mode"],
            "release_cap": best_arm["release_cap"],
            "num_operator_arms": len(operator_arms),
            "event_type_bucket": event_type_bucket,
            "release_expansion_added_motifs": best_arm["release_expansion_added_motifs"],
            "release_expansion_target_count": best_arm["release_expansion_target_count"],
            "problem_size": problem_size,
            "controller_profile": controller_profile,
        }
    )
    shortlist_entries: list[dict[str, Any]] = []
    if acceptance_candidate_top_k > 1:
        shortlist_entries.extend(
            _selection_to_shortlist_entry(arm["selection"], arm, event_type_bucket=event_type_bucket)
            for arm in ranked_arms[:acceptance_candidate_top_k]
        )
    if bool(controller_cfg.get("acceptance_emit_intensified_candidate", False)):
        intensify_fill_ratio = max(
            release_expansion_fill_ratio,
            min(1.0, float(controller_cfg.get("acceptance_intensify_fill_ratio", 1.0))),
        )
        intensified_target_release_count = min(
            best_arm["release_cap"],
            len(forced_release)
            + int(round((best_arm["release_cap"] - len(forced_release)) * intensify_fill_ratio)),
        )
        if _should_emit_intensified_candidate(
            snapshot_row=snapshot_row,
            controller_cfg=controller_cfg,
            current_release_count=len(selection.released_op_ids),
            target_release_count=intensified_target_release_count,
        ):
            intensified_selection = _clone_selection(selection)
            family_candidates = [candidate for candidate in candidates if candidate.family == best_arm["family"]]
            intensified_selection, intensified_added_motifs = _expand_selection_release(
                selection=intensified_selection,
                candidates=family_candidates,
                forced_release=forced_release,
                release_cap=best_arm["release_cap"],
                target_release_count=intensified_target_release_count,
                min_q_prob=release_expansion_min_q_prob,
            )
            intensified_added_release = max(
                0,
                len(intensified_selection.released_op_ids) - len(selection.released_op_ids),
            )
            min_added_release = max(1, int(controller_cfg.get("acceptance_intensify_min_added_release_ops", 1)))
            if intensified_added_motifs > 0 and intensified_added_release >= min_added_release:
                intensified_arm = dict(best_arm)
                intensified_arm["operator_key"] = f"{best_arm['operator_key']}|intensified"
                intensified_arm["selection_mode"] = "intensified"
                intensified_arm["release_expansion_added_motifs"] = (
                    int(best_arm["release_expansion_added_motifs"]) + intensified_added_motifs
                )
                intensified_arm["release_expansion_target_count"] = intensified_target_release_count
                shortlist_entries.append(
                    _selection_to_shortlist_entry(
                        intensified_selection,
                        intensified_arm,
                        event_type_bucket=event_type_bucket,
                    )
                )
    if shortlist_entries:
        unique_entries: list[dict[str, Any]] = []
        seen_signatures: set[tuple[str, tuple[int, ...], tuple[str, ...]]] = set()
        for entry in shortlist_entries:
            signature = (
                str(entry.get("selected_operator_key", "")),
                tuple(int(op_id) for op_id in entry.get("released_op_ids", [])),
                tuple(str(motif_id) for motif_id in entry.get("selected_motif_ids", [])),
            )
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            unique_entries.append(entry)
        selection.metadata["acceptance_candidate_shortlist"] = unique_entries
    return selection


def update_alns_lite_controller_state(
    controller_state: dict[str, Any],
    selection: MotifSelectionDecision,
    objective_before: float,
    objective_after: float,
    runtime_sec: float,
    feasible: bool,
    controller_cfg: dict[str, Any],
) -> dict[str, float]:
    family = str(selection.metadata.get("selected_operator_family") or "")
    if not family or selection.metadata.get("selection_source") != "alns_lite":
        return {"normalized_reward": 0.0, "reward_ema_after": 0.0, "family_pulls_after": 0.0}

    raw_reward = float(objective_before) - float(objective_after)
    raw_reward -= float(controller_cfg.get("reward_runtime_penalty", 0.0)) * float(runtime_sec)
    raw_reward -= float(controller_cfg.get("reward_infeasible_penalty", 0.0)) * (0.0 if feasible else 1.0)

    normalization = str(controller_cfg.get("reward_normalization", "relative")).lower()
    if normalization == "relative":
        normalized_reward = raw_reward / max(1.0, abs(float(objective_before)))
    else:
        normalized_reward = raw_reward

    reward_clip_abs = float(controller_cfg.get("reward_clip_abs", 0.25))
    normalized_reward = max(-reward_clip_abs, min(reward_clip_abs, normalized_reward))

    family_stats = controller_state.setdefault("family_stats", {})
    stats = family_stats.setdefault(
        family,
        {
            "pulls": 0,
            "reward_ema": 0.0,
            "reward_sum": 0.0,
            "positive_updates": 0,
        },
    )
    stats["pulls"] = int(stats.get("pulls", 0)) + 1
    stats["reward_sum"] = float(stats.get("reward_sum", 0.0)) + normalized_reward
    if normalized_reward > 0.0:
        stats["positive_updates"] = int(stats.get("positive_updates", 0)) + 1
    reward_ema_decay = float(controller_cfg.get("reward_ema_decay", 0.8))
    previous_ema = float(stats.get("reward_ema", 0.0))
    if int(stats["pulls"]) <= 1:
        reward_ema_after = normalized_reward
    else:
        reward_ema_after = reward_ema_decay * previous_ema + (1.0 - reward_ema_decay) * normalized_reward
    stats["reward_ema"] = reward_ema_after
    controller_state["total_updates"] = int(controller_state.get("total_updates", 0)) + 1
    return {
        "normalized_reward": float(normalized_reward),
        "reward_ema_after": float(reward_ema_after),
        "family_pulls_after": float(stats["pulls"]),
    }
