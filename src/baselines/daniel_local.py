from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import importlib
import sys

import numpy as np
import torch

from src.baselines.base import BaselineOutput
from src.baselines.dispatching import dispatching_release_count
from src.solver.base import RepairDecision


DEFAULT_DANIEL_NETWORK = {
    "fea_j_input_dim": 10,
    "fea_m_input_dim": 8,
    "dropout_prob": 0.0,
    "num_heads_OAB": [4, 4],
    "num_heads_MAB": [4, 4],
    "layer_fea_output_dim": [32, 8],
    "num_mlp_layers_actor": 3,
    "hidden_dim_actor": 64,
    "num_mlp_layers_critic": 3,
    "hidden_dim_critic": 64,
}


@dataclass(slots=True)
class DanielBundle:
    model: torch.nn.Module
    checkpoint_path: Path
    checkpoint_label: str
    benchmark_root: Path
    device: str


@dataclass(slots=True)
class DanielReducedSubproblem:
    job_lengths: list[int]
    op_pt_matrix: np.ndarray
    local_to_original_op_ids: list[int]
    ranking_candidate_op_ids: list[int]
    closure_original_op_ids: list[int]


_DANIEL_IMPORTS: dict[str, Any] | None = None
_DANIEL_CACHE: dict[tuple[str, str, str], DanielBundle] = {}


def _normalize_path(path_like: str | Path | None) -> Path | None:
    if path_like is None:
        return None
    return Path(path_like).expanduser().resolve()


def _discover_benchmark_root(explicit_root: str | Path | None = None) -> Path:
    explicit = _normalize_path(explicit_root)
    if explicit is not None:
        if explicit.is_dir():
            return explicit
        raise FileNotFoundError(f"DANIEL benchmark root does not exist: {explicit}")

    desktop = Path.home() / "Desktop"
    candidates = sorted(desktop.rglob("Job_Shop_Scheduling_Benchmark_Environments_and_Instances"))
    if not candidates:
        raise FileNotFoundError("Could not auto-discover the Job Shop Scheduling Benchmark repository on Desktop.")
    return candidates[0].resolve()


def _ensure_daniel_imports(benchmark_root: Path) -> dict[str, Any]:
    global _DANIEL_IMPORTS
    if _DANIEL_IMPORTS is not None:
        return _DANIEL_IMPORTS

    root_text = str(benchmark_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    fjsp_module = importlib.import_module("solution_methods.DANIEL.fjsp_env_same_op_nums")
    common_utils = importlib.import_module("solution_methods.DANIEL.common_utils")
    model_module = importlib.import_module("solution_methods.DANIEL.model.main_model")
    _DANIEL_IMPORTS = {
        "FJSPEnvForSameOpNums": fjsp_module.FJSPEnvForSameOpNums,
        "greedy_select_action": common_utils.greedy_select_action,
        "DANIEL": model_module.DANIEL,
    }
    return _DANIEL_IMPORTS


def _parse_checkpoint_shape(path: Path) -> tuple[int, int] | None:
    stem = path.stem
    base = stem.split("+", 1)[0]
    if "x" not in base:
        return None
    left, right = base.split("x", 1)
    try:
        return int(left), int(right)
    except ValueError:
        return None


def _select_checkpoint(
    checkpoint_dir: Path,
    *,
    num_jobs: int,
    num_machines: int,
    preferred_label: str | None = None,
) -> tuple[Path, str]:
    if preferred_label:
        preferred_path = checkpoint_dir / f"{preferred_label}.pth"
        if preferred_path.exists():
            return preferred_path.resolve(), preferred_label
        raise FileNotFoundError(f"Requested DANIEL checkpoint is missing: {preferred_path}")

    candidates: list[tuple[int, int, str, Path]] = []
    for checkpoint_path in sorted(checkpoint_dir.glob("*.pth")):
        shape = _parse_checkpoint_shape(checkpoint_path)
        if shape is None:
            continue
        jobs, machines = shape
        distance = abs(jobs - num_jobs) + abs(machines - num_machines)
        candidates.append((distance, jobs * machines, checkpoint_path.stem, checkpoint_path.resolve()))

    if not candidates:
        raise FileNotFoundError(f"No usable DANIEL checkpoints found under {checkpoint_dir}")

    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    _, _, label, path = candidates[0]
    return path, label


def _build_model_config(device: str) -> dict[str, Any]:
    return {
        "device": {"name": device, "id": "0"},
        "network": dict(DEFAULT_DANIEL_NETWORK),
    }


def load_daniel_bundle(
    cfg: dict[str, Any],
    *,
    num_jobs: int,
    num_machines: int,
    device: str = "cpu",
) -> DanielBundle:
    daniel_cfg = cfg.get("daniel_baseline", {})
    benchmark_root = _discover_benchmark_root(daniel_cfg.get("benchmark_root"))
    checkpoint_source = str(daniel_cfg.get("checkpoint_source", "SD2"))
    checkpoint_dir = benchmark_root / "solution_methods" / "DANIEL" / "save" / checkpoint_source
    checkpoint_path, checkpoint_label = _select_checkpoint(
        checkpoint_dir,
        num_jobs=num_jobs,
        num_machines=num_machines,
        preferred_label=daniel_cfg.get("checkpoint_label"),
    )
    cache_key = (str(checkpoint_path), str(device), checkpoint_label)
    cached = _DANIEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    imports = _ensure_daniel_imports(benchmark_root)
    model = imports["DANIEL"](_build_model_config(device))
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    bundle = DanielBundle(
        model=model,
        checkpoint_path=checkpoint_path,
        checkpoint_label=checkpoint_label,
        benchmark_root=benchmark_root,
        device=device,
    )
    _DANIEL_CACHE[cache_key] = bundle
    return bundle


def build_daniel_reduced_subproblem(instance, snapshot) -> DanielReducedSubproblem:
    completed = set(snapshot.completed_op_ids)
    active = set(snapshot.active_op_ids)
    window_set = set(snapshot.window_op_ids)

    closure: set[int] = set()
    for op_id in snapshot.window_op_ids:
        op = instance.get_operation(op_id)
        job = instance.get_job(op.job_id)
        for predecessor in job.operations[: op.op_index + 1]:
            predecessor_id = predecessor.op_global_id
            if predecessor_id in completed or predecessor_id in active:
                continue
            closure.add(predecessor_id)

    ranking_candidates = [
        op_id
        for op_id in snapshot.window_op_ids
        if op_id in window_set and op_id not in completed and op_id not in active
    ]

    local_to_original: list[int] = []
    pt_rows: list[list[int]] = []
    job_lengths: list[int] = []
    num_machines = int(instance.num_machines)

    for job in instance.jobs:
        included_ops = [op for op in job.operations if op.op_global_id in closure]
        if not included_ops:
            continue
        job_lengths.append(len(included_ops))
        for op in included_ops:
            row = [0] * num_machines
            for option in op.options:
                row[int(option.machine_id)] = max(1, int(round(float(option.processing_time))))
            local_to_original.append(int(op.op_global_id))
            pt_rows.append(row)

    if not pt_rows or not job_lengths:
        op_pt_matrix = np.zeros((0, num_machines), dtype=np.int32)
    else:
        op_pt_matrix = np.asarray(pt_rows, dtype=np.int32)

    return DanielReducedSubproblem(
        job_lengths=job_lengths,
        op_pt_matrix=op_pt_matrix,
        local_to_original_op_ids=local_to_original,
        ranking_candidate_op_ids=ranking_candidates,
        closure_original_op_ids=sorted(closure),
    )


def _rollout_daniel_priority(
    subproblem: DanielReducedSubproblem,
    bundle: DanielBundle,
) -> tuple[list[int], list[int]]:
    if not subproblem.job_lengths or subproblem.op_pt_matrix.size == 0:
        return [], []

    imports = _ensure_daniel_imports(bundle.benchmark_root)
    env_cls = imports["FJSPEnvForSameOpNums"]
    greedy_select_action = imports["greedy_select_action"]

    env = env_cls(
        n_j=len(subproblem.job_lengths),
        n_m=int(subproblem.op_pt_matrix.shape[1]),
        device=torch.device(bundle.device),
    )
    env.set_initial_data(
        np.asarray([subproblem.job_lengths], dtype=np.int32),
        np.asarray([subproblem.op_pt_matrix], dtype=np.int32),
    )

    state = env.state
    done = False
    ordered_original_ids: list[int] = []
    chosen_machine_ids: list[int] = []
    while not done:
        with torch.no_grad():
            pi, _ = bundle.model(
                fea_j=state.fea_j_tensor,
                op_mask=state.op_mask_tensor,
                candidate=state.candidate_tensor,
                fea_m=state.fea_m_tensor,
                mch_mask=state.mch_mask_tensor,
                comp_idx=state.comp_idx_tensor,
                dynamic_pair_mask=state.dynamic_pair_mask_tensor,
                fea_pairs=state.fea_pairs_tensor,
            )
        action = greedy_select_action(pi)
        chosen_job = int((action // env.number_of_machines).item())
        chosen_machine = int((action % env.number_of_machines).item())
        chosen_local_op = int(env.candidate[env.env_idxs, chosen_job].item())
        ordered_original_ids.append(subproblem.local_to_original_op_ids[chosen_local_op])
        chosen_machine_ids.append(chosen_machine)
        state, _, done = env.step(action.cpu().numpy())

    return ordered_original_ids, chosen_machine_ids


def daniel_local_decision(
    instance,
    incumbent,
    snapshot,
    cfg: dict[str, Any],
    *,
    device: str = "cpu",
) -> BaselineOutput:
    reduced = build_daniel_reduced_subproblem(instance, snapshot)
    release_candidates = list(reduced.ranking_candidate_op_ids)
    if not release_candidates:
        immutable = sorted(set(snapshot.completed_op_ids + snapshot.active_op_ids))
        return BaselineOutput(
            decision=RepairDecision(
                immutable_op_ids=immutable,
                kept_op_ids=list(snapshot.window_op_ids),
                released_op_ids=[],
                metadata={
                    "selector_type": "daniel_local",
                    "checkpoint_label": "",
                    "closure_size": int(len(reduced.closure_original_op_ids)),
                },
            ),
            name="daniel_local",
        )

    bundle = load_daniel_bundle(
        cfg,
        num_jobs=max(1, len(reduced.job_lengths)),
        num_machines=int(instance.num_machines),
        device=device,
    )
    ordered_original_ids, chosen_machine_ids = _rollout_daniel_priority(reduced, bundle)

    candidate_set = set(release_candidates)
    ranked_release: list[int] = []
    seen: set[int] = set()
    for op_id in ordered_original_ids:
        if op_id not in candidate_set or op_id in seen:
            continue
        ranked_release.append(op_id)
        seen.add(op_id)

    for op_id in release_candidates:
        if op_id not in seen:
            ranked_release.append(op_id)

    release_count = dispatching_release_count(len(ranked_release))
    release = ranked_release[:release_count]
    keep = [op_id for op_id in snapshot.window_op_ids if op_id not in release]
    immutable = sorted(set(snapshot.completed_op_ids + snapshot.active_op_ids))

    preview_pairs = []
    for op_id, machine_id in zip(ordered_original_ids[: min(12, len(ordered_original_ids))], chosen_machine_ids[:12]):
        preview_pairs.append({"op_id": int(op_id), "machine_id": int(machine_id)})

    return BaselineOutput(
        decision=RepairDecision(
            immutable_op_ids=immutable,
            kept_op_ids=keep,
            released_op_ids=release,
            metadata={
                "selector_type": "daniel_local",
                "checkpoint_label": bundle.checkpoint_label,
                "checkpoint_path": str(bundle.checkpoint_path),
                "benchmark_root": str(bundle.benchmark_root),
                "closure_size": int(len(reduced.closure_original_op_ids)),
                "ranking_candidate_count": int(len(reduced.ranking_candidate_op_ids)),
                "rollout_preview": preview_pairs,
                "release_fraction": float(release_count) / float(max(1, len(ranked_release))),
            },
        ),
        name="daniel_local",
    )
