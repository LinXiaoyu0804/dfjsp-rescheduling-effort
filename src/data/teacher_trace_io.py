from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch


def freeze_graph_tensors(graph: dict[str, Any]) -> dict[str, Any]:
    frozen: dict[str, Any] = {}
    for key, value in graph.items():
        if isinstance(value, torch.Tensor):
            frozen[key] = value.detach().cpu().clone()
        else:
            frozen[key] = deepcopy(value)
    return frozen


def build_operation_dataset_record(
    sample_metadata: dict[str, Any],
    snapshot_row: dict[str, Any],
    shard_path: str | Path,
    sample_index: int,
) -> dict[str, Any]:
    return {
        "dataset_index": int(sample_index),
        "teacher_shard_path": str(Path(shard_path).as_posix()),
        "episode_id": str(sample_metadata["episode_id"]),
        "event_id": str(sample_metadata["event_id"]),
        "instance_id": str(snapshot_row["instance_id"]),
        "instance_path": str(snapshot_row["instance_path"]),
        "seed": int(snapshot_row["seed"]),
        "tau": float(snapshot_row["tau"]),
        "event_type": str(sample_metadata["event_type"]),
        "window_size": int(snapshot_row["window_size"]),
        "forced_release_count": int(snapshot_row["forced_release_count"]),
        "num_window_ops": int(sample_metadata["num_window_ops"]),
        "teacher_release_count": int(sample_metadata["teacher_release_count"]),
        "teacher_keep_count": int(sample_metadata["teacher_keep_count"]),
        "teacher_objective": None
        if sample_metadata.get("teacher_objective") is None
        else float(sample_metadata["teacher_objective"]),
        "teacher_feasible": bool(sample_metadata["teacher_feasible"]),
        "teacher_solver_status": str(sample_metadata["teacher_solver_status"]),
        "teacher_runtime_sec": float(sample_metadata["teacher_runtime_sec"]),
        "teacher_shrink_stage": str(sample_metadata["teacher_stage"]),
    }
