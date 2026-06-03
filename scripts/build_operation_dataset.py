from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from _bootstrap import REPO_ROOT  # noqa: F401

from src.data.teacher_trace_io import build_operation_dataset_record
from src.utils.io import ensure_dir, load_jsonl, save_jsonl


def _resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-dir", default="outputs/teacher/traces")
    parser.add_argument("--snapshots", default="outputs/states/state_snapshots.jsonl")
    parser.add_argument("--output-path", default="outputs/datasets/slr2_operation_dataset.pt")
    parser.add_argument("--metadata-jsonl", default=None)
    parser.add_argument("--summary-csv", default=None)
    args = parser.parse_args()

    teacher_dir = _resolve_path(args.teacher_dir)
    snapshots_path = _resolve_path(args.snapshots)
    output_path = _resolve_path(args.output_path)
    metadata_jsonl = _resolve_path(args.metadata_jsonl) if args.metadata_jsonl else output_path.with_suffix(".jsonl")
    summary_csv = _resolve_path(args.summary_csv) if args.summary_csv else output_path.with_name(f"{output_path.stem}_summary.csv")

    snapshot_rows = load_jsonl(snapshots_path)
    snapshot_index = {
        (str(row["episode_id"]), str(row["event_id"])): row
        for row in snapshot_rows
    }

    merged_samples: list[dict] = []
    metadata_rows: list[dict] = []
    for shard_path in sorted(teacher_dir.glob("*.pt")):
        shard_samples = torch.load(shard_path, map_location="cpu", weights_only=False)
        for sample in shard_samples:
            metadata = dict(sample.get("metadata", {}))
            key = (str(metadata["episode_id"]), str(metadata["event_id"]))
            if key not in snapshot_index:
                raise KeyError(f"Missing snapshot row for teacher sample {key}.")
            snapshot_row = snapshot_index[key]
            dataset_index = len(merged_samples)
            dataset_record = build_operation_dataset_record(metadata, snapshot_row, shard_path, dataset_index)
            sample["metadata"] = {**metadata, **dataset_record}
            merged_samples.append(sample)
            metadata_rows.append(dataset_record)

    ensure_dir(output_path.parent)
    torch.save(merged_samples, output_path)
    save_jsonl(metadata_rows, metadata_jsonl)

    summary_df = pd.DataFrame(metadata_rows)
    grouped = (
        summary_df.groupby(["instance_id", "seed"], as_index=False)
        .agg(
            num_events=("dataset_index", "count"),
            mean_window_size=("window_size", "mean"),
            mean_teacher_release_count=("teacher_release_count", "mean"),
            feasible_rate=("teacher_feasible", "mean"),
        )
        .sort_values(["instance_id", "seed"])
    )
    grouped.to_csv(summary_csv, index=False)
    print(f"Wrote {len(merged_samples)} operation samples to {output_path}")


if __name__ == "__main__":
    main()
