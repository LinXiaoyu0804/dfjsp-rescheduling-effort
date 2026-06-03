from __future__ import annotations

from pathlib import Path

import pandas as pd


def aggregate_csv_files(input_dir: str | Path, output_path: str | Path) -> pd.DataFrame:
    input_dir = Path(input_dir)
    frames = [pd.read_csv(path) for path in input_dir.glob("*.csv")]
    if not frames:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")
    combined = pd.concat(frames, ignore_index=True)
    summary = combined.groupby("method", as_index=False).mean(numeric_only=True)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    return summary


def combine_csv_files(input_dir: str | Path, output_path: str | Path) -> pd.DataFrame:
    input_dir = Path(input_dir)
    frames = [pd.read_csv(path) for path in input_dir.glob("*.csv")]
    if not frames:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")
    combined = pd.concat(frames, ignore_index=True)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    return combined
