from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import REPO_ROOT

from src.eval.intensity_sensitivity import run_full_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Reweight intensity-grid event outcomes and export sensitivity figures.")
    parser.add_argument("--input-root", default="outputs/intensity_grid")
    parser.add_argument("--decomp-root", default="outputs/intensity_grid_decomp")
    parser.add_argument("--output-root", default="outputs/sensitivity")
    parser.add_argument("--train-seed-parity", type=int, default=0, choices=[0, 1])
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    args = parser.parse_args()

    input_root = (REPO_ROOT / Path(args.input_root)).resolve()
    decomp_root = (REPO_ROOT / Path(args.decomp_root)).resolve()
    output_root = (REPO_ROOT / Path(args.output_root)).resolve()
    print(f"Loading intensity grid from: {input_root}")
    print(f"Writing decomposition to: {decomp_root}")
    print(f"Writing sensitivity outputs to: {output_root}")
    run_full_pipeline(
        input_root=input_root,
        decomp_root=decomp_root,
        sensitivity_root=output_root,
        repo_root=REPO_ROOT,
        train_seed_parity=int(args.train_seed_parity),
        bootstrap_reps=int(args.bootstrap_reps),
    )
    print("Completed intensity sensitivity analysis.")


if __name__ == "__main__":
    main()
