# Objective Weight Sensitivity

Source decomposition: `outputs\intensity_grid_decomp`.

Baseline weights: `alpha=1`, `beta=1`, `gamma=0.2`. Gamma factors scanned: `[0.0, 0.25, 0.5, 1.0, 2.0, 4.0]`.

Honest policy split: even seeds for training, odd seeds for testing. The upgrade threshold is selected from leave-one-training-seed-out predictions only; test rows are not used for fitting or threshold tuning.

Capture negative-to-positive threshold: not observed.

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
