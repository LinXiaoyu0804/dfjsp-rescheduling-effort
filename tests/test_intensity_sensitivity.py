from __future__ import annotations

import numpy as np

from src.eval.intensity_sensitivity import (
    choose_cheapest_best,
    normalized_reward,
    tune_upgrade_threshold,
)


def test_normalized_reward_uses_relative_incumbent_objective() -> None:
    reward = normalized_reward(
        makespan_before=100.0,
        tardiness_before=50.0,
        instability_before=0.0,
        makespan_after=90.0,
        tardiness_after=45.0,
        instability_after=10.0,
        alpha=1.0,
        beta=1.0,
        gamma=0.5,
    )
    assert reward == (150.0 - 140.0) / 150.0


def test_oracle_tie_chooses_cheapest_intensity() -> None:
    selected = choose_cheapest_best({"L0": 0.0, "L1": 0.2, "L2": 0.2, "L3": 0.1})
    assert selected == "L1"


def test_threshold_tuning_prefers_conservative_equal_reward_policy() -> None:
    predicted_gain = np.asarray([0.2, 0.2, -0.1])
    predicted_level_idx = np.asarray([1, 1, 1])
    reward_matrix = np.asarray(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    threshold, mean_reward, upgrade_rate = tune_upgrade_threshold(
        predicted_gain=predicted_gain,
        predicted_level_idx=predicted_level_idx,
        reward_matrix=reward_matrix,
    )
    assert threshold >= 0.2
    assert mean_reward == 0.0
    assert upgrade_rate == 0.0
