# Intensity Grid Decomposition

Source: `outputs\intensity_grid`.

Before components were recomputed from cached `incumbent_ref` schedules and frozen episode JSON traces by replaying `DFJSPReschedulingEnv.apply_event()` / `build_window()` only. No CP-SAT repair was invoked, and no intensity-grid cell was re-solved.

Fields added to each event row:

- `makespan_before`
- `tardiness_before`
- `instability_before` fixed to `0.0` for relative-to-incumbent reward reweighting
- `incumbent_instability_raw_before` retained only for auditing the original environment objective
- `objective_before_replay_weighted` using the original environment weights
- `size` as the post-event operation count

The existing source `reward_delta` is raw weighted-objective difference, not normalized: `objective_before.weighted_sum - weighted_objective_after`. The sensitivity outputs use `(J_before - J_after) / max(1, abs(J_before))`, with `J = alpha*Cmax + beta*sumT + gamma*I`.
