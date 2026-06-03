# Reproduction Gate

status: PASS

Scope: budget_sec=5.0, Brandimarte Mk8/Mk9/Mk10, frozen mixed traces.

| mode | n | mean_reward_delta | feasible_rate | mean_released_op_count |
|---|---:|---:|---:|---:|
| L0 | 240 | 0.010975 | 0.812500 | 8.613 |
| L3 | 240 | -0.052335 | 0.554167 | 30.900 |

Checks:
- always-L0 release rule delegates to heuristic_rh per event: PASS
- always-L3 release rule delegates to full_reoptimization per event: PASS
- sampled 20-event monotone release counts L0 <= L1 <= L2 <= L3: PASS
