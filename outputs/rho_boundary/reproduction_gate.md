# R0 Reproduction Gate

status: PASS

Scope: R0 standard mixed traces, selected rho-boundary instances, budgets 0.5s and 1.0s.

| metric | value | gate |
|---|---:|---|
| mean oracle headroom | 0.017766% | target around 0.03% |
| test honest capture | 46.266649% | low-to-small, abs <= 50% |
| test mean policy gain | 0.00803238% | absolute gain <= 0.01% |
| max mean single non-L0 absolute gain | 0.01252155% | <= 0.02% |
| frac oracle headroom > 0 | 0.191071 | diagnostic |
| n events after budget/intensity matching | 1120 | diagnostic |

Honest policy details:
- threshold: `3.36858124358e-05`
- train events: `640`
- test events: `480`
- test upgrade rate: `0.637500`
