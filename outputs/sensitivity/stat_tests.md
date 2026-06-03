# Baseline Gamma Statistical Tests

Baseline gamma: `0.2`.

| comparison | n | mean paired diff | 95% bootstrap CI | Wilcoxon p | note |
|---|---:|---:|---:|---:|---|
| Policy - Always-L0 | 4400 | 0.00002736 | [0.00001410, 0.00004116] | 4.42516e-11 |  |
| Policy - Oracle | 4400 | -0.00030506 | [-0.00036235, -0.00025041] | 6.82776e-138 |  |

The paired bootstrap resamples test events with replacement. Wilcoxon is two-sided with Pratt zero handling.
