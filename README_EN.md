# Rescheduling Effort in Dynamic Flexible Job Shops

**Realizable value and the applicability of learning — an empirical study.** Code and experiment workspace for the paper.

> 中文说明见 [README_ZH.md](README_ZH.md). Full paper↔code↔table/figure mapping in [PAPER_MAP.md](PAPER_MAP.md).

## Research question

For predictive-reactive rescheduling of a dynamic flexible job shop (DFJSP) after disruptions, this paper studies an upstream question: under **incremental constraints, a response deadline, and an exact local-repair backend**, how much **realizable** improvement does deciding online *what and how much to repair* (the rescheduling effort) yield over the trivial **minimal right-shift** baseline?

To make this strictly comparable, the online decision is abstracted as a choice of **rescheduling effort** along a monotone effort ladder sharing one CP-SAT backend:

- **L0 — minimal right-shift**: shift-only feasibility recovery, no solver call, fastest, almost always feasible;
- **L1 — small neighborhood**: release direct job predecessors/successors and machine neighbors;
- **L2 — impact cluster**: additionally release the routing propagation segment and affected machines (M3/M4 motifs);
- **L3 — full reopen**: release the whole active window, approximating full reoptimization.

The four levels share one solver, one stability mechanism, and one response budget B_t; the only difference is the release-set size, decoupling *repair effort* from *the operator used*.

## Main findings

Across Brandimarte (Mk6–Mk10), three synthetic scales, three disturbance types, five budgets, and a ~30× range of ρ_t (~1.3×10⁴ events), a **two-mechanism** picture emerges:

1. At **low disturbance**, effort has structurally little leverage on the global objective (Proposition 1): the composite objective differs by < 0.13% across the four levels, the hindsight oracle over L0 is only ~0.03%, and the honest realizable gain — though statistically significant — is negligible (~10⁻³%);
2. At **high disturbance**, aggressive repair becomes infeasible within the budget (a **feasibility wall**), and the choice set collapses to minimal right-shift.

No operating point is both significantly effective and budget-feasible, so **effort selection lacks realizable value** across the realizable range, while **minimal right-shift stays robustly feasible**. Value more likely lies in an offline robustness layer.

## Layout

```
src/            core library
  scheduling/   effort ladder (L0-L3), composite objective, active window, incumbent, rho_t
  solver/       CP-SAT exact-repair backend, solver-time accounting
  baselines/    L0 (right-shift), L3 (full reopt), MWKR/ATC, DANIEL, DDPG, learned rule selector
  eval/         realizable-value-ceiling evaluation, gamma sensitivity, rho boundary, external baselines, metrics
  events/ env/  dynamic disturbance generation and rescheduling environment
  motifs/        L2 impact-cluster motif extraction
  data/ graph/ utils/
scripts/        14 paper experiment entrypoints (below)
configs/        instance, environment, solver, baseline configs
tests/          unit tests
outputs/        paper artifacts and frozen event trajectories (below)
```

`outputs/` ships the following **paper artifacts** and **frozen inputs**:

- `episodes/` — frozen event trajectories used in the paper (Brandimarte held-out + synthetic 30×10/50×15/100×20), fully reproducible by seed;
- `intensity_grid/`, `intensity_grid_decomp/` — §7.2–7.4 effort grid and objective-component decomposition;
- `sensitivity/` — §7.5 figures F1–F4, γ-sensitivity tables, statistical tests;
- `rho_boundary/` — §7.6 R0–R5 feasibility wall and headroom curves;
- `external_baselines/ddpg/` — the trained DDPG baseline checkpoint and its training data.

## Environment

Recommended `Python 3.10/3.11`:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Key dependencies: `ortools` (CP-SAT backend), `torch` (learned baselines), `numpy/pandas/scipy/matplotlib/seaborn`.

## Reproduction

The repo ships: the frozen event trajectories (`outputs/episodes/`), the paper figures
(`outputs/sensitivity/F1–F4`, `outputs/rho_boundary/*.png`), the summary-table CSVs, and the
trained DDPG checkpoint. The bulky per-event effort-grid jsonl are deterministically
regenerable by the pipeline and are therefore not shipped (see `.gitignore`).

```bash
# §7.2–7.5 effort grid -> oracle ceiling / realizability / gamma sensitivity -> figs F1-F4, Tables 3-7
python scripts/run_intensity_grid.py            # recompute grid -> outputs/intensity_grid(_decomp)
python scripts/analyze_intensity_sensitivity.py # -> outputs/sensitivity (figures + stats)

# §7.6 rho boundary and feasibility wall -> Table 8, Figs 5/6 (self-contained; uses shipped episodes/incumbents)
python scripts/run_rho_boundary_experiment.py

# §7.7 like-for-like external baselines -> Table 9 (DDPG checkpoint ships with the repo)
python scripts/evaluate_external_baselines.py \
  --config configs/default.yaml configs/env/formal_dynamic_stronger_v2.yaml \
           configs/solver/cp_repair_default.yaml configs/baselines/ddpg.yaml \
           configs/baselines/learned_rule_selector.yaml \
  --baselines heuristic_rh dispatching_mwkr dispatching_atc full_reoptimization daniel_local ddpg \
  --eval-episodes-dir outputs/episodes/brandimarte_heldout/episodes \
  --output-dir outputs/external_baselines/table9
```

> To just inspect the paper results, open the shipped `outputs/sensitivity/` (F1–F4 + summary/stat CSVs)
> and `outputs/rho_boundary/` (figures + summary CSVs).
>
> The **full pipeline from raw instances** (including event trajectories and supervised data for the
> learned baselines) is listed in [PAPER_MAP.md](PAPER_MAP.md) §1.

## Tests

```bash
python -m unittest discover -s tests
```

(Run inside an environment with `torch`/`ortools` installed; see [PAPER_MAP.md](PAPER_MAP.md) §4 for the test↔paper mapping.)
