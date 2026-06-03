from __future__ import annotations

from itertools import combinations

import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon


def wilcoxon_pairwise(df: pd.DataFrame, metric: str, method_col: str = "method", seed_col: str = "seed") -> pd.DataFrame:
    methods = sorted(df[method_col].unique())
    rows = []
    for a, b in combinations(methods, 2):
        pivot = df[df[method_col].isin([a, b])].pivot_table(index=seed_col, columns=method_col, values=metric, aggfunc="mean").dropna()
        if pivot.shape[0] < 2:
            rows.append(
                {
                    "method_a": a,
                    "method_b": b,
                    "statistic": None,
                    "p_value": None,
                    "num_pairs": int(pivot.shape[0]),
                    "note": "insufficient_paired_observations",
                }
            )
            continue
        diffs = pivot[a] - pivot[b]
        if (diffs == 0).all():
            rows.append(
                {
                    "method_a": a,
                    "method_b": b,
                    "statistic": 0.0,
                    "p_value": 1.0,
                    "num_pairs": int(pivot.shape[0]),
                    "note": "identical_samples",
                }
            )
            continue
        stat, p = wilcoxon(pivot[a], pivot[b])
        rows.append(
            {
                "method_a": a,
                "method_b": b,
                "statistic": stat,
                "p_value": p,
                "num_pairs": int(pivot.shape[0]),
                "note": "",
            }
        )
    return pd.DataFrame(rows)


def friedman_holm(df: pd.DataFrame, metric: str, method_col: str = "method", block_col: str = "seed") -> tuple[dict, pd.DataFrame]:
    pivot = df.pivot_table(index=block_col, columns=method_col, values=metric, aggfunc="mean").dropna()
    methods = list(pivot.columns)
    if len(methods) < 2 or pivot.shape[0] < 2:
        result = {
            "statistic": None,
            "p_value": None,
            "num_blocks": int(pivot.shape[0]),
            "num_methods": int(len(methods)),
            "note": "insufficient_observations_for_friedman",
        }
        rank_df = pd.DataFrame(
            {
                "method": methods,
                "avg_rank": [None for _ in methods],
                "holm_threshold": [None for _ in methods],
            }
        )
        return result, rank_df

    samples = [pivot[m].values for m in methods]
    stat, p = friedmanchisquare(*samples)

    ranks = pivot.apply(lambda row: rankdata(row.values, method="average"), axis=1, result_type="expand")
    avg_ranks = ranks.mean(axis=0).values
    rank_df = pd.DataFrame({"method": methods, "avg_rank": avg_ranks}).sort_values("avg_rank")
    m = len(methods)
    rank_df["holm_threshold"] = [0.05 / (m - i) for i in range(m)]
    return {"statistic": stat, "p_value": p, "num_blocks": int(pivot.shape[0]), "num_methods": int(len(methods)), "note": ""}, rank_df
