from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_overall_comparison(df: pd.DataFrame, metric: str, output_path: str | Path) -> None:
    plt.figure(figsize=(8, 4))
    sns.barplot(data=df, x="method", y=metric)
    plt.xticks(rotation=20)
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_anytime_curve(df: pd.DataFrame, metric: str, output_path: str | Path) -> None:
    plt.figure(figsize=(8, 4))
    sns.lineplot(data=df, x="event_time", y=metric, hue="method")
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_tradeoff(df: pd.DataFrame, output_path: str | Path) -> None:
    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=df, x="instability", y="makespan", hue="method")
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
