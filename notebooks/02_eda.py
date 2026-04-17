"""Deliverable 2 — Exploratory network analysis across time.

done by dhruv

Computes per-snapshot metrics and writes:
    - results/tables/02_metrics.csv
    - results/figures/02_*.png

Run:
    python notebooks/02_eda.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from src.snapshots import load_snapshots
from src.metrics import compute_metrics, degree_distribution

FIG_DIR = ROOT / "results" / "figures"
TBL_DIR = ROOT / "results" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(context="notebook", style="whitegrid")


def _time_series_plot(df: pd.DataFrame, cols: list[tuple[str, str]], title: str, fname: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for col, label in cols:
        ax.plot(df["start"], df[col], marker="o", label=label, linewidth=1.6)
    ax.set_xlabel("Snapshot start")
    ax.set_title(title)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=140)
    plt.close(fig)


def plot_basic_size(df: pd.DataFrame) -> None:
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.plot(df["start"], df["nodes"], color="C0", marker="o", label="nodes")
    ax1.set_ylabel("nodes", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax2 = ax1.twinx()
    ax2.plot(df["start"], df["edges"], color="C3", marker="s", label="edges")
    ax2.set_ylabel("edges", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")
    ax1.set_title("Network size over time")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_size.png", dpi=140)
    plt.close(fig)


def plot_degree_distributions(snapshots, indices: list[int]) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for i in indices:
        if i >= len(snapshots):
            continue
        s = snapshots[i]
        degs = degree_distribution(s)
        degs = degs[degs > 0]
        if len(degs) == 0:
            continue
        # Complementary CDF on log-log axes — cleaner than a histogram for heavy tails.
        sorted_d = np.sort(degs)
        ccdf = 1.0 - np.arange(len(sorted_d)) / len(sorted_d)
        ax.loglog(sorted_d, ccdf, marker=".", linestyle="none",
                  label=f"{s.start.date()} (n={s.graph.number_of_nodes()})")
    ax.set_xlabel("degree k")
    ax.set_ylabel("P(K >= k)")
    ax.set_title("Degree distribution across time (CCDF, log-log)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_degree_ccdf.png", dpi=140)
    plt.close(fig)


def main() -> None:
    snapshots = load_snapshots()
    print(f"Loaded {len(snapshots)} snapshots.")

    rows = []
    for s in tqdm(snapshots, desc="metrics"):
        rows.append(compute_metrics(s).__dict__)
    df = pd.DataFrame(rows)
    df["start"] = pd.to_datetime(df["start"])

    out_csv = TBL_DIR / "02_metrics.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved metrics table -> {out_csv.relative_to(ROOT)}")
    print(df.to_string(index=False))

    plot_basic_size(df)
    _time_series_plot(df, [("density", "density")],
                      "Density over time", "02_density.png")
    _time_series_plot(df,
                      [("avg_clustering", "avg. clustering"),
                       ("transitivity", "transitivity")],
                      "Clustering over time", "02_clustering.png")
    _time_series_plot(df,
                      [("lcc_fraction", "largest CC / N")],
                      "Largest connected component (fraction of nodes)",
                      "02_lcc_fraction.png")
    _time_series_plot(df,
                      [("lcc_diameter", "diameter"),
                       ("lcc_avg_path_length", "avg path length")],
                      "Diameter and average path length (on LCC)",
                      "02_path_length.png")
    _time_series_plot(df,
                      [("mean_betweenness", "mean betweenness"),
                       ("max_betweenness", "max betweenness")],
                      "Betweenness centrality over time",
                      "02_betweenness.png")
    _time_series_plot(df, [("assortativity", "degree assortativity")],
                      "Degree assortativity over time", "02_assortativity.png")

    # Degree CCDFs at early / mid / late snapshots for the report.
    indices = [0, len(snapshots) // 2, len(snapshots) - 1]
    plot_degree_distributions(snapshots, indices)

    print(f"\nSaved {len(list(FIG_DIR.glob('02_*.png')))} figures -> {FIG_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
