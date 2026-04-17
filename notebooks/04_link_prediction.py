"""Deliverable 4 — Link prediction: baselines vs GraphSAGE.

Writes:
    - results/tables/04_link_prediction.csv
    - results/figures/04_lp_metrics.png

Run:
    python notebooks/04_link_prediction.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

from src.data_loader import load_edges
from src.link_prediction import build_split, score_heuristics, train_gnn, eval_gnn

FIG_DIR = ROOT / "results" / "figures"
TBL_DIR = ROOT / "results" / "tables"

sns.set_theme(context="notebook", style="whitegrid")


def main() -> None:
    edges = load_edges()
    print(f"Loaded {len(edges):,} edges.")

    print("Building chronological split (train=85% of edges by time) ...")
    dataset = build_split(edges, train_frac=0.85, seed=0)
    print(f"  train graph: {dataset.train_graph.number_of_nodes():,} nodes, "
          f"{dataset.train_graph.number_of_edges():,} edges")
    print(f"  test positives (new future edges on known nodes): {len(dataset.test_pos):,}")
    print(f"  test negatives: {len(dataset.test_neg):,}")

    print("\n--- Heuristic baselines ---")
    base_df = score_heuristics(dataset)
    print(base_df.to_string(index=False))

    print("\n--- GraphSAGE ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model, data, _ = train_gnn(dataset, device=device, epochs=150, lr=5e-3)
    gnn_result = eval_gnn(model, data, dataset)
    print(gnn_result)

    all_df = pd.concat([base_df, pd.DataFrame([gnn_result])], ignore_index=True)
    out_csv = TBL_DIR / "04_link_prediction.csv"
    all_df.to_csv(out_csv, index=False)
    print(f"\nSaved metrics -> {out_csv.relative_to(ROOT)}")
    print(all_df.to_string(index=False))

    # Plot side-by-side AUC / AP.
    melted = all_df.melt(id_vars="method", value_vars=["AUC", "AP"],
                         var_name="metric", value_name="score")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.barplot(data=melted, x="method", y="score", hue="metric", ax=ax)
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Link prediction: baselines vs GraphSAGE")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20)
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{h:.3f}", (p.get_x() + p.get_width() / 2, h),
                        ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_lp_metrics.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
