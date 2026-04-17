"""Deliverable 3 — Dynamic community detection + temporal event tracking.

Writes:
    - results/tables/03_communities_summary.csv
    - results/tables/03_community_events.csv
    - results/figures/03_*.png

Run:
    python notebooks/03_communities.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.snapshots import load_snapshots
from src.community import detect_all, track_events, communities_summary

FIG_DIR = ROOT / "results" / "figures"
TBL_DIR = ROOT / "results" / "tables"
PROC_DIR = ROOT / "data" / "processed"

sns.set_theme(context="notebook", style="whitegrid")

EVENT_ORDER = ["birth", "death", "growth", "shrinkage", "continue", "merge", "split"]


def plot_community_counts(summary: pd.DataFrame) -> None:
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.plot(summary["start"], summary["num_communities"], marker="o", color="C0", label="# communities")
    ax1.set_ylabel("# communities", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax2 = ax1.twinx()
    ax2.plot(summary["start"], summary["modularity"], marker="s", color="C3", label="modularity")
    ax2.set_ylabel("modularity", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")
    ax1.set_title("Community count and modularity over time")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_communities_over_time.png", dpi=140)
    plt.close(fig)


def plot_event_timeline(events: pd.DataFrame, summary: pd.DataFrame) -> None:
    if events.empty:
        return
    # Map snapshot index to start date for x axis.
    idx_to_date = dict(zip(summary["snapshot_index"], summary["start"]))
    events = events.copy()
    events["date"] = events["t_to"].map(idx_to_date)

    counts = (events.groupby(["date", "event"]).size().unstack(fill_value=0)
              .reindex(columns=EVENT_ORDER, fill_value=0))
    # Use string labels on x to avoid pandas period-freq plotting issues.
    counts.index = [d.strftime("%Y-%m") for d in counts.index]
    ax = counts.plot(kind="bar", stacked=True, figsize=(11, 5), width=0.9,
                     colormap="tab10")
    ax.set_title("Community events per transition (t -> t+1)")
    ax.set_xlabel("snapshot start (t+1)")
    ax.set_ylabel("event count")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="event", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_event_timeline.png", dpi=140)
    plt.close()


def plot_event_totals(events: pd.DataFrame) -> None:
    if events.empty:
        return
    totals = events["event"].value_counts().reindex(EVENT_ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 4))
    totals.plot(kind="bar", ax=ax, color=sns.color_palette("tab10", n_colors=len(EVENT_ORDER)))
    ax.set_title("Total community events across all transitions")
    ax.set_ylabel("count")
    ax.set_xlabel("")
    for p in ax.patches:
        ax.annotate(str(int(p.get_height())), (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_event_totals.png", dpi=140)
    plt.close(fig)


def main() -> None:
    snapshots = load_snapshots()
    print(f"Loaded {len(snapshots)} snapshots. Running Louvain on each ...")
    cs_list = detect_all(snapshots, min_size=5, resolution=1.0, seed=0)

    # Save the per-snapshot partitions for reuse by the dashboard.
    with open(PROC_DIR / "communities.pkl", "wb") as f:
        pickle.dump(cs_list, f, protocol=pickle.HIGHEST_PROTOCOL)

    summary = communities_summary(cs_list)
    summary.to_csv(TBL_DIR / "03_communities_summary.csv", index=False)
    print("\nPer-snapshot community summary:")
    print(summary.to_string(index=False))

    events = track_events(cs_list, match_threshold=0.1, grow_ratio=1.15,
                          shrink_ratio=0.85, use_containment=True)
    events.to_csv(TBL_DIR / "03_community_events.csv", index=False)
    print(f"\nTracked {len(events)} events across {len(cs_list) - 1} transitions.")
    print(events["event"].value_counts().reindex(EVENT_ORDER, fill_value=0).to_string())

    plot_community_counts(summary)
    plot_event_timeline(events, summary)
    plot_event_totals(events)
    print(f"\nSaved figures -> {FIG_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
