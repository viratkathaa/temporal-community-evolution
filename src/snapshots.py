"""Build time-sliced graph snapshots from a temporal edge list.

done by virat
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import pandas as pd

PROC_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
SNAPSHOT_FILE = PROC_DIR / "snapshots.pkl"


@dataclass
class Snapshot:
    """A single time window of the temporal graph."""

    index: int
    start: pd.Timestamp
    end: pd.Timestamp
    graph: nx.Graph

    def summary(self) -> dict:
        g = self.graph
        return {
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "nodes": g.number_of_nodes(),
            "edges": g.number_of_edges(),
            "density": nx.density(g) if g.number_of_nodes() > 1 else 0.0,
        }


def build_snapshots(
    edges: pd.DataFrame,
    freq: str = "QS",
    directed: bool = False,
    min_edges: int = 100,
) -> list[Snapshot]:
    """Slice the temporal edge list into snapshots of the given pandas freq.

    Args:
        edges: DataFrame with columns [src, dst, ts, datetime] sorted by ts.
        freq: pandas offset alias. "QS" = quarter start (≈3 months). Use "MS"
            for monthly, "YS" for yearly. Quarterly gives ~24 snapshots over
            the 6-year sx-mathoverflow span — a good tradeoff.
        directed: whether to build DiGraphs. Most temporal community work uses
            undirected graphs; flip this on if you care about answer direction.
        min_edges: skip snapshots with fewer than this many edges (early-year
            windows on this dataset can be sparse).
    """
    if "datetime" not in edges.columns:
        edges = edges.copy()
        edges["datetime"] = pd.to_datetime(edges["ts"], unit="s", utc=True)

    # Bin each edge into a period, then group.
    periods = edges["datetime"].dt.to_period(_freq_to_period(freq))
    snapshots: list[Snapshot] = []
    graph_cls = nx.DiGraph if directed else nx.Graph

    for idx, (period, chunk) in enumerate(edges.groupby(periods, sort=True)):
        if len(chunk) < min_edges:
            continue
        g = graph_cls()
        # Use weight = number of interactions between the pair in the window.
        pair_counts = (
            chunk.groupby(["src", "dst"]).size().reset_index(name="weight")
        )
        g.add_weighted_edges_from(
            pair_counts[["src", "dst", "weight"]].itertuples(index=False, name=None)
        )
        snapshots.append(
            Snapshot(
                index=idx,
                start=period.start_time.tz_localize("UTC"),
                end=period.end_time.tz_localize("UTC"),
                graph=g,
            )
        )

    # Reindex sequentially after min_edges filtering.
    for new_idx, s in enumerate(snapshots):
        s.index = new_idx
    return snapshots


def _freq_to_period(freq: str) -> str:
    """Map pandas offset alias to the compatible period alias for to_period."""
    mapping = {"QS": "Q", "MS": "M", "YS": "Y", "AS": "Y", "W": "W", "D": "D"}
    return mapping.get(freq, freq)


def save_snapshots(snapshots: list[Snapshot], path: Path = SNAPSHOT_FILE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(snapshots, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_snapshots(path: Path = SNAPSHOT_FILE) -> list[Snapshot]:
    with open(path, "rb") as f:
        return pickle.load(f)


def summary_table(snapshots: list[Snapshot]) -> pd.DataFrame:
    return pd.DataFrame([s.summary() for s in snapshots])
