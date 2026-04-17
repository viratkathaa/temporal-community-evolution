"""Deliverable 1 — download sx-mathoverflow and build quarterly snapshots.

Run:
    python notebooks/01_build_snapshots.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import load_edges
from src.snapshots import build_snapshots, save_snapshots, summary_table

OUT_TABLE = ROOT / "results" / "tables" / "01_snapshot_summary.csv"


def main() -> None:
    edges = load_edges()
    print(f"Loaded {len(edges):,} temporal edges.")
    print(f"Nodes: {edges[['src', 'dst']].stack().nunique():,}")
    print(f"Time span: {edges['datetime'].min()}  ->  {edges['datetime'].max()}")

    snapshots = build_snapshots(edges, freq="QS", directed=False, min_edges=100)
    print(f"\nBuilt {len(snapshots)} snapshots (quarterly).")

    save_snapshots(snapshots)

    table = summary_table(snapshots)
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_TABLE, index=False)

    print("\nSnapshot summary:")
    print(table.to_string(index=False))
    print(f"\nSaved table -> {OUT_TABLE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
