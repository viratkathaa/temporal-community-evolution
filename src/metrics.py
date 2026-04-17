"""Per-snapshot network metrics for temporal EDA.

done by dhruv
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import networkx as nx
import numpy as np
import pandas as pd

from .snapshots import Snapshot


@dataclass
class SnapshotMetrics:
    index: int
    start: pd.Timestamp
    nodes: int
    edges: int
    density: float
    avg_degree: float
    max_degree: int
    avg_clustering: float
    transitivity: float
    num_components: int
    lcc_size: int
    lcc_fraction: float
    lcc_diameter: int | float
    lcc_avg_path_length: float
    assortativity: float
    mean_betweenness: float
    max_betweenness: float


def compute_metrics(
    snapshot: Snapshot,
    betweenness_k: int = 200,
    diameter_sample: int = 500,
    seed: int = 0,
) -> SnapshotMetrics:
    """Compute a fixed panel of metrics for one snapshot.

    Expensive metrics (betweenness, diameter) are approximated with sampling
    so the whole pipeline runs in minutes on a laptop.
    """
    g = snapshot.graph
    n, m = g.number_of_nodes(), g.number_of_edges()

    degrees = np.fromiter((d for _, d in g.degree()), dtype=int, count=n) if n else np.array([0])
    avg_deg = float(degrees.mean()) if n else 0.0
    max_deg = int(degrees.max()) if n else 0

    avg_clust = nx.average_clustering(g) if n else 0.0
    trans = nx.transitivity(g) if n else 0.0

    components = list(nx.connected_components(g)) if n else []
    num_cc = len(components)
    if components:
        lcc_nodes = max(components, key=len)
        lcc_size = len(lcc_nodes)
        lcc = g.subgraph(lcc_nodes).copy()
    else:
        lcc_size = 0
        lcc = g

    lcc_frac = lcc_size / n if n else 0.0

    # Diameter / avg path length on the LCC. For larger LCCs, approximate via
    # BFS from a sample of source nodes.
    if lcc_size >= 2:
        if lcc_size <= diameter_sample:
            lcc_diameter = nx.diameter(lcc)
            lcc_apl = nx.average_shortest_path_length(lcc)
        else:
            rng = np.random.default_rng(seed)
            nodes_list = list(lcc.nodes())
            sample = rng.choice(nodes_list, size=diameter_sample, replace=False)
            ecc = []
            path_sums = []
            path_counts = []
            for s in sample:
                lengths = nx.single_source_shortest_path_length(lcc, s)
                if len(lengths) > 1:
                    ecc.append(max(lengths.values()))
                    path_sums.append(sum(lengths.values()))
                    path_counts.append(len(lengths) - 1)
            lcc_diameter = int(max(ecc)) if ecc else 0
            lcc_apl = (
                float(sum(path_sums) / sum(path_counts)) if sum(path_counts) > 0 else 0.0
            )
    else:
        lcc_diameter = 0
        lcc_apl = 0.0

    try:
        assort = nx.degree_assortativity_coefficient(g)
        if np.isnan(assort):
            assort = 0.0
    except Exception:
        assort = 0.0

    # Betweenness: exact on small graphs, k-sampled approximation on larger ones.
    if n == 0:
        bc_vals = np.array([0.0])
    elif n <= betweenness_k:
        bc = nx.betweenness_centrality(g, normalized=True)
        bc_vals = np.fromiter(bc.values(), dtype=float)
    else:
        bc = nx.betweenness_centrality(g, k=betweenness_k, normalized=True, seed=seed)
        bc_vals = np.fromiter(bc.values(), dtype=float)

    return SnapshotMetrics(
        index=snapshot.index,
        start=snapshot.start,
        nodes=n,
        edges=m,
        density=nx.density(g) if n > 1 else 0.0,
        avg_degree=avg_deg,
        max_degree=max_deg,
        avg_clustering=avg_clust,
        transitivity=trans,
        num_components=num_cc,
        lcc_size=lcc_size,
        lcc_fraction=lcc_frac,
        lcc_diameter=lcc_diameter,
        lcc_avg_path_length=lcc_apl,
        assortativity=float(assort),
        mean_betweenness=float(bc_vals.mean()),
        max_betweenness=float(bc_vals.max()),
    )


def metrics_table(snapshots: list[Snapshot], **kwargs) -> pd.DataFrame:
    rows = [asdict(compute_metrics(s, **kwargs)) for s in snapshots]
    return pd.DataFrame(rows)


def degree_distribution(snapshot: Snapshot) -> np.ndarray:
    g = snapshot.graph
    return np.fromiter((d for _, d in g.degree()), dtype=int, count=g.number_of_nodes())
