"""Community detection per snapshot and temporal event tracking.

Approach:
    1. Run Louvain modularity optimization on each snapshot independently.
    2. Match communities across consecutive snapshots via Jaccard overlap of
       their node sets, classifying events as birth, death, growth, shrinkage,
       merge, or split.

This is the classic "matching" framework used in temporal community mining
(Palla et al. 2007; Greene et al. 2010). More sophisticated approaches (dynamic
Louvain with smoothing, temporal stochastic block models) are mentioned in the
report as extensions.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import community as community_louvain  # python-louvain
import networkx as nx
import pandas as pd

from .snapshots import Snapshot


@dataclass
class CommunitySnap:
    """Communities detected in a single snapshot."""
    snapshot_index: int
    start: pd.Timestamp
    partition: dict[int, int]  # node -> community id (local to this snapshot)
    modularity: float
    communities: dict[int, set[int]] = field(default_factory=dict)

    @classmethod
    def from_partition(cls, snap: Snapshot, partition: dict[int, int], modularity: float) -> "CommunitySnap":
        communities: dict[int, set[int]] = defaultdict(set)
        for node, cid in partition.items():
            communities[cid].add(node)
        return cls(
            snapshot_index=snap.index,
            start=snap.start,
            partition=partition,
            modularity=modularity,
            communities=dict(communities),
        )


def detect_louvain(snapshot: Snapshot, resolution: float = 1.0, seed: int = 0) -> CommunitySnap:
    g = snapshot.graph
    if g.number_of_nodes() == 0:
        return CommunitySnap(snapshot.index, snapshot.start, {}, 0.0, {})
    partition = community_louvain.best_partition(g, resolution=resolution, random_state=seed)
    mod = community_louvain.modularity(partition, g)
    return CommunitySnap.from_partition(snapshot, partition, mod)


def detect_all(snapshots: list[Snapshot], min_size: int = 5, **kwargs) -> list[CommunitySnap]:
    """Run Louvain on every snapshot; drop communities smaller than `min_size`."""
    out = []
    for s in snapshots:
        cs = detect_louvain(s, **kwargs)
        cs.communities = {cid: nodes for cid, nodes in cs.communities.items() if len(nodes) >= min_size}
        out.append(cs)
    return out


# ---- Event tracking ---------------------------------------------------------

@dataclass
class CommunityEvent:
    t_from: int | None
    t_to: int | None
    event: str  # birth | death | growth | shrinkage | continue | merge | split
    source_cids: tuple[int, ...]
    target_cids: tuple[int, ...]
    size_from: int
    size_to: int
    jaccard: float


def _jaccard(a: set[int], b: set[int]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def _containment(a: set[int], b: set[int]) -> float:
    """|A ∩ B| / min(|A|, |B|). Better than Jaccard when snapshots have
    disjoint active users (common in quarterly temporal graphs)."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def track_events(
    community_snaps: list[CommunitySnap],
    match_threshold: float = 0.3,
    grow_ratio: float = 1.15,
    shrink_ratio: float = 0.85,
    use_containment: bool = True,
) -> pd.DataFrame:
    """Match communities across consecutive snapshots and emit events.

    Args:
        match_threshold: minimum overlap score for a pair to count as the same
            community.
        grow_ratio / shrink_ratio: size change thresholds for growth/shrinkage.
        use_containment: if True, score pairs by |A ∩ B| / min(|A|, |B|) instead
            of Jaccard. This is more appropriate when snapshots contain largely
            different active users (each quarter filters to users active in that
            quarter), so two "same" communities can share a small fraction of
            members even when they are the same clique of long-term users.
    """
    score_fn = _containment if use_containment else _jaccard
    events: list[CommunityEvent] = []

    for t in range(len(community_snaps) - 1):
        src = community_snaps[t]
        dst = community_snaps[t + 1]

        # Pairwise Jaccard matrix between src and dst communities.
        pairs: list[tuple[int, int, float]] = []
        for cid_a, nodes_a in src.communities.items():
            for cid_b, nodes_b in dst.communities.items():
                j = score_fn(nodes_a, nodes_b)
                if j >= match_threshold:
                    pairs.append((cid_a, cid_b, j))

        # Count how many targets each source matches and vice versa.
        src_matches: dict[int, list[tuple[int, float]]] = defaultdict(list)
        dst_matches: dict[int, list[tuple[int, float]]] = defaultdict(list)
        for a, b, j in pairs:
            src_matches[a].append((b, j))
            dst_matches[b].append((a, j))

        # Classify source communities.
        for cid_a, nodes_a in src.communities.items():
            matches = src_matches.get(cid_a, [])
            if not matches:
                events.append(CommunityEvent(
                    t_from=src.snapshot_index, t_to=dst.snapshot_index,
                    event="death", source_cids=(cid_a,), target_cids=(),
                    size_from=len(nodes_a), size_to=0, jaccard=0.0,
                ))
                continue
            if len(matches) >= 2:
                events.append(CommunityEvent(
                    t_from=src.snapshot_index, t_to=dst.snapshot_index,
                    event="split", source_cids=(cid_a,),
                    target_cids=tuple(b for b, _ in matches),
                    size_from=len(nodes_a),
                    size_to=sum(len(dst.communities[b]) for b, _ in matches),
                    jaccard=max(j for _, j in matches),
                ))
                continue
            # Single match -> continue / grow / shrink (or merge, handled via dst below).
            cid_b, j = matches[0]
            size_b = len(dst.communities[cid_b])
            ratio = size_b / len(nodes_a)
            if len(dst_matches[cid_b]) >= 2:
                continue  # merge event is emitted from the dst side, skip here
            if ratio >= grow_ratio:
                ev = "growth"
            elif ratio <= shrink_ratio:
                ev = "shrinkage"
            else:
                ev = "continue"
            events.append(CommunityEvent(
                t_from=src.snapshot_index, t_to=dst.snapshot_index,
                event=ev, source_cids=(cid_a,), target_cids=(cid_b,),
                size_from=len(nodes_a), size_to=size_b, jaccard=j,
            ))

        # Merges and births from the dst side.
        for cid_b, nodes_b in dst.communities.items():
            matches = dst_matches.get(cid_b, [])
            if not matches:
                events.append(CommunityEvent(
                    t_from=src.snapshot_index, t_to=dst.snapshot_index,
                    event="birth", source_cids=(), target_cids=(cid_b,),
                    size_from=0, size_to=len(nodes_b), jaccard=0.0,
                ))
                continue
            if len(matches) >= 2:
                events.append(CommunityEvent(
                    t_from=src.snapshot_index, t_to=dst.snapshot_index,
                    event="merge",
                    source_cids=tuple(a for a, _ in matches),
                    target_cids=(cid_b,),
                    size_from=sum(len(src.communities[a]) for a, _ in matches),
                    size_to=len(nodes_b),
                    jaccard=max(j for _, j in matches),
                ))

    return pd.DataFrame([e.__dict__ for e in events])


def communities_summary(community_snaps: list[CommunitySnap]) -> pd.DataFrame:
    rows = []
    for cs in community_snaps:
        sizes = [len(v) for v in cs.communities.values()]
        rows.append({
            "snapshot_index": cs.snapshot_index,
            "start": cs.start,
            "num_communities": len(cs.communities),
            "modularity": cs.modularity,
            "largest_community": max(sizes) if sizes else 0,
            "median_community": int(pd.Series(sizes).median()) if sizes else 0,
        })
    return pd.DataFrame(rows)
