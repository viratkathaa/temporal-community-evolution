"""Deliverable 4 — Link prediction: heuristic baselines + GraphSAGE.

Task:
    Given edges that occurred before a cutoff time `t_split`, predict which
    *new* edges will appear in the window [t_split, t_end].

Baselines (NetworkX):
    - Common Neighbors
    - Jaccard Coefficient
    - Adamic-Adar
    - Preferential Attachment

GNN:
    - 2-layer GraphSAGE trained on the training graph with self-supervised
      binary link prediction (BCE loss over positive training edges vs random
      negatives). Evaluated on held-out future edges.
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import negative_sampling, to_undirected


@dataclass
class LPDataset:
    """Holds a temporal link-prediction split."""
    train_graph: nx.Graph
    test_pos: list[tuple[int, int]]
    test_neg: list[tuple[int, int]]
    node_list: list[int]                   # consistent ordering -> indices
    node_to_idx: dict[int, int]


def build_split(
    edges: pd.DataFrame,
    train_frac: float = 0.85,
    num_test_neg: int | None = None,
    seed: int = 0,
) -> LPDataset:
    """Build a chronological train/test split.

    Train: all edges with ts before the `train_frac` quantile.
    Test positives: edges after the cutoff whose endpoints both appeared in the
      training graph AND that are NOT present in the training graph (truly new
      links). This is the standard "future link prediction" setup.
    Test negatives: same count as positives, sampled from node pairs that are
      not connected in the training graph.
    """
    rng = np.random.default_rng(seed)
    edges = edges.sort_values("ts").reset_index(drop=True)
    t_split = edges["ts"].quantile(train_frac)

    train_df = edges[edges["ts"] <= t_split]
    test_df = edges[edges["ts"] > t_split]

    g_train = nx.Graph()
    g_train.add_edges_from(train_df[["src", "dst"]].itertuples(index=False, name=None))

    known_nodes = set(g_train.nodes())
    # Keep only edges between known nodes and that are not already in g_train.
    test_pos = []
    seen = set()
    for u, v in test_df[["src", "dst"]].itertuples(index=False, name=None):
        if u == v or u not in known_nodes or v not in known_nodes:
            continue
        key = (min(u, v), max(u, v))
        if key in seen or g_train.has_edge(u, v):
            continue
        seen.add(key)
        test_pos.append(key)

    if num_test_neg is None:
        num_test_neg = len(test_pos)

    # Sample negative pairs uniformly until we have enough.
    node_arr = np.fromiter(known_nodes, dtype=np.int64)
    test_neg = []
    existing = {(min(u, v), max(u, v)) for u, v in g_train.edges()} | set(test_pos)
    tries = 0
    while len(test_neg) < num_test_neg and tries < num_test_neg * 50:
        u, v = rng.choice(node_arr, 2, replace=False)
        key = (int(min(u, v)), int(max(u, v)))
        if key in existing:
            tries += 1
            continue
        test_neg.append(key)
        existing.add(key)
        tries += 1

    node_list = sorted(g_train.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    return LPDataset(g_train, test_pos, test_neg, node_list, node_to_idx)


# ---- Heuristic baselines ----------------------------------------------------

_HEURISTICS = {
    "common_neighbors": lambda g, u, v: len(list(nx.common_neighbors(g, u, v))),
    "jaccard": lambda g, u, v: _pair_score(nx.jaccard_coefficient(g, [(u, v)])),
    "adamic_adar": lambda g, u, v: _pair_score(nx.adamic_adar_index(g, [(u, v)])),
    "preferential_attachment": lambda g, u, v: _pair_score(
        nx.preferential_attachment(g, [(u, v)])
    ),
}


def _pair_score(gen) -> float:
    for _, _, s in gen:
        return float(s)
    return 0.0


def score_heuristics(dataset: LPDataset) -> pd.DataFrame:
    """Score baselines on test positives + negatives, return AUC + AP per method."""
    g = dataset.train_graph
    rows = []
    y_true = np.concatenate([np.ones(len(dataset.test_pos)),
                             np.zeros(len(dataset.test_neg))])
    pairs = dataset.test_pos + dataset.test_neg
    for name, fn in _HEURISTICS.items():
        scores = np.array([fn(g, u, v) for u, v in pairs])
        # NaN/inf guard for Adamic-Adar on isolated pairs.
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        rows.append({
            "method": name,
            "AUC": roc_auc_score(y_true, scores),
            "AP": average_precision_score(y_true, scores),
        })
    return pd.DataFrame(rows)


# ---- GraphSAGE --------------------------------------------------------------

class GraphSAGENet(torch.nn.Module):
    """GraphSAGE with a learnable node-embedding table concatenated to the
    structural features. Standard transductive link-prediction recipe when
    nodes carry no rich attributes: the embedding gives the model per-node
    capacity and the convs mix neighbor information into it. Dot-product
    decoder with no output normalization, BCE loss."""

    def __init__(
        self,
        num_nodes: int,
        feat_dim: int,
        emb_dim: int = 64,
        hid_dim: int = 128,
        out_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.node_emb = torch.nn.Embedding(num_nodes, emb_dim)
        torch.nn.init.normal_(self.node_emb.weight, std=0.01)
        in_dim = feat_dim + emb_dim
        self.conv1 = SAGEConv(in_dim, hid_dim)
        self.conv2 = SAGEConv(hid_dim, out_dim)
        self.dropout = dropout

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        idx = torch.arange(x.size(0), device=x.device)
        h = torch.cat([x, self.node_emb(idx)], dim=-1)
        h = F.relu(self.conv1(h, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        return h

    def score_pairs(self, z: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        src, dst = edges
        return (z[src] * z[dst]).sum(dim=-1)


def _structural_features(g: nx.Graph, node_list: list[int]) -> np.ndarray:
    """3-D per-node features: log-degree, clustering, PageRank.
    All deterministic from the training graph, so safe at test time.
    """
    deg = dict(g.degree())
    clust = nx.clustering(g)
    pr = nx.pagerank(g, alpha=0.85, max_iter=200, tol=1e-6)
    feats = np.zeros((len(node_list), 3), dtype=np.float32)
    for i, n in enumerate(node_list):
        feats[i, 0] = np.log1p(deg.get(n, 0))
        feats[i, 1] = clust.get(n, 0.0)
        feats[i, 2] = pr.get(n, 0.0)
    # Z-score normalize each feature column.
    mu = feats.mean(axis=0, keepdims=True)
    sigma = feats.std(axis=0, keepdims=True) + 1e-8
    return (feats - mu) / sigma


def build_pyg_data(dataset: LPDataset, device: torch.device) -> Data:
    node_to_idx = dataset.node_to_idx
    edges = np.array(
        [(node_to_idx[u], node_to_idx[v]) for u, v in dataset.train_graph.edges()],
        dtype=np.int64,
    ).T
    edge_index = torch.from_numpy(edges).to(device)
    edge_index = to_undirected(edge_index, num_nodes=len(dataset.node_list))

    x_np = _structural_features(dataset.train_graph, dataset.node_list)
    x = torch.from_numpy(x_np).to(device)
    return Data(x=x, edge_index=edge_index, num_nodes=len(dataset.node_list))


def train_gnn(
    dataset: LPDataset,
    device: torch.device | None = None,
    epochs: int = 150,
    lr: float = 5e-3,
    emb_dim: int = 32,
    hid_dim: int = 128,
    out_dim: int = 64,
    seed: int = 0,
    verbose: bool = True,
) -> tuple[GraphSAGENet, Data, list[dict]]:
    """Train GraphSAGE with BCE on (training edge, random negative) pairs."""
    torch.manual_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = build_pyg_data(dataset, device)
    model = GraphSAGENet(
        num_nodes=data.num_nodes,
        feat_dim=data.num_node_features,
        emb_dim=emb_dim,
        hid_dim=hid_dim,
        out_dim=out_dim,
    ).to(device)
    # Normalize features so concatenation with small-init embeddings is balanced.
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        z = model.encode(data.x, data.edge_index)

        pos_edge = data.edge_index
        neg_edge = negative_sampling(
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=pos_edge.size(1),
            method="sparse",
        )
        pos_logits = model.score_pairs(z, pos_edge)
        neg_logits = model.score_pairs(z, neg_edge)
        logits = torch.cat([pos_logits, neg_logits])
        labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)])
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        opt.step()

        if verbose and (epoch == 1 or epoch % 20 == 0 or epoch == epochs):
            print(f"  epoch {epoch:3d}  loss={loss.item():.4f}")
        history.append({"epoch": epoch, "loss": loss.item()})
    return model, data, history


@torch.no_grad()
def eval_gnn(model: GraphSAGENet, data: Data, dataset: LPDataset) -> dict:
    model.eval()
    z = model.encode(data.x, data.edge_index).cpu().numpy()
    node_to_idx = dataset.node_to_idx

    def score_list(pairs):
        idx_pairs = np.array([(node_to_idx[u], node_to_idx[v]) for u, v in pairs])
        return (z[idx_pairs[:, 0]] * z[idx_pairs[:, 1]]).sum(axis=-1)

    pos_scores = score_list(dataset.test_pos)
    neg_scores = score_list(dataset.test_neg)
    y_true = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
    y_score = np.concatenate([pos_scores, neg_scores])
    return {
        "method": "graphsage",
        "AUC": roc_auc_score(y_true, y_score),
        "AP": average_precision_score(y_true, y_score),
    }
