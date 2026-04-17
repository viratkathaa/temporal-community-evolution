"""Interactive dashboard for the temporal community project (Deliverable 5).

done by virat

Run:
    python dashboard/app.py
Then open http://127.0.0.1:8050 in a browser.

Prerequisites:
    Run notebooks/01_build_snapshots.py, 02_eda.py, 03_communities.py, and
    04_link_prediction.py first so the CSVs and pickle files exist.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

from src.snapshots import load_snapshots

# ---- Load cached artifacts --------------------------------------------------

SNAPSHOTS = load_snapshots()
with open(ROOT / "data" / "processed" / "communities.pkl", "rb") as f:
    COMMUNITIES = pickle.load(f)

METRICS = pd.read_csv(ROOT / "results" / "tables" / "02_metrics.csv", parse_dates=["start"])
COMM_SUMMARY = pd.read_csv(ROOT / "results" / "tables" / "03_communities_summary.csv", parse_dates=["start"])
EVENTS = pd.read_csv(ROOT / "results" / "tables" / "03_community_events.csv")
LP = pd.read_csv(ROOT / "results" / "tables" / "04_link_prediction.csv")

EVENT_ORDER = ["birth", "death", "growth", "shrinkage", "continue", "merge", "split"]


# ---- Helpers ----------------------------------------------------------------

def time_series_fig(df: pd.DataFrame, cols: list[str], title: str) -> go.Figure:
    fig = go.Figure()
    for col in cols:
        fig.add_trace(go.Scatter(x=df["start"], y=df[col], mode="lines+markers", name=col))
    fig.update_layout(title=title, template="plotly_white", height=380, margin=dict(l=40, r=20, t=40, b=40))
    return fig


def snapshot_graph_fig(snap_index: int, max_community_size: int = 300) -> go.Figure:
    """Render the top community of a snapshot as a node-link diagram.

    We cap rendered nodes at `max_community_size` so the browser stays snappy
    even when a snapshot has a 1000+ node largest community.
    """
    snap = SNAPSHOTS[snap_index]
    cs = COMMUNITIES[snap_index]
    if not cs.communities:
        return go.Figure().update_layout(title="No communities in this snapshot")

    # Pick the largest community.
    top_cid, top_nodes = max(cs.communities.items(), key=lambda kv: len(kv[1]))
    nodes = list(top_nodes)[:max_community_size]
    sub = snap.graph.subgraph(nodes).copy()

    pos = nx.spring_layout(sub, seed=42, k=1.0 / np.sqrt(max(sub.number_of_nodes(), 1)))

    edge_x, edge_y = [], []
    for u, v in sub.edges():
        edge_x += [pos[u][0], pos[v][0], None]
        edge_y += [pos[u][1], pos[v][1], None]

    node_x = [pos[n][0] for n in sub.nodes()]
    node_y = [pos[n][1] for n in sub.nodes()]
    deg = dict(sub.degree())
    size = [5 + 2 * np.log1p(deg[n]) for n in sub.nodes()]
    text = [f"user {n} (deg {deg[n]})" for n in sub.nodes()]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                             line=dict(width=0.4, color="#888"),
                             hoverinfo="none", showlegend=False))
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers",
                             marker=dict(size=size, color=[deg[n] for n in sub.nodes()],
                                         colorscale="Viridis", showscale=True,
                                         colorbar=dict(title="degree")),
                             text=text, hoverinfo="text", showlegend=False))
    fig.update_layout(
        title=f"Snapshot {snap_index} ({snap.start.date()} → {snap.end.date()}) — "
              f"largest community: {len(top_nodes)} nodes, showing {sub.number_of_nodes()}",
        template="plotly_white", height=560, margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return fig


def community_size_fig(snap_index: int) -> go.Figure:
    cs = COMMUNITIES[snap_index]
    sizes = sorted((len(v) for v in cs.communities.values()), reverse=True)
    fig = go.Figure(data=[go.Bar(x=list(range(len(sizes))), y=sizes)])
    fig.update_layout(title=f"Community sizes in snapshot {snap_index} ({len(sizes)} communities)",
                      xaxis_title="community rank", yaxis_title="size",
                      template="plotly_white", height=380, margin=dict(l=40, r=20, t=40, b=40))
    return fig


def event_stacked_fig() -> go.Figure:
    if EVENTS.empty:
        return go.Figure()
    idx_to_date = dict(zip(COMM_SUMMARY["snapshot_index"], COMM_SUMMARY["start"]))
    ev = EVENTS.copy()
    ev["date"] = ev["t_to"].map(idx_to_date)
    counts = (ev.groupby(["date", "event"]).size().unstack(fill_value=0)
              .reindex(columns=EVENT_ORDER, fill_value=0))
    counts.index = pd.to_datetime(counts.index)
    fig = go.Figure()
    for col in EVENT_ORDER:
        fig.add_trace(go.Bar(x=counts.index, y=counts[col], name=col))
    fig.update_layout(barmode="stack", template="plotly_white",
                      title="Community events per transition",
                      xaxis_title="snapshot start (t+1)", yaxis_title="event count",
                      height=420, margin=dict(l=40, r=20, t=40, b=40))
    return fig


def lp_fig() -> go.Figure:
    melted = LP.melt(id_vars="method", value_vars=["AUC", "AP"],
                     var_name="metric", value_name="score")
    fig = px.bar(melted, x="method", y="score", color="metric", barmode="group",
                 text="score", template="plotly_white",
                 title="Link prediction: baselines vs GraphSAGE")
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(yaxis=dict(range=[0.5, 1.0]), height=440,
                      margin=dict(l=40, r=20, t=40, b=40))
    return fig


# ---- App layout -------------------------------------------------------------

app = Dash(__name__)
app.title = "Temporal Community Evolution"

snapshot_marks = {i: str(SNAPSHOTS[i].start.date()) for i in range(0, len(SNAPSHOTS),
                                                                  max(1, len(SNAPSHOTS) // 6))}

app.layout = html.Div(
    style={"maxWidth": "1180px", "margin": "0 auto", "padding": "20px",
           "fontFamily": "system-ui, sans-serif"},
    children=[
        html.H1("Temporal Dynamics and Community Evolution in sx-mathoverflow"),
        html.P("Interactive companion to the Network Science group project. "
               "Use the tabs to explore the network over time."),
        dcc.Tabs(id="tabs", value="overview", children=[
            dcc.Tab(label="Network overview", value="overview"),
            dcc.Tab(label="Snapshot explorer", value="explorer"),
            dcc.Tab(label="Community evolution", value="communities"),
            dcc.Tab(label="Link prediction", value="lp"),
        ]),
        html.Div(id="tab-content", style={"marginTop": "16px"}),
    ],
)


@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab: str):
    if tab == "overview":
        return html.Div([
            dcc.Graph(figure=time_series_fig(METRICS, ["nodes", "edges"], "Network size")),
            dcc.Graph(figure=time_series_fig(METRICS, ["density"], "Density")),
            dcc.Graph(figure=time_series_fig(METRICS, ["avg_clustering", "transitivity"],
                                             "Clustering")),
            dcc.Graph(figure=time_series_fig(METRICS, ["lcc_diameter", "lcc_avg_path_length"],
                                             "Diameter and avg path length (LCC)")),
            dcc.Graph(figure=time_series_fig(METRICS, ["mean_betweenness", "max_betweenness"],
                                             "Betweenness centrality")),
            dcc.Graph(figure=time_series_fig(METRICS, ["assortativity"], "Degree assortativity")),
        ])

    if tab == "explorer":
        return html.Div([
            html.Label("Snapshot:"),
            dcc.Slider(id="snap-slider", min=0, max=len(SNAPSHOTS) - 1, step=1,
                       value=len(SNAPSHOTS) // 2, marks=snapshot_marks,
                       tooltip={"placement": "bottom", "always_visible": False}),
            dcc.Graph(id="snap-graph"),
            dcc.Graph(id="snap-comm-sizes"),
        ])

    if tab == "communities":
        return html.Div([
            dcc.Graph(figure=time_series_fig(COMM_SUMMARY, ["num_communities"], "# communities")),
            dcc.Graph(figure=time_series_fig(COMM_SUMMARY, ["modularity"], "Modularity")),
            dcc.Graph(figure=event_stacked_fig()),
        ])

    if tab == "lp":
        return html.Div([
            dcc.Graph(figure=lp_fig()),
            html.P("Evaluation: chronological split at the 85th-percentile edge timestamp. "
                   "Test set: 13,124 new future edges between nodes already present in the "
                   "training graph, paired with the same number of sampled non-edges."),
        ])

    return html.Div()


@app.callback(Output("snap-graph", "figure"), Input("snap-slider", "value"))
def update_snap_graph(idx: int):
    return snapshot_graph_fig(int(idx))


@app.callback(Output("snap-comm-sizes", "figure"), Input("snap-slider", "value"))
def update_snap_sizes(idx: int):
    return community_size_fig(int(idx))


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
