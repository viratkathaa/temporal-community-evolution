# Temporal Dynamics and Community Evolution in Online Social Networks

Network Science course group project.

## Overview

We investigate how communities form, evolve, merge, and dissolve over time in an online
social network. The project combines classical network science metrics with modern graph
machine learning to uncover patterns invisible in static snapshots.

**Dataset:** [`sx-mathoverflow`](https://snap.stanford.edu/data/sx-mathoverflow.html) from the
Stanford SNAP collection — 24,818 users, 506,550 timestamped interactions
(answers, comments, mentions) spanning ~6 years.

## Deliverables

| # | Deliverable | Output |
|---|---|---|
| 1 | Dataset curation & temporal snapshots | `src/snapshots.py`, `data/processed/snapshots.pkl` |
| 2 | Exploratory network analysis across time | `notebooks/02_eda.py`, figures in `results/figures/` |
| 3 | Dynamic community detection & event tracking | `notebooks/03_communities.py`, event timeline |
| 4 | Link prediction with GNN vs Jaccard / Adamic-Adar | `notebooks/04_link_prediction.py`, metrics table |
| 5 | Interactive dashboard + final report | `dashboard/app.py`, `report/final_report.pdf` |

## Repo layout

```
.
├── data/
│   ├── raw/                 # downloaded SNAP file
│   └── processed/           # time-sliced graph snapshots
├── src/                     # reusable modules (loader, metrics, community, gnn)
├── notebooks/               # one runnable script per deliverable
├── dashboard/               # Plotly Dash app (deliverable 5)
├── results/
│   ├── figures/
│   └── tables/
├── report/                  # final written report
├── requirements.txt
└── README.md
```

## Setup

```bash
# create & activate a venv (Windows bash)
python -m venv .venv
source .venv/Scripts/activate

# core deps
pip install -r requirements.txt

# PyTorch + PyG with CUDA 12.1 (adjust to your CUDA version, RTX 5060 supports 12.x)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric
```

## Reproducing the pipeline

```bash
python notebooks/01_build_snapshots.py   # downloads + slices the dataset
python notebooks/02_eda.py               # computes metrics across snapshots
python notebooks/03_communities.py       # dynamic Louvain + event tracking
python notebooks/04_link_prediction.py   # GraphSAGE vs heuristic baselines
python dashboard/app.py                  # launches the interactive dashboard
```

## Team

- Virat (2022578)
- Dhruv Kumar (2023202)
- Ruchir Bhatowa (2022419)
