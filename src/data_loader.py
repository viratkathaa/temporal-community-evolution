"""Download and parse the SNAP sx-mathoverflow temporal dataset.

done by virat
"""
from __future__ import annotations

import gzip
import shutil
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

SNAP_URL = "https://snap.stanford.edu/data/sx-mathoverflow.txt.gz"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW_GZ = RAW_DIR / "sx-mathoverflow.txt.gz"
RAW_TXT = RAW_DIR / "sx-mathoverflow.txt"


def download(force: bool = False) -> Path:
    """Download the SNAP .txt.gz file if not already present. Returns the extracted .txt path."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if RAW_TXT.exists() and not force:
        return RAW_TXT

    if not RAW_GZ.exists() or force:
        print(f"Downloading {SNAP_URL} ...")
        with requests.get(SNAP_URL, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(RAW_GZ, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True, desc="download"
            ) as bar:
                for chunk in r.iter_content(chunk_size=1 << 15):
                    f.write(chunk)
                    bar.update(len(chunk))

    print("Extracting ...")
    with gzip.open(RAW_GZ, "rb") as f_in, open(RAW_TXT, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    return RAW_TXT


def load_edges() -> pd.DataFrame:
    """Load edges as a DataFrame with columns [src, dst, ts] sorted by timestamp.

    The SNAP sx-mathoverflow file has space-separated lines: `SRC DST UNIXTS`.
    It aggregates three interaction types (a2q, c2q, c2a). We treat them as one
    interaction graph here; separating them is a useful extension later.
    """
    path = download()
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["src", "dst", "ts"],
        dtype={"src": "int64", "dst": "int64", "ts": "int64"},
        engine="c",
    )
    df = df.sort_values("ts", kind="mergesort").reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    return df


if __name__ == "__main__":
    edges = load_edges()
    print(f"Loaded {len(edges):,} edges")
    print(f"Unique nodes: {pd.concat([edges['src'], edges['dst']]).nunique():,}")
    print(f"Time span: {edges['datetime'].min()}  ->  {edges['datetime'].max()}")
    print(edges.head())
