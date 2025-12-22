# python/data/fi2010.py
"""
FI-2010 dataset adapter (robust implementation).

Supported input formats (path points to a file):
1) .npz: contains X and optionally Y
   - X: (N, D) float
   - Y: (N, K) int labels for multiple horizons (optional)
2) .npy: contains X only
3) .txt / .csv: numeric table; first 4*L columns are LOB features; remaining columns may be labels

Expected per-row LOB layout by default (per level):
  [ask_p, ask_s, bid_p, bid_s] repeated for level 0..L-1
You can change layout via cfg in loaders (see loaders.py).

Output frame dict:
  {"bid_p": (L,), "ask_p": (L,), "bid_s": (L,), "ask_s": (L,), "trade_vol": float}
"""
from __future__ import annotations
from typing import Dict, Iterator, List, Optional, Tuple
import os
import numpy as np


# ---- helpers: file loading ----

def _load_matrix_any(path: str, delimiter: Optional[str] = None) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path, mmap_mode="r")
    if ext == ".npz":
        z = np.load(path, allow_pickle=False)
        # Prefer common keys
        if "X" in z:
            return z["X"]
        # fallback: first array
        keys = list(z.keys())
        if not keys:
            raise ValueError(f"Empty npz: {path}")
        return z[keys[0]]
    if ext in (".txt", ".csv"):
        if delimiter is None:
            delimiter = "," if ext == ".csv" else None  # None -> whitespace
        return np.loadtxt(path, delimiter=delimiter)
    raise ValueError(f"Unsupported FI-2010 file extension: {ext} ({path})")


def _load_labels_from_npz(path: str) -> Optional[np.ndarray]:
    ext = os.path.splitext(path)[1].lower()
    if ext != ".npz":
        return None
    z = np.load(path, allow_pickle=False)
    for k in ("Y", "y", "labels"):
        if k in z:
            return z[k]
    return None


# ---- LOB parsing ----

def _parse_lob_row(
    row: np.ndarray,
    L: int,
    layout: Tuple[str, str, str, str] = ("ask_p", "ask_s", "bid_p", "bid_s"),
) -> Dict:
    """
    Parse first 4*L values in `row` into bid/ask price/size arrays.
    Default assumes per level: ask_p, ask_s, bid_p, bid_s.
    """
    need = 4 * L
    if row.shape[0] < need:
        raise ValueError(f"Row dim too small: {row.shape[0]} < {need} for L={L}")

    # reshape (L,4)
    x = row[:need].reshape(L, 4)
    cols = {layout[i]: x[:, i].astype(np.float32) for i in range(4)}

    ask_p = cols["ask_p"]
    ask_s = cols["ask_s"]
    bid_p = cols["bid_p"]
    bid_s = cols["bid_s"]

    return {
        "bid_p": bid_p,
        "ask_p": ask_p,
        "bid_s": bid_s,
        "ask_s": ask_s,
        "trade_vol": 0.0,  # FI-2010 commonly doesn't include trade volume per row
    }


def fi2010_stream(
    path: str,
    L: int = 20,
    *,
    delimiter: Optional[str] = None,
    layout: Tuple[str, str, str, str] = ("ask_p", "ask_s", "bid_p", "bid_s"),
    start: int = 0,
    stop: Optional[int] = None,
) -> Iterator[Dict]:
    """
    Stream frames from FI-2010 file.
    Path may be .npz/.npy/.txt/.csv.
    """
    X = _load_matrix_any(path, delimiter=delimiter)
    # Ensure 2D
    if X.ndim != 2:
        raise ValueError(f"FI-2010 X must be 2D, got {X.shape} from {path}")

    n = X.shape[0]
    if stop is None or stop > n:
        stop = n

    for i in range(start, stop):
        row = np.asarray(X[i], dtype=np.float32)
        yield _parse_lob_row(row, L=L, layout=layout)


# ---- labels ----

def _mid_from_row(row: np.ndarray, L: int, layout: Tuple[str, str, str, str]) -> float:
    # level 0 (best)
    x = row[: 4 * L].reshape(L, 4)
    idx = {layout[i]: i for i in range(4)}
    ask_p0 = float(x[0, idx["ask_p"]])
    bid_p0 = float(x[0, idx["bid_p"]])
    return 0.5 * (ask_p0 + bid_p0)


def fi2010_labels_from_mids(mids: np.ndarray, horizons: List[int]) -> Dict[int, np.ndarray]:
    """
    Fallback label generation when FI-2010 file does NOT provide labels.
    3-class: 0=down,1=flat,2=up using a quantile threshold on |Δmid|.
    """
    labels = {}
    mids = mids.astype(np.float32)
    for h in horizons:
        h = int(h)
        fut = np.roll(mids, -h)
        d = fut - mids
        d[-h:] = 0.0
        thr = np.quantile(np.abs(d[:-h]), 0.6) if len(d) > h else 0.0
        y = np.zeros_like(d, dtype=np.int64)
        y[d > thr] = 2
        y[np.abs(d) <= thr] = 1
        y[d < -thr] = 0
        labels[h] = y
    return labels


def fi2010_labels(
    X: np.ndarray,
    horizons: List[int],
    *,
    provided_Y: Optional[np.ndarray] = None,
    provided_horizons: Optional[List[int]] = None,
    layout: Tuple[str, str, str, str] = ("ask_p", "ask_s", "bid_p", "bid_s"),
    L: int = 20,
) -> Dict[int, np.ndarray]:
    """
    Prefer labels from file if provided_Y exists.
    Otherwise generate labels from mid prices.
    """
    horizons = [int(h) for h in horizons]

    if provided_Y is not None:
        Y = np.asarray(provided_Y)
        if Y.ndim == 1:
            # single horizon labels
            if len(horizons) != 1:
                raise ValueError("provided_Y is 1D but horizons has multiple values")
            return {horizons[0]: Y.astype(np.int64)}

        if Y.ndim != 2:
            raise ValueError(f"provided_Y must be 1D or 2D, got {Y.shape}")

        # map columns to horizons
        if provided_horizons is None:
            # common FI-2010 multi-label sets in the wild often include 5 horizons.
            # If K==5, assume order [10, 20, 50, 100, 200] unless user overrides.
            K = Y.shape[1]
            if K == 5:
                provided_horizons = [10, 20, 50, 100, 200]
            else:
                # fallback: assume last len(horizons) match the requested horizons order
                provided_horizons = horizons[-K:] if K <= len(horizons) else horizons

        col_map = {int(h): j for j, h in enumerate(provided_horizons)}
        out: Dict[int, np.ndarray] = {}
        for h in horizons:
            if h in col_map and col_map[h] < Y.shape[1]:
                out[h] = Y[:, col_map[h]].astype(np.int64)
            else:
                # missing -> fallback computed later
                out[h] = None  # type: ignore

        # fill missing using mids
        if any(v is None for v in out.values()):
            mids = np.asarray([_mid_from_row(np.asarray(X[i]), L=L, layout=layout) for i in range(X.shape[0])], dtype=np.float32)
            computed = fi2010_labels_from_mids(mids, horizons)
            for h in horizons:
                if out[h] is None:
                    out[h] = computed[h]
        return out

    # no provided Y -> compute from mids
    mids = np.asarray([_mid_from_row(np.asarray(X[i]), L=L, layout=layout) for i in range(X.shape[0])], dtype=np.float32)
    return fi2010_labels_from_mids(mids, horizons)
