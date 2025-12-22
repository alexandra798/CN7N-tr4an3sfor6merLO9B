# python/data/loaders.py
from __future__ import annotations
import os
import numpy as np
from typing import Dict, Iterator, List, Optional, Tuple
import random

from fi2010 import fi2010_stream, _load_matrix_any, _load_labels_from_npz

def synth_l2_stream(L: int = 20, n: int = 200000, seed: int = 42) -> Iterator[Dict]:
    rng = random.Random(seed)
    mid = 100.0
    spread = 0.01
    bid_s = np.ones((L,), dtype=np.float32) * 10
    ask_s = np.ones((L,), dtype=np.float32) * 10

    for _ in range(n):
        mid += rng.gauss(0, 0.002)
        spread = max(0.005, min(0.05, spread + rng.gauss(0, 0.0005)))
        bid0 = mid - spread / 2
        ask0 = mid + spread / 2
        tick = 0.01
        bid_p = np.asarray([bid0 - i*tick for i in range(L)], dtype=np.float32)
        ask_p = np.asarray([ask0 + i*tick for i in range(L)], dtype=np.float32)
        bid_s = np.clip(bid_s * (1.0 + rng.gauss(0, 0.02)) + rng.gauss(0, 0.5), 1.0, 200.0).astype(np.float32)
        ask_s = np.clip(ask_s * (1.0 + rng.gauss(0, 0.02)) + rng.gauss(0, 0.5), 1.0, 200.0).astype(np.float32)
        trade_vol = max(0.0, rng.gauss(0.5, 0.2))
        yield {"bid_p": bid_p, "ask_p": ask_p, "bid_s": bid_s, "ask_s": ask_s, "trade_vol": float(trade_vol)}

def make_synth_labels(mids: np.ndarray, horizons: List[int]) -> Dict[int, np.ndarray]:
    labels = {}
    for h in horizons:
        fut = np.roll(mids, -h)
        d = fut - mids
        d[-h:] = 0.0
        thr = np.quantile(np.abs(d[:-h]), 0.6) if len(d) > h else 0.0
        y = np.zeros_like(d, dtype=np.int64)
        y[d > thr] = 2
        y[np.abs(d) <= thr] = 1
        y[d < -thr] = 0
        labels[int(h)] = y
    return labels


# --------- FI-2010 helpers (simple + configurable) ---------

def fi2010_resolve_file(cfg: dict, split: str) -> str:
    """
    Resolve FI-2010 file path from cfg.
    You can configure:
      data:
        dataset: "FI-2010"
        fi2010:
          path_train: ".../train.npz"
          path_val: ".../val.npz"
          path_test: ".../test.npz"
    or:
      fi2010:
        path: ".../somefile.npz"  (then split ignored)
    """
    d = cfg["data"].get("fi2010", {}) if "data" in cfg else cfg.get("fi2010", {})
    if "path" in d:
        return d["path"]
    key = f"path_{split}"
    if key in d:
        return d[key]
    # fallback: common names inside root
    root = d.get("root", "")
    if not root:
        raise ValueError("FI-2010: please set data.fi2010.path or data.fi2010.root/path_{split}")
    for cand in (f"{split}.npz", f"{split}.npy", f"{split}.txt", f"{split}.csv"):
        p = os.path.join(root, cand)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Cannot resolve FI-2010 split file under root={root} for split={split}")


def load_fi2010_XY(cfg: dict, split: str) -> Tuple[np.ndarray, Optional[np.ndarray], List[int]]:
    """
    Load X and optional Y from FI-2010 storage.
    Also returns provided_horizons if user config supplies it.
    """
    path = fi2010_resolve_file(cfg, split)
    d = cfg["data"].get("fi2010", {})
    delimiter = d.get("delimiter", None)

    X = _load_matrix_any(path, delimiter=delimiter)
    Y = _load_labels_from_npz(path)

    provided_horizons = d.get("label_horizons", None)
    if provided_horizons is not None:
        provided_horizons = [int(h) for h in provided_horizons]

    return X, Y, provided_horizons


def get_stream_builder(cfg: dict, split: str):
    """
    Returns (stream_iter, labels_dict_builder_or_ready)
    """
    dataset = str(cfg["data"].get("dataset", "SYNTH")).upper()
    L = int(cfg["data"]["L"])
    horizons = [int(h) for h in cfg["data"]["horizons"]]

    if dataset == "SYNTH":
        seed = int(cfg["train"]["seed"]) + (1 if split == "train" else 7)
        n = int(cfg["data"].get("synth_n", 250000))
        stream = synth_l2_stream(L=L, n=n, seed=seed)
        return stream, None  # labels computed later from mids in train.py

    if dataset in ("FI-2010", "FI2010", "FI_2010"):
        d = cfg["data"].get("fi2010", {})
        # default layout per level: [ask_p, ask_s, bid_p, bid_s]
        layout = tuple(d.get("layout", ["ask_p", "ask_s", "bid_p", "bid_s"]))  # type: ignore
        delimiter = d.get("delimiter", None)

        path = fi2010_resolve_file(cfg, split)
        stream = fi2010_stream(path, L=L, delimiter=delimiter, layout=layout)  # yields dict frames

        # labels will be built from X/Y in train.py (need full alignment)
        return stream, {"path": path, "layout": layout, "delimiter": delimiter}

    raise ValueError(f"Unknown dataset: {dataset}")
