# python/train.py
from __future__ import annotations
import os, argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utils import load_yaml, set_seed, ensure_dir
from data.loaders import (
    synth_l2_stream, make_synth_labels,
    get_stream_builder, load_fi2010_XY
)
from data.features import make_feature_frame, update_norm_stats, save_norm_stats, load_norm_stats, normalize, NormStats
from data.windows import RingWindow
from model.lit_cvg import LiTCVG_Lite
from model.losses import multi_horizon_ce
from data.fi2010 import fi2010_labels

ART = "artifacts"

class WindowDataset(Dataset):
    def __init__(self, windows: np.ndarray, labels: dict, horizons: list):
        self.x = windows.astype(np.float32)  # (N,T,L,C)
        self.labels = {int(h): labels[int(h)].astype(np.int64) for h in horizons}
        self.horizons = [int(h) for h in horizons]

    def __len__(self): return self.x.shape[0]

    def __getitem__(self, i):
        x = self.x[i]
        y = {h: self.labels[h][i] for h in self.horizons}
        return x, y

def build_windows_from_stream(cfg: dict, norm: NormStats, N: int, split: str = "train"):
    """
    Build (windows, labels) for either SYNTH or FI-2010.
    windows: (N, T, L, C)
    labels: {h: (N,)}
    """
    T = int(cfg["data"]["T"]); L = int(cfg["data"]["L"]); C = int(cfg["data"]["C_raw"])
    horizons = [int(h) for h in cfg["data"]["horizons"]]
    dataset = str(cfg["data"].get("dataset", "SYNTH")).upper()

    stream, meta = get_stream_builder(cfg, split)

    # If FI-2010, we also load X/Y to get labels in a stable way.
    labels_full = None
    if dataset in ("FI-2010", "FI2010", "FI_2010"):
        X, Y, provided_horizons = load_fi2010_XY(cfg, split)
        layout = meta["layout"] if meta else ("ask_p","ask_s","bid_p","bid_s")
        labels_full = fi2010_labels(
            X=X,
            horizons=horizons,
            provided_Y=Y,
            provided_horizons=provided_horizons,
            layout=layout,
            L=L,
        )
        # We will align label index with stream index (row index).
        # Note: stream yields frames starting at row 0.

    # For SYNTH, we build labels from mids once we have enough stream in memory.
    # To avoid overcomplication, we keep SYNTH behavior the same as before (materialize).
    if dataset == "SYNTH":
        # Materialize just enough frames for N windows + max horizon + warmup
        stream_list = list(synth_l2_stream(
            L=L,
            n=N + max(horizons) + 2,
            seed=int(cfg["train"]["seed"]) + (1 if split == "train" else 7),
        ))
        mids = np.asarray([0.5*(s["bid_p"][0] + s["ask_p"][0]) for s in stream_list], dtype=np.float32)
        labels_full = make_synth_labels(mids, horizons)
        stream = iter(stream_list)

    win = RingWindow(T=T, L=L, C=C)
    # prime prev
    prev = next(stream)
    cum_vol = 0.0

    xs = []
    ys = {h: [] for h in horizons}

    # stream index: prev corresponds to index 0, first cur is index 1
    t = 0
    for cur in stream:
        t += 1
        frame, cum_vol = make_feature_frame(prev, cur, L=L, cum_vol_prev=cum_vol)
        frame = normalize(frame, norm)
        win.push(frame)

        view = win.view()
        if view is not None:
            if len(xs) >= N:
                break
            xs.append(view[0])  # (T,L,C)
            for h in horizons:
                ys[h].append(int(labels_full[h][t]))  # label aligned to current time index
        prev = cur

    windows = np.stack(xs, axis=0)
    labels = {h: np.asarray(ys[h], dtype=np.int64) for h in horizons}
    return windows, labels

def make_norm(cfg: dict, out_path: str, split: str = "train"):
    L = int(cfg["data"]["L"])
    dataset = str(cfg["data"].get("dataset", "SYNTH")).upper()

    if dataset == "SYNTH":
        stream = synth_l2_stream(L=L, n=25000, seed=int(cfg["train"]["seed"]))
        prev = next(stream)
        cum_vol = 0.0
        def frames_iter():
            nonlocal prev, cum_vol
            for cur in stream:
                frame, cum_vol = make_feature_frame(prev, cur, L=L, cum_vol_prev=cum_vol)
                prev = cur
                yield frame
        stats = update_norm_stats(frames_iter(), warmup=20000)
        save_norm_stats(out_path, stats)
        return stats

    if dataset in ("FI-2010", "FI2010", "FI_2010"):
        # stream warmup
        stream, _ = get_stream_builder(cfg, split)
        prev = next(stream)
        cum_vol = 0.0
        def frames_iter():
            nonlocal prev, cum_vol
            for cur in stream:
                frame, cum_vol = make_feature_frame(prev, cur, L=L, cum_vol_prev=cum_vol)
                prev = cur
                yield frame
        stats = update_norm_stats(frames_iter(), warmup=20000)
        save_norm_stats(out_path, stats)
        return stats

    raise ValueError(f"Unknown dataset for norm: {dataset}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--make-norm", action="store_true")
    ap.add_argument("--norm", default=os.path.join(ART, "norm_stats.json"))
    ap.add_argument("--ckpt", default=os.path.join(ART, "ckpt.pt"))
    ap.add_argument("--split", default="train", choices=["train","val","test"])
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    ensure_dir(ART)
    set_seed(int(cfg["train"]["seed"]))

    if args.make_norm or (not os.path.exists(args.norm)):
        norm = make_norm(cfg, args.norm, split=args.split)
    else:
        norm = load_norm_stats(args.norm)

    steps_per_epoch = int(cfg["data"].get("steps_per_epoch", 200))
    bs = int(cfg["data"]["batch_size"])
    N = steps_per_epoch * bs

    windows, labels = build_windows_from_stream(cfg, norm, N=N, split=args.split)
    ds = WindowDataset(windows, labels, cfg["data"]["horizons"])
    dl = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=0)

    mcfg = cfg["model"]
    model = LiTCVG_Lite(
        T=int(cfg["data"]["T"]), L=int(cfg["data"]["L"]), C=int(cfg["data"]["C_raw"]),
        pT=int(mcfg["pT"]), pL=int(mcfg["pL"]),
        d_model=int(mcfg["d_model"]), depth=int(mcfg["depth"]),
        g_dim=int(mcfg.get("g_dim", 16)), horizons=cfg["data"]["horizons"]
    )
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["optim"]["lr"]), weight_decay=float(cfg["optim"]["weight_decay"]))

    weights = {int(cfg["data"]["horizons"][0]): 1.0}
    for h in cfg["data"]["horizons"][1:]:
        weights[int(h)] = 0.6 if int(h) == int(cfg["data"]["horizons"][1]) else 0.4

    model.train()
    for epoch in range(int(cfg["train"]["epochs"])):
        total = 0.0
        for x, y in dl:
            x = x.float()
            yt = {int(h): torch.as_tensor(y[int(h)], dtype=torch.long) for h in y}
            out = model(x)
            loss = multi_horizon_ce(out, yt, weights)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
        print(f"epoch={epoch} loss={total/len(dl):.4f}")

    torch.save({"model": model.state_dict(), "cfg": cfg}, args.ckpt)
    print(f"saved ckpt -> {args.ckpt}")

if __name__ == "__main__":
    main()
