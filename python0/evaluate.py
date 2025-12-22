from __future__ import annotations
import os, argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from utils import load_yaml, ensure_dir, set_seed
from data.features import load_norm_stats
from train import build_synth_windows
from model.lit_cvg import LiTCVG_Lite

def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--norm", default="artifacts/norm_stats.json")
    ap.add_argument("--outdir", default="artifacts/plots")
    ap.add_argument("--N", type=int, default=2048)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    ensure_dir(args.outdir)
    set_seed(int(cfg["train"]["seed"]) + 123)

    norm = load_norm_stats(args.norm)
    windows, labels = build_synth_windows(cfg, norm, N=args.N, seed=int(cfg["train"]["seed"]) + 7)

    # model load
    mcfg = cfg["model"]
    model = LiTCVG_Lite(
        T=int(cfg["data"]["T"]), L=int(cfg["data"]["L"]), C=int(cfg["data"]["C_raw"]),
        pT=int(mcfg["pT"]), pL=int(mcfg["pL"]),
        d_model=int(mcfg["d_model"]), depth=int(mcfg["depth"]),
        g_dim=int(mcfg.get("g_dim", 16)), horizons=cfg["data"]["horizons"]
    )
    sd = torch.load(args.ckpt, map_location="cpu")["model"]
    model.load_state_dict(sd, strict=True)
    model.eval()

    x = torch.from_numpy(windows).float()
    with torch.no_grad():
        out = model(x)
    horizons = [int(h) for h in cfg["data"]["horizons"]]

    # simple accuracy per horizon
    for h in horizons:
        key = f"logits_{h}"
        p = out[key].numpy()
        pred = p.argmax(axis=-1)
        acc = (pred == labels[h]).mean()
        print(f"h={h} acc={acc:.3f}")

    # a tiny plot: class-2 probability histogram for h=10
    key = f"logits_{horizons[0]}"
    prob = softmax_np(out[key].numpy(), axis=-1)[:, 2]
    plt.figure()
    plt.hist(prob, bins=50)
    plt.title(f"p(up) histogram (h={horizons[0]})")
    outp = os.path.join(args.outdir, "prob_hist.png")
    plt.savefig(outp, dpi=150, bbox_inches="tight")
    print(f"saved plot -> {outp}")

if __name__ == "__main__":
    main()
