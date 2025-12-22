# python/parity_dump.py
from __future__ import annotations
import os, argparse
import numpy as np
import torch

from utils import load_yaml, ensure_dir, set_seed
from data.features import load_norm_stats
from train import build_windows_from_stream
from model.lit_cvg import LiTCVG_Lite

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--norm", default="artifacts/norm_stats.json")
    ap.add_argument("--outdir", default="artifacts/golden")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--split", default="val", choices=["train","val","test"])
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    ensure_dir(args.outdir)
    set_seed(int(cfg["train"]["seed"]) + 999)

    norm = load_norm_stats(args.norm)
    windows, _ = build_windows_from_stream(cfg, norm, N=args.n, split=args.split)

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

    np.save(os.path.join(args.outdir, "inputs.npy"), windows.astype(np.float32))
    for h in cfg["data"]["horizons"]:
        key = f"logits_{int(h)}"
        np.save(os.path.join(args.outdir, f"{key}.npy"), out[key].numpy().astype(np.float32))

    print(f"saved golden inputs/outputs to {args.outdir}")

if __name__ == "__main__":
    main()
