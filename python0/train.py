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
from data.features import (
    make_feature_frame, update_norm_stats, save_norm_stats, 
    load_norm_stats, normalize, NormStats, FeatureConfig
)
from data.windows import RingWindow
from model.lit_cvg import LiTCVG_Lite
from model.losses import multi_horizon_ce, adaptive_weights
from data.fi2010 import fi2010_labels

ART = "artifacts"


class WindowDataset(Dataset):
    def __init__(self, windows: np.ndarray, labels: dict, horizons: list):
        self.x = windows.astype(np.float32)  # (N,T,L,C)
        self.labels = {int(h): labels[int(h)].astype(np.int64) for h in horizons}
        self.horizons = [int(h) for h in horizons]

    def __len__(self): 
        return self.x.shape[0]

    def __getitem__(self, i):
        x = self.x[i]
        y = {h: self.labels[h][i] for h in self.horizons}
        return x, y


def build_windows_from_stream(cfg: dict, norm: NormStats, N: int, 
                              split: str = "train",
                              feature_config: FeatureConfig = None):
    """Build (windows, labels) for either SYNTH or FI-2010.
    
    Args:
        cfg: Configuration dict
        norm: Normalization statistics
        N: Number of windows to generate
        split: 'train', 'val', or 'test'
        feature_config: Feature configuration (if None, use default)
    
    Returns:
        windows: (N, T, L, C_raw)
        labels: {h: (N,)}
    """
    T = int(cfg["data"]["T"])
    L = int(cfg["data"]["L"])
    horizons = [int(h) for h in cfg["data"]["horizons"]]
    dataset = str(cfg["data"].get("dataset", "SYNTH")).upper()
    
    # Feature configuration
    if feature_config is None:
        feature_config = FeatureConfig(
            include_price=cfg["data"].get("include_price", False),
            include_volume_profile=cfg["data"].get("include_volume_profile", False),
            ofi_levels=cfg["data"].get("ofi_levels", 1)
        )
    
    C = feature_config.get_channel_count()
    
    stream, meta = get_stream_builder(cfg, split)

    # Load labels for FI-2010
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

    # For SYNTH, materialize and generate labels
    if dataset == "SYNTH":
        stream_list = list(synth_l2_stream(
            L=L,
            n=N + max(horizons) + 2,
            seed=int(cfg["train"]["seed"]) + (1 if split == "train" else 7),
        ))
        mids = np.asarray([0.5*(s["bid_p"][0] + s["ask_p"][0]) for s in stream_list], dtype=np.float32)
        labels_full = make_synth_labels(mids, horizons)
        stream = iter(stream_list)

    win = RingWindow(T=T, L=L, C=C)
    prev = next(stream)
    cum_vol = 0.0

    xs = []
    ys = {h: [] for h in horizons}

    # Build windows
    t = 0
    for cur in stream:
        t += 1
        frame, cum_vol = make_feature_frame(
            prev, cur, L=L, cum_vol_prev=cum_vol, config=feature_config
        )
        frame = normalize(frame, norm)
        win.push(frame)

        view = win.view()
        if view is not None:
            if len(xs) >= N:
                break
            xs.append(view[0])  # (T,L,C)
            for h in horizons:
                ys[h].append(int(labels_full[h][t]))
        prev = cur

    windows = np.stack(xs, axis=0)
    labels = {h: np.asarray(ys[h], dtype=np.int64) for h in horizons}
    
    return windows, labels


def make_norm(cfg: dict, out_path: str, split: str = "train", 
              feature_config: FeatureConfig = None):
    """Compute normalization statistics."""
    L = int(cfg["data"]["L"])
    dataset = str(cfg["data"].get("dataset", "SYNTH")).upper()
    
    if feature_config is None:
        feature_config = FeatureConfig(
            include_price=cfg["data"].get("include_price", False),
            include_volume_profile=cfg["data"].get("include_volume_profile", False)
        )

    if dataset == "SYNTH":
        stream = synth_l2_stream(L=L, n=25000, seed=int(cfg["train"]["seed"]))
        prev = next(stream)
        cum_vol = 0.0
        def frames_iter():
            nonlocal prev, cum_vol
            for cur in stream:
                frame, cum_vol = make_feature_frame(
                    prev, cur, L=L, cum_vol_prev=cum_vol, config=feature_config
                )
                prev = cur
                yield frame
        stats = update_norm_stats(frames_iter(), warmup=20000)
        save_norm_stats(out_path, stats)
        return stats

    if dataset in ("FI-2010", "FI2010", "FI_2010"):
        stream, _ = get_stream_builder(cfg, split)
        prev = next(stream)
        cum_vol = 0.0
        def frames_iter():
            nonlocal prev, cum_vol
            for cur in stream:
                frame, cum_vol = make_feature_frame(
                    prev, cur, L=L, cum_vol_prev=cum_vol, config=feature_config
                )
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
    
    # Feature configuration
    feature_config = FeatureConfig(
        include_price=cfg["data"].get("include_price", False),
        include_volume_profile=cfg["data"].get("include_volume_profile", False),
        ofi_levels=cfg["data"].get("ofi_levels", 1)
    )
    
    # Update C_raw in config based on feature config
    C_raw = feature_config.get_channel_count()
    cfg["data"]["C_raw"] = C_raw
    
    print(f"📊 Feature Configuration:")
    print(f"   Include price: {feature_config.include_price}")
    print(f"   Include volume profile: {feature_config.include_volume_profile}")
    print(f"   OFI levels: {feature_config.ofi_levels}")
    print(f"   Total channels (C_raw): {C_raw}\n")

    # Normalization statistics
    if args.make_norm or (not os.path.exists(args.norm)):
        print("📈 Computing normalization statistics...")
        norm = make_norm(cfg, args.norm, split=args.split, feature_config=feature_config)
    else:
        print(f"📂 Loading normalization statistics from {args.norm}")
        norm = load_norm_stats(args.norm)
    
    # Verify norm stats match C_raw
    if norm.mean.shape[0] != C_raw:
        raise ValueError(
            f"Norm stats channel count mismatch: "
            f"loaded={norm.mean.shape[0]}, expected={C_raw}. "
            f"Run with --make-norm to regenerate."
        )

    # Build dataset
    steps_per_epoch = int(cfg["data"].get("steps_per_epoch", 200))
    bs = int(cfg["data"]["batch_size"])
    N = steps_per_epoch * bs

    print(f"🔨 Building training dataset ({N} windows)...")
    windows, labels = build_windows_from_stream(
        cfg, norm, N=N, split=args.split, feature_config=feature_config
    )
    print(f"   Windows shape: {windows.shape}")
    print(f"   Labels: {list(labels.keys())}\n")
    
    ds = WindowDataset(windows, labels, cfg["data"]["horizons"])
    dl = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=0)

    # Build model
    mcfg = cfg["model"]
    model = LiTCVG_Lite(
        T=int(cfg["data"]["T"]), 
        L=int(cfg["data"]["L"]), 
        C=int(C_raw),  # Use computed C_raw
        pT=int(mcfg["pT"]), 
        pL=int(mcfg["pL"]),
        d_model=int(mcfg["d_model"]), 
        depth=int(mcfg["depth"]),
        g_dim=int(mcfg.get("g_dim", 16)), 
        horizons=cfg["data"]["horizons"]
    )
    
    print(f"🧠 Model: LiTCVG_Lite")
    print(f"   Params: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Optimizer
    opt = torch.optim.AdamW(
        model.parameters(), 
        lr=float(cfg["optim"]["lr"]), 
        weight_decay=float(cfg["optim"]["weight_decay"])
    )

    # Loss weights (adaptive exponential decay)
    weight_strategy = cfg["train"].get("weight_strategy", "exponential")
    decay = cfg["train"].get("weight_decay", 0.015)
    
    if weight_strategy == "adaptive":
        weights = adaptive_weights(
            cfg["data"]["horizons"], 
            decay=decay, 
            strategy="exponential"
        )
    else:
        # Legacy weights
        weights = {int(cfg["data"]["horizons"][0]): 1.0}
        for h in cfg["data"]["horizons"][1:]:
            weights[int(h)] = 0.6 if int(h) == int(cfg["data"]["horizons"][1]) else 0.4
    
    print(f"⚖️  Loss weights ({weight_strategy}):")
    for h, w in weights.items():
        print(f"   Horizon {h:3d}: {w:.4f}")
    print()

    # Training loop
    model.train()
    best_loss = float('inf')
    
    for epoch in range(int(cfg["train"]["epochs"])):
        total_loss = 0.0
        horizon_losses = {h: 0.0 for h in cfg["data"]["horizons"]}
        
        for batch_idx, (x, y) in enumerate(dl):
            x = x.float()
            yt = {int(h): torch.as_tensor(y[int(h)], dtype=torch.long) for h in y}
            
            # Forward pass
            out = model(x)
            loss = multi_horizon_ce(out, yt, weights, return_details=False)
            
            # Backward pass
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            total_loss += float(loss.item())
            
            # Per-horizon losses (for monitoring)
            with torch.no_grad():
                for h in cfg["data"]["horizons"]:
                    key = f"logits_{int(h)}"
                    h_loss = torch.nn.functional.cross_entropy(
                        out[key], yt[int(h)]
                    )
                    horizon_losses[int(h)] += float(h_loss.item())
        
        avg_loss = total_loss / len(dl)
        
        # Print epoch summary
        print(f"Epoch {epoch+1:2d}/{cfg['train']['epochs']} | Loss: {avg_loss:.4f}", end="")
        for h in cfg["data"]["horizons"]:
            h_avg = horizon_losses[int(h)] / len(dl)
            print(f" | h{h}={h_avg:.4f}", end="")
        print()
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "model": model.state_dict(), 
                "cfg": cfg,
                "feature_config": {
                    "include_price": feature_config.include_price,
                    "include_volume_profile": feature_config.include_volume_profile,
                    "ofi_levels": feature_config.ofi_levels,
                    "C_raw": C_raw
                }
            }, args.ckpt)
            print(f"   ✅ Saved best checkpoint (loss={best_loss:.4f})")
    
    print(f"\n✅ Training complete! Best loss: {best_loss:.4f}")
    print(f"📁 Checkpoint saved to {args.ckpt}")


if __name__ == "__main__":
    main()