from __future__ import annotations
import os, argparse
import torch
from utils import load_yaml, ensure_dir
from model.lit_cvg import LiTCVG_Lite

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="artifacts/model.onnx")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    ensure_dir(os.path.dirname(args.out))

    model = LiTCVG_Lite(
        T=int(cfg["T"]), L=int(cfg["L"]), C=int(cfg["C_raw"]),
        pT=int(cfg["pT"]), pL=int(cfg["pL"]),
        d_model=int(cfg["d_model"]), depth=int(cfg["depth"]),
        g_dim=int(cfg.get("g_dim", 16)), horizons=cfg["horizons"]
    )
    sd = torch.load(args.ckpt, map_location="cpu")["model"]
    model.load_state_dict(sd, strict=True)
    model.eval()

    dummy = torch.zeros((1, int(cfg["T"]), int(cfg["L"]), int(cfg["C_raw"])), dtype=torch.float32)
    out_names = [f"logits_{int(h)}" for h in cfg["horizons"]]

    dynamic_axes = cfg.get("dynamic_axes", {})
    dyn = {"input_lob": {0: "B"}}
    if isinstance(dynamic_axes, dict) and "input_lob" in dynamic_axes:
        dyn["input_lob"] = {int(k): str(v) for k,v in dynamic_axes["input_lob"].items()}

    # ONNX export: forward returns dict, so wrap to tuple in a helper
    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            y = self.m(x)
            return tuple(y[n] for n in out_names)

    wrap = Wrap(model)
    torch.onnx.export(
        wrap, dummy, args.out,
        input_names=["input_lob"],
        output_names=out_names,
        opset_version=int(cfg.get("opset", 17)),
        dynamic_axes=dyn,
    )
    print(f"exported -> {args.out}")

if __name__ == "__main__":
    main()
