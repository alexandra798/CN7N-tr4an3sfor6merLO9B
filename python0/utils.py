from __future__ import annotations
import argparse, yaml, os, random
import numpy as np
import torch

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def parse_args_config(desc: str):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--config", required=True, help="Path to yaml config")
    return p.parse_args()
