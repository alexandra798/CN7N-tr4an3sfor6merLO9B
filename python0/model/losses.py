from __future__ import annotations
from typing import Dict
import torch
import torch.nn.functional as F

def multi_horizon_ce(outputs: Dict[str, torch.Tensor],
                     targets: Dict[int, torch.Tensor],
                     weights: Dict[int, float]) -> torch.Tensor:
    """Multi-task cross entropy.

    outputs keys: 'logits_{h}'
    targets: {h: (B,) int64}
    weights: {h: float}
    """
    loss = 0.0
    tot = 0.0
    for h, w in weights.items():
        key = f"logits_{int(h)}"
        if key not in outputs:
            raise KeyError(f"Missing output {key}")
        if int(h) not in targets:
            raise KeyError(f"Missing target for horizon {h}")
        l = F.cross_entropy(outputs[key], targets[int(h)])
        loss = loss + float(w) * l
        tot += float(w)
    return loss / max(tot, 1e-12)
