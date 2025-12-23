from __future__ import annotations
from typing import Dict, List
import numpy as np
import torch
import torch.nn.functional as F


def adaptive_weights(horizons: List[int], decay: float = 0.01, 
                     strategy: str = "exponential") -> Dict[int, float]:
    """Generate adaptive weights for multi-horizon tasks.
    
    Args:
        horizons: List of prediction horizons (e.g., [10, 50, 100])
        decay: Decay rate for exponential weighting
        strategy: "exponential", "inverse", or "uniform"
    
    Returns:
        Normalized weights dict {horizon: weight}
        
    Strategies:
        - exponential: w_h = exp(-decay * h), prioritizes short-term
        - inverse: w_h = 1/h, also prioritizes short-term
        - uniform: w_h = 1, equal weighting
    """
    horizons = sorted([int(h) for h in horizons])
    
    if strategy == "exponential":
        # Exponential decay: shorter horizons have higher weight
        weights = {h: np.exp(-decay * h) for h in horizons}
    elif strategy == "inverse":
        # Inverse weighting: w_h = 1/h
        weights = {h: 1.0 / h for h in horizons}
    elif strategy == "uniform":
        # Uniform: equal weight
        weights = {h: 1.0 for h in horizons}
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Normalize to sum to 1
    total = sum(weights.values())
    return {h: w / total for h, w in weights.items()}


def multi_horizon_ce(outputs: Dict[str, torch.Tensor],
                     targets: Dict[int, torch.Tensor],
                     weights: Dict[int, float],
                     return_details: bool = False) -> torch.Tensor | Dict:
    """Multi-task cross entropy with optional detailed statistics.

    Args:
        outputs: {'logits_{h}': (B, num_classes)}
        targets: {h: (B,) int64 labels}
        weights: {h: float weight}
        return_details: If True, return dict with per-horizon losses
        
    Returns:
        If return_details=False: scalar loss tensor
        If return_details=True: {
            'loss': scalar,
            'loss_h10': scalar,
            'loss_h50': scalar,
            ...
        }
    """
    loss = 0.0
    tot = 0.0
    details = {}
    
    for h, w in weights.items():
        key = f"logits_{int(h)}"
        if key not in outputs:
            raise KeyError(f"Missing output {key}")
        if int(h) not in targets:
            raise KeyError(f"Missing target for horizon {h}")
        
        # Compute cross-entropy for this horizon
        l = F.cross_entropy(outputs[key], targets[int(h)])
        
        # Accumulate weighted loss
        loss = loss + float(w) * l
        tot += float(w)
        
        # Store per-horizon loss for diagnostics
        if return_details:
            details[f'loss_h{h}'] = l.item()
    
    final_loss = loss / max(tot, 1e-12)
    
    if return_details:
        details['loss'] = final_loss.item() if isinstance(final_loss, torch.Tensor) else final_loss
        details['weights'] = weights
        return details
    
    return final_loss


def multi_horizon_mse(outputs: Dict[str, torch.Tensor],
                      targets: Dict[int, torch.Tensor],
                      weights: Dict[int, float]) -> torch.Tensor:
    """Multi-task MSE loss for regression tasks.
    
    Args:
        outputs: {'logits_{h}': (B, 1) or (B,)}
        targets: {h: (B,) float targets}
        weights: {h: float weight}
    """
    loss = 0.0
    tot = 0.0
    
    for h, w in weights.items():
        key = f"logits_{int(h)}"
        if key not in outputs:
            raise KeyError(f"Missing output {key}")
        if int(h) not in targets:
            raise KeyError(f"Missing target for horizon {h}")
        
        pred = outputs[key].squeeze()
        target = targets[int(h)].float()
        
        l = F.mse_loss(pred, target)
        loss = loss + float(w) * l
        tot += float(w)
    
    return loss / max(tot, 1e-12)


def focal_loss(logits: torch.Tensor, targets: torch.Tensor, 
               alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """Focal loss for handling class imbalance.
    
    Useful when certain directions (e.g., up/down) are rare.
    
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    ce_loss = F.cross_entropy(logits, targets, reduction='none')
    p_t = torch.exp(-ce_loss)
    focal = alpha * (1 - p_t) ** gamma * ce_loss
    return focal.mean()


def multi_horizon_focal(outputs: Dict[str, torch.Tensor],
                        targets: Dict[int, torch.Tensor],
                        weights: Dict[int, float],
                        alpha: float = 0.25,
                        gamma: float = 2.0) -> torch.Tensor:
    """Multi-task focal loss for imbalanced classification."""
    loss = 0.0
    tot = 0.0
    
    for h, w in weights.items():
        key = f"logits_{int(h)}"
        if key not in outputs:
            raise KeyError(f"Missing output {key}")
        if int(h) not in targets:
            raise KeyError(f"Missing target for horizon {h}")
        
        l = focal_loss(outputs[key], targets[int(h)], alpha=alpha, gamma=gamma)
        loss = loss + float(w) * l
        tot += float(w)
    
    return loss / max(tot, 1e-12)


# ===== Auxiliary losses for microstructure =====

def ofi_auxiliary_loss(ofi_pred: torch.Tensor, ofi_target: torch.Tensor) -> torch.Tensor:
    """Auxiliary loss for predicting future OFI.
    
    Can be added to main loss to improve microstructure sensitivity:
        total_loss = main_loss + lambda_ofi * ofi_auxiliary_loss(...)
    """
    return F.mse_loss(ofi_pred, ofi_target)


# ===== Testing utilities =====

def test_adaptive_weights():
    """Test adaptive weight strategies"""
    horizons = [10, 50, 100]
    
    # Test exponential decay
    w_exp = adaptive_weights(horizons, decay=0.015, strategy="exponential")
    print(f"✅ Exponential weights: {w_exp}")
    assert abs(sum(w_exp.values()) - 1.0) < 1e-6, "Weights must sum to 1"
    assert w_exp[10] > w_exp[50] > w_exp[100], "Should prioritize short-term"
    
    # Test inverse
    w_inv = adaptive_weights(horizons, strategy="inverse")
    print(f"✅ Inverse weights: {w_inv}")
    assert abs(sum(w_inv.values()) - 1.0) < 1e-6
    
    # Test uniform
    w_uni = adaptive_weights(horizons, strategy="uniform")
    print(f"✅ Uniform weights: {w_uni}")
    assert abs(sum(w_uni.values()) - 1.0) < 1e-6
    assert w_uni[10] == w_uni[50] == w_uni[100]
    
    print("✅ All weight strategy tests passed!")


def test_multi_horizon_ce():
    """Test multi-horizon CE loss"""
    B = 8
    horizons = [10, 50, 100]
    
    # Dummy outputs
    outputs = {
        f"logits_{h}": torch.randn(B, 3) for h in horizons
    }
    
    # Dummy targets
    targets = {
        h: torch.randint(0, 3, (B,)) for h in horizons
    }
    
    # Test with adaptive weights
    weights = adaptive_weights(horizons, decay=0.01)
    
    # Test basic loss
    loss = multi_horizon_ce(outputs, targets, weights)
    print(f"✅ Basic loss: {loss.item():.4f}")
    assert loss.item() > 0, "Loss should be positive"
    
    # Test with details
    details = multi_horizon_ce(outputs, targets, weights, return_details=True)
    print(f"✅ Detailed loss: {details}")
    assert 'loss' in details
    assert 'loss_h10' in details
    assert 'weights' in details
    
    print("✅ All multi-horizon CE tests passed!")


if __name__ == "__main__":
    test_adaptive_weights()
    print()
    test_multi_horizon_ce()