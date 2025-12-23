import json
from dataclasses import dataclass, field
from typing import Dict, Iterable, Tuple, Optional, List
import numpy as np

L_DEFAULT = 20


@dataclass
class FeatureConfig:
    """Configuration for feature generation.
    
    Controls which features to include and how to compute them.
    Allows flexible channel count (C_raw) based on selected features.
    """
    include_price: bool = False           # Include normalized price levels
    include_volume_profile: bool = False  # Include cumulative volume profile
    ofi_levels: int = 1                   # Number of levels for OFI (1=best, 5=top5)
    price_normalization: str = "mid"      # "mid" or "log" for price normalization
    
    def get_channel_count(self) -> int:
        """Calculate total number of channels based on config."""
        # Base: bid_size, ask_size
        C = 2
        
        # OFI (broadcast to all levels)
        C += 1
        
        # Spread, mid, dmid, cum_vol (broadcast to all levels)
        C += 4
        
        # Optional: price levels (per level, not broadcast)
        if self.include_price:
            C += 2  # bid_p_norm, ask_p_norm
        
        # Optional: volume profile
        if self.include_volume_profile:
            C += 1  # cumulative volume percentage
        
        return C


@dataclass
class NormStats:
    mean: np.ndarray
    std: np.ndarray

    def to_jsonable(self) -> Dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @staticmethod
    def from_jsonable(d: Dict) -> "NormStats":
        mean = np.asarray(d["mean"], dtype=np.float32)
        std = np.asarray(d["std"], dtype=np.float32)
        std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
        return NormStats(mean=mean, std=std)


def compute_ofi(prev_bid_p: float, prev_ask_p: float, prev_bid_s: float, prev_ask_s: float,
                cur_bid_p: float, cur_ask_p: float, cur_bid_s: float, cur_ask_s: float) -> float:
    """Best-level OFI (Cont et al.-style) simplified and deterministic.

    OFI = Δbid_size * I(bid_price unchanged or improved) - Δask_size * I(ask_price unchanged or improved)
    
    Piecewise definition:
    - If bid price increased: +cur_bid_s
    - If bid price unchanged: +(cur_bid_s - prev_bid_s)
    - If bid price decreased: -prev_bid_s
    
    - If ask price decreased: +prev_ask_s
    - If ask price unchanged: +(prev_ask_s - cur_ask_s)
    - If ask price increased: -cur_ask_s

    This returns a signed number; caller may scale/normalize.
    """
    ofi_bid = 0.0
    if cur_bid_p > prev_bid_p:
        ofi_bid = cur_bid_s
    elif cur_bid_p == prev_bid_p:
        ofi_bid = cur_bid_s - prev_bid_s
    else:
        ofi_bid = -prev_bid_s

    ofi_ask = 0.0
    if cur_ask_p < prev_ask_p:
        ofi_ask = prev_ask_s
    elif cur_ask_p == prev_ask_p:
        ofi_ask = prev_ask_s - cur_ask_s
    else:
        ofi_ask = -cur_ask_s

    return float(ofi_bid + ofi_ask)


def compute_multi_level_ofi(prev: Dict, cur: Dict, levels: int = 5) -> np.ndarray:
    """Compute OFI for multiple price levels.
    
    Returns: (levels,) array of OFI values
    """
    ofi_vec = np.zeros(levels, dtype=np.float32)
    
    for lv in range(min(levels, len(prev["bid_p"]))):
        ofi_lv = compute_ofi(
            prev["bid_p"][lv], prev["ask_p"][lv], 
            prev["bid_s"][lv], prev["ask_s"][lv],
            cur["bid_p"][lv], cur["ask_p"][lv],
            cur["bid_s"][lv], cur["ask_s"][lv]
        )
        ofi_vec[lv] = ofi_lv
    
    return ofi_vec


def make_feature_frame(prev: Dict, cur: Dict, L: int = L_DEFAULT,
                       cum_vol_prev: float = 0.0,
                       config: Optional[FeatureConfig] = None) -> Tuple[np.ndarray, float]:
    """Map one LOB snapshot pair -> (L, C_raw) feature frame.

    Expected dict format (minimal):
      prev/cur: {
        "bid_p": (L,), "ask_p": (L,),
        "bid_s": (L,), "ask_s": (L,),
        "trade_vol": float (optional)
      }

    Returns:
      frame: (L, C_raw) where C_raw depends on config
      cum_vol: updated cumulative volume
      
    Default C_raw=7 (if config is None):
      [ bid_size, ask_size, ofi_best1, spread, mid, dmid, cum_vol ]
      
    With config.include_price=True, C_raw=9:
      [ bid_size, ask_size, ofi, spread, mid, dmid, cum_vol, bid_p_norm, ask_p_norm ]
    """
    if config is None:
        config = FeatureConfig()
    
    # Extract best level for scalar features
    bid_p0_prev, ask_p0_prev = float(prev["bid_p"][0]), float(prev["ask_p"][0])
    bid_p0_cur,  ask_p0_cur  = float(cur["bid_p"][0]),  float(cur["ask_p"][0])
    bid_s0_prev, ask_s0_prev = float(prev["bid_s"][0]), float(prev["ask_s"][0])
    bid_s0_cur,  ask_s0_cur  = float(cur["bid_s"][0]),  float(cur["ask_s"][0])

    # Compute OFI
    if config.ofi_levels == 1:
        ofi = compute_ofi(bid_p0_prev, ask_p0_prev, bid_s0_prev, ask_s0_prev,
                         bid_p0_cur,  ask_p0_cur,  bid_s0_cur,  ask_s0_cur)
        ofi_vec = np.full((L,), ofi, dtype=np.float32)
    else:
        # Multi-level OFI (aggregate or use first level)
        ofi_multi = compute_multi_level_ofi(prev, cur, config.ofi_levels)
        ofi = float(ofi_multi[0])  # Use best level for broadcast
        ofi_vec = np.full((L,), ofi, dtype=np.float32)

    # Compute spread, mid, dmid
    spread = ask_p0_cur - bid_p0_cur
    mid = 0.5 * (ask_p0_cur + bid_p0_cur)
    mid_prev = 0.5 * (ask_p0_prev + bid_p0_prev)
    dmid = mid - mid_prev

    # Cumulative volume
    trade_vol = float(cur.get("trade_vol", 0.0))
    cum_vol = cum_vol_prev + trade_vol

    # Build feature channels
    channels = []
    
    # 1. Base: bid_size, ask_size (per level)
    bid_s = np.asarray(cur["bid_s"][:L], dtype=np.float32)
    ask_s = np.asarray(cur["ask_s"][:L], dtype=np.float32)
    channels.append(bid_s[:, None])  # (L,1)
    channels.append(ask_s[:, None])  # (L,1)
    
    # 2. OFI (broadcast to all levels)
    channels.append(ofi_vec[:, None])  # (L,1)
    
    # 3. Spread, mid, dmid, cum_vol (broadcast to all levels)
    scalars = np.asarray([spread, mid, dmid, cum_vol], dtype=np.float32)
    scalars_L = np.repeat(scalars[None, :], repeats=L, axis=0)  # (L,4)
    channels.append(scalars_L)
    
    # 4. Optional: normalized price levels (per level, not broadcast)
    if config.include_price:
        bid_p = np.asarray(cur["bid_p"][:L], dtype=np.float32)
        ask_p = np.asarray(cur["ask_p"][:L], dtype=np.float32)
        
        if config.price_normalization == "mid":
            # Mid-centered normalization
            bid_p_norm = (bid_p - mid) / (mid + 1e-8)
            ask_p_norm = (ask_p - mid) / (mid + 1e-8)
        elif config.price_normalization == "log":
            # Log price difference
            bid_p_norm = np.log(bid_p / (mid + 1e-8) + 1e-8)
            ask_p_norm = np.log(ask_p / (mid + 1e-8) + 1e-8)
        else:
            # Default: mid-centered
            bid_p_norm = (bid_p - mid) / (mid + 1e-8)
            ask_p_norm = (ask_p - mid) / (mid + 1e-8)
        
        channels.append(bid_p_norm[:, None])  # (L,1)
        channels.append(ask_p_norm[:, None])  # (L,1)
    
    # 5. Optional: volume profile
    if config.include_volume_profile:
        # Cumulative volume percentage (simple version)
        total_vol = bid_s.sum() + ask_s.sum() + 1e-8
        vol_profile = (bid_s + ask_s) / total_vol
        channels.append(vol_profile[:, None])  # (L,1)
    
    # Concatenate all channels
    frame = np.concatenate(channels, axis=1)  # (L, C_raw)
    
    return frame.astype(np.float32), float(cum_vol)


def normalize(frame_LC: np.ndarray, stats: NormStats) -> np.ndarray:
    """Normalize last dimension C using global mean/std."""
    if frame_LC.shape[-1] != stats.mean.shape[0]:
        raise ValueError(f"C mismatch: frame C={frame_LC.shape[-1]} vs stats C={stats.mean.shape[0]}")
    return (frame_LC - stats.mean[None, :]) / stats.std[None, :]


def update_norm_stats(frames: Iterable[np.ndarray], warmup: int = 20000) -> NormStats:
    """Streaming mean/std over frames of shape (L,C). Uses Welford's algorithm."""
    n = 0
    mean = None
    M2 = None
    for frame in frames:
        # collapse L dimension so each (L,C) contributes L samples
        x = frame.reshape(-1, frame.shape[-1]).astype(np.float64)  # (L, C)
        if mean is None:
            mean = np.zeros((x.shape[1],), dtype=np.float64)
            M2 = np.zeros((x.shape[1],), dtype=np.float64)
        for row in x:
            n += 1
            delta = row - mean
            mean += delta / n
            delta2 = row - mean
            M2 += delta * delta2
        if n >= warmup * frame.shape[0]:
            break
    if n < 2:
        raise ValueError("Not enough samples for norm stats")
    var = M2 / (n - 1)
    std = np.sqrt(np.maximum(var, 1e-12))
    return NormStats(mean=mean.astype(np.float32), std=std.astype(np.float32))


def save_norm_stats(path: str, stats: NormStats) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats.to_jsonable(), f, indent=2)


def load_norm_stats(path: str) -> NormStats:
    with open(path, "r", encoding="utf-8") as f:
        return NormStats.from_jsonable(json.load(f))


# ===== Testing utility =====
def test_feature_configs():
    """Test different feature configurations"""
    # Dummy LOB frames
    prev = {
        "bid_p": np.array([100.0 - i*0.01 for i in range(20)], dtype=np.float32),
        "ask_p": np.array([100.1 + i*0.01 for i in range(20)], dtype=np.float32),
        "bid_s": np.ones(20, dtype=np.float32) * 10,
        "ask_s": np.ones(20, dtype=np.float32) * 10,
        "trade_vol": 0.0
    }
    
    cur = {
        "bid_p": np.array([100.01 - i*0.01 for i in range(20)], dtype=np.float32),
        "ask_p": np.array([100.11 + i*0.01 for i in range(20)], dtype=np.float32),
        "bid_s": np.ones(20, dtype=np.float32) * 11,
        "ask_s": np.ones(20, dtype=np.float32) * 9,
        "trade_vol": 1.5
    }
    
    # Test 1: Default config (7 channels)
    config1 = FeatureConfig()
    frame1, cum_vol1 = make_feature_frame(prev, cur, L=20, config=config1)
    print(f"✅ Default config: shape={frame1.shape}, expected=(20,7)")
    assert frame1.shape == (20, 7), f"Expected (20,7), got {frame1.shape}"
    
    # Test 2: With price (9 channels)
    config2 = FeatureConfig(include_price=True)
    frame2, cum_vol2 = make_feature_frame(prev, cur, L=20, config=config2)
    print(f"✅ With price: shape={frame2.shape}, expected=(20,9)")
    assert frame2.shape == (20, 9), f"Expected (20,9), got {frame2.shape}"
    
    # Test 3: With price + volume profile (10 channels)
    config3 = FeatureConfig(include_price=True, include_volume_profile=True)
    frame3, cum_vol3 = make_feature_frame(prev, cur, L=20, config=config3)
    print(f"✅ With price + vol_profile: shape={frame3.shape}, expected=(20,10)")
    assert frame3.shape == (20, 10), f"Expected (20,10), got {frame3.shape}"
    
    # Test 4: Channel count calculation
    assert config1.get_channel_count() == 7
    assert config2.get_channel_count() == 9
    assert config3.get_channel_count() == 10
    print("✅ Channel count calculations correct!")
    
    # Test 5: OFI computation
    ofi = compute_ofi(100.0, 100.1, 10.0, 10.0, 100.01, 100.11, 11.0, 9.0)
    print(f"✅ OFI computed: {ofi:.4f}")
    
    print("\n✅ All feature config tests passed!")


if __name__ == "__main__":
    test_feature_configs()