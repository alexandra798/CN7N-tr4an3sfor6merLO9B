import json
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple, Optional
import numpy as np

L_DEFAULT = 20

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
    Here we apply the common piecewise definition:

    If bid price increased: +cur_bid_s
    If bid price unchanged: +(cur_bid_s - prev_bid_s)
    If bid price decreased: -prev_bid_s

    If ask price decreased: +prev_ask_s
    If ask price unchanged: +(prev_ask_s - cur_ask_s)
    If ask price increased: -cur_ask_s

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

def make_feature_frame(prev: Dict, cur: Dict, L: int = L_DEFAULT,
                       cum_vol_prev: float = 0.0) -> Tuple[np.ndarray, float]:
    """Map one LOB snapshot pair -> (L, C_raw) feature frame.

    Expected dict format (minimal):
      prev/cur: {
        "bid_p": (L,), "ask_p": (L,),
        "bid_s": (L,), "ask_s": (L,),
        "trade_vol": float (optional)
      }

    Contract default C_raw=7:
      [ bid_size, ask_size, ofi_best1, spread, mid, dmid, cum_vol ]
    - bid_size/ask_size are per level
    - other features are broadcast to all L levels (so each level has same extra scalars)
    """
    bid_p0_prev, ask_p0_prev = float(prev["bid_p"][0]), float(prev["ask_p"][0])
    bid_p0_cur,  ask_p0_cur  = float(cur["bid_p"][0]),  float(cur["ask_p"][0])
    bid_s0_prev, ask_s0_prev = float(prev["bid_s"][0]), float(prev["ask_s"][0])
    bid_s0_cur,  ask_s0_cur  = float(cur["bid_s"][0]),  float(cur["ask_s"][0])

    ofi = compute_ofi(bid_p0_prev, ask_p0_prev, bid_s0_prev, ask_s0_prev,
                      bid_p0_cur,  ask_p0_cur,  bid_s0_cur,  ask_s0_cur)

    spread = ask_p0_cur - bid_p0_cur
    mid = 0.5 * (ask_p0_cur + bid_p0_cur)
    mid_prev = 0.5 * (ask_p0_prev + bid_p0_prev)
    dmid = mid - mid_prev

    trade_vol = float(cur.get("trade_vol", 0.0))
    cum_vol = cum_vol_prev + trade_vol

    bid_s = np.asarray(cur["bid_s"][:L], dtype=np.float32)
    ask_s = np.asarray(cur["ask_s"][:L], dtype=np.float32)

    scalars = np.asarray([ofi, spread, mid, dmid, cum_vol], dtype=np.float32)
    scalars_L = np.repeat(scalars[None, :], repeats=L, axis=0)  # (L,5)

    frame = np.concatenate([bid_s[:, None], ask_s[:, None], scalars_L], axis=1)  # (L,7)
    return frame.astype(np.float32), float(cum_vol)

def normalize(frame_LC: np.ndarray, stats: NormStats) -> np.ndarray:
    """Normalize last dimension C using global mean/std."""
    if frame_LC.shape[-1] != stats.mean.shape[0]:
        raise ValueError(f"C mismatch: frame C={frame_LC.shape[-1]} vs stats C={stats.mean.shape[0]}")
    return (frame_LC - stats.mean[None, :]) / stats.std[None, :]

def update_norm_stats(frames: Iterable[np.ndarray], warmup: int = 20000) -> NormStats:
    """Streaming mean/std over frames of shape (L,C). Uses Welford."""
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
