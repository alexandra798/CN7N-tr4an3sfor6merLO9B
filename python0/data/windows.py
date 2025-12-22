from __future__ import annotations
import numpy as np
from typing import Optional

class RingWindow:
    """Streaming window buffer. Stores frames as (T,L,C) in time order.

    - push expects (L,C)
    - view returns (1,T,L,C) when full, otherwise None
    """
    def __init__(self, T: int, L: int, C: int):
        self.T, self.L, self.C = int(T), int(L), int(C)
        self.buf = np.zeros((self.T, self.L, self.C), dtype=np.float32)
        self.idx = 0
        self.full = False

    def push(self, feat_LC: np.ndarray) -> None:
        if feat_LC.shape != (self.L, self.C):
            raise ValueError(f"Expected {(self.L, self.C)}, got {feat_LC.shape}")
        self.buf[self.idx, :, :] = feat_LC
        self.idx = (self.idx + 1) % self.T
        if self.idx == 0:
            self.full = True

    def view(self) -> Optional[np.ndarray]:
        if not self.full:
            return None
        # return time-ordered view: oldest -> newest
        # idx points to where the next insert will go (i.e., the oldest frame)
        if self.idx == 0:
            ordered = self.buf
        else:
            ordered = np.concatenate([self.buf[self.idx:], self.buf[:self.idx]], axis=0)
        return ordered[None, :, :, :]  # (1,T,L,C)
