from __future__ import annotations
import math
from typing import Dict, Tuple, Iterable, Sequence
import torch
import torch.nn as nn

class CVMix(nn.Module):
    """Light cross-level/channel mixing, ONNX-friendly.

    Input: (B,T,L,C) -> Output: (B,T,L,C)
    Implemented as a per-timepoint linear over channels plus depthwise-like mixing over levels.
    """
    def __init__(self, L: int, C: int):
        super().__init__()
        self.chan = nn.Linear(C, C, bias=True)
        self.level = nn.Conv1d(in_channels=C, out_channels=C, kernel_size=3, padding=1, groups=C, bias=True)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,L,C)
        B,T,L,C = x.shape
        y = self.chan(x)  # (B,T,L,C)
        y = self.act(y)
        # level mixing: reshape to (B*T, C, L)
        z = y.reshape(B*T, L, C).transpose(1,2)  # (B*T, C, L)
        z = self.level(z)  # (B*T, C, L)
        z = z.transpose(1,2).reshape(B,T,L,C)
        return self.act(z)

class TinyGraphSummary(nn.Module):
    """Extremely small attention summary over levels.

    Input: (B,T,L,C) -> Output: (B,T,g_dim)
    """
    def __init__(self, L: int, C: int, g_dim: int = 16, heads: int = 2):
        super().__init__()
        self.q = nn.Linear(C, g_dim, bias=False)
        self.k = nn.Linear(C, g_dim, bias=False)
        self.v = nn.Linear(C, g_dim, bias=False)
        self.heads = heads
        self.scale = (g_dim // heads) ** -0.5 if g_dim % heads == 0 else (g_dim ** -0.5)
        self.out = nn.Linear(g_dim, g_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B,T,L,C = x.shape
        q = self.q(x)  # (B,T,L,g)
        k = self.k(x)
        v = self.v(x)
        g = q.shape[-1]
        # heads
        h = self.heads
        if g % h != 0:
            # fallback: single head
            h = 1
        d = g // h
        q = q.reshape(B*T, L, h, d).transpose(1,2)  # (B*T,h,L,d)
        k = k.reshape(B*T, L, h, d).transpose(1,2)
        v = v.reshape(B*T, L, h, d).transpose(1,2)
        att = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B*T,h,L,L)
        att = torch.softmax(att, dim=-1)
        ctx = torch.matmul(att, v)  # (B*T,h,L,d)
        # pool across levels -> (B*T,h,d)
        ctx = ctx.mean(dim=2)
        ctx = ctx.reshape(B*T, h*d)
        ctx = self.out(ctx).reshape(B, T, -1)
        return ctx

class PatchEmbed(nn.Module):
    """Structured patch embedding.

    Split (T,L) into patches (pT,pL), flatten each patch (pT*pL*C) -> d_model.
    Output tokens: (B, N, d_model) where N=(T/pT)*(L/pL).
    """
    def __init__(self, T: int, L: int, C: int, pT: int, pL: int, d_model: int):
        super().__init__()
        assert T % pT == 0 and L % pL == 0, "T%pT==0 and L%pL==0 required"
        self.T, self.L, self.C = T, L, C
        self.pT, self.pL = pT, pL
        self.nT, self.nL = T // pT, L // pL
        self.d_model = d_model
        self.proj = nn.Linear(pT * pL * C, d_model, bias=True)
        self.pos = nn.Parameter(torch.zeros(1, self.nT * self.nL, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,L,C)
        B,T,L,C = x.shape
        pT,pL = self.pT, self.pL
        # reshape into patches
        x = x.reshape(B, self.nT, pT, self.nL, pL, C)          # (B,nT,pT,nL,pL,C)
        x = x.permute(0,1,3,2,4,5).contiguous()               # (B,nT,nL,pT,pL,C)
        x = x.reshape(B, self.nT * self.nL, pT * pL * C)      # (B,N,patch_dim)
        x = self.proj(x) + self.pos
        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, mlp_ratio: float = 1.5, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead,
                                          dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.ln1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + y
        z = self.ln2(x)
        z = self.mlp(z)
        return x + z

class LiTCVG_Lite(nn.Module):
    """LiT-CVG Lite: CVMix + GraphSummary + PatchEmbed + small Transformer + multi-horizon heads."""
    def __init__(self, T: int = 100, L: int = 20, C: int = 7,
                 pT: int = 10, pL: int = 5, d_model: int = 64,
                 depth: int = 2, horizons: Sequence[int] = (10,50,100),
                 g_dim: int = 16, nhead: int = 4):
        super().__init__()
        self.T, self.L, self.C = T, L, C
        self.horizons = list(map(int, horizons))
        self.cvmix = CVMix(L=L, C=C)
        self.graph = TinyGraphSummary(L=L, C=C, g_dim=g_dim, heads=2)
        self.patch = PatchEmbed(T=T, L=L, C=C, pT=pT, pL=pL, d_model=d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model=d_model, nhead=nhead) for _ in range(depth)])
        self.ln = nn.LayerNorm(d_model)
        self.pool = nn.AdaptiveAvgPool1d(1)
        # heads: 3-class classification
        self.heads = nn.ModuleDict({f"logits_{h}": nn.Linear(d_model + g_dim, 3) for h in self.horizons})

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: (B,T,L,C)
        x = self.cvmix(x)
        g = self.graph(x)  # (B,T,g_dim)
        # pool graph over time
        g_pool = g.mean(dim=1)  # (B,g_dim)
        tok = self.patch(x)     # (B,N,d_model)
        for blk in self.blocks:
            tok = blk(tok)
        tok = self.ln(tok)
        # token pooling
        # (B,N,d) -> (B,d,N) -> avg over N
        pooled = tok.transpose(1,2)
        pooled = self.pool(pooled).squeeze(-1)  # (B,d_model)
        feat = torch.cat([pooled, g_pool], dim=-1)
        out = {name: head(feat) for name, head in self.heads.items()}
        return out
