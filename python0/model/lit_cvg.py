from __future__ import annotations
import math
from typing import Dict, Tuple, Iterable, Sequence
import torch
import torch.nn as nn

class CVMix(nn.Module):
    """Light cross-level/channel mixing with channel doubling, ONNX-friendly.

    Input: (B,T,L,C) -> Output: (B,T,L,2C)
    
    Design: 
    - Cross-level mixing: Linear over L dimension to capture price-level interactions
    - Channel mixing: Optional projection over C dimension  
    - Output: Concatenate original and mixed features -> 2C channels
    """
    def __init__(self, L: int, C: int):
        super().__init__()
        self.L = L
        self.C = C
        # Cross-level mixing: operates on L dimension
        self.mix_level = nn.Linear(L, L, bias=False)
        nn.init.eye_(self.mix_level.weight)  # Initialize as identity, learn deviations
        
        # Optional channel projection
        self.mix_chan = nn.Linear(C, C, bias=False)
        nn.init.eye_(self.mix_chan.weight)
        
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,L,C)
        B, T, L, C = x.shape
        
        # Reshape to apply level mixing: (B,T,L,C) -> (B,C,T,L)
        x_perm = x.permute(0, 3, 1, 2).contiguous()  # (B,C,T,L)
        
        # Apply level mixing: mix across L dimension
        # Reshape to (B*C*T, L) for batch processing
        x_flat = x_perm.reshape(B * C * T, L)
        x_mixed = self.mix_level(x_flat)  # (B*C*T, L)
        x_mixed = x_mixed.view(B, C, T, L)
        
        # Apply activation
        x_mixed = self.act(x_mixed)
        
        # Optional channel mixing
        x_mixed = x_mixed.permute(0, 2, 3, 1).contiguous()  # (B,T,L,C)
        x_mixed = self.mix_chan(x_mixed)
        x_mixed = self.act(x_mixed)
        
        # ⭐ KEY: Concatenate along channel dimension -> (B,T,L,2C)
        return torch.cat([x, x_mixed], dim=-1)


class TinyGraphSummary(nn.Module):
    """Extremely small attention summary over levels with proper multi-head.

    Input: (B,T,L,Cprime) -> Output: (B,T,g_dim)
    
    Design:
    - Multi-head attention over L (price levels) at each timestep
    - Aggregate across levels to get graph summary
    - Standard scaled dot-product attention
    """
    def __init__(self, L: int, Cprime: int, g_dim: int = 16, heads: int = 2):
        super().__init__()
        assert g_dim % heads == 0, f"g_dim={g_dim} must be divisible by heads={heads}"
        
        self.L = L
        self.Cprime = Cprime
        self.g_dim = g_dim
        self.heads = heads
        self.d_k = g_dim // heads  # dimension per head
        
        # Standard multi-head projections
        self.q = nn.Linear(Cprime, g_dim, bias=False)
        self.k = nn.Linear(Cprime, g_dim, bias=False)
        self.v = nn.Linear(Cprime, g_dim, bias=False)
        
        self.scale = self.d_k ** -0.5
        self.out = nn.Linear(g_dim, g_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,L,Cprime)
        B, T, L, D = x.shape
        
        # Project to Q, K, V
        q = self.q(x)  # (B,T,L,g_dim)
        k = self.k(x)  # (B,T,L,g_dim)
        v = self.v(x)  # (B,T,L,g_dim)
        
        # Reshape for multi-head: (B,T,L,g_dim) -> (B,T,heads,L,d_k)
        q = q.view(B, T, L, self.heads, self.d_k).transpose(2, 3)  # (B,T,h,L,d_k)
        k = k.view(B, T, L, self.heads, self.d_k).transpose(2, 3)  # (B,T,h,L,d_k)
        v = v.view(B, T, L, self.heads, self.d_k).transpose(2, 3)  # (B,T,h,L,d_k)
        
        # Scaled dot-product attention: (B,T,h,L,L)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        
        # Apply attention to values: (B,T,h,L,d_k)
        ctx = torch.matmul(attn, v)
        
        # Pool across levels (aggregate graph): (B,T,h,d_k)
        ctx = ctx.mean(dim=3)
        
        # Concatenate heads: (B,T,g_dim)
        ctx = ctx.transpose(2, 3).contiguous()  # (B,T,d_k,h)
        ctx = ctx.view(B, T, self.g_dim)
        
        # Output projection
        return self.out(ctx)


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
        B, T, L, C = x.shape
        pT, pL = self.pT, self.pL
        # reshape into patches
        x = x.reshape(B, self.nT, pT, self.nL, pL, C)          # (B,nT,pT,nL,pL,C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()           # (B,nT,nL,pT,pL,C)
        x = x.reshape(B, self.nT * self.nL, pT * pL * C)       # (B,N,patch_dim)
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
    """LiT-CVG Lite: CVMix + GraphSummary + PatchEmbed + small Transformer + multi-horizon heads.
    
    Architecture:
    1. CVMix: Cross-level mixing with channel doubling (C -> 2C)
    2. GraphSummary: Attention-based level aggregation
    3. PatchEmbed: Structured spatiotemporal patches
    4. Transformer: Self-attention over patches
    5. Multi-horizon heads: Joint prediction for alpha term structure
    """
    def __init__(self, T: int = 100, L: int = 20, C: int = 7,
                 pT: int = 10, pL: int = 5, d_model: int = 64,
                 depth: int = 2, horizons: Sequence[int] = (10, 50, 100),
                 g_dim: int = 16, nhead: int = 4):
        super().__init__()
        self.T, self.L, self.C = T, L, C
        self.horizons = list(map(int, horizons))
        
        # CVMix: C -> 2C
        self.cvmix = CVMix(L=L, C=C)
        Cprime = C * 2  # ⭐ After CVMix, channels are doubled
        
        # Graph summary and patch embed use Cprime
        self.graph = TinyGraphSummary(L=L, Cprime=Cprime, g_dim=g_dim, heads=2)
        self.patch = PatchEmbed(T=T, L=L, C=Cprime, pT=pT, pL=pL, d_model=d_model)
        
        # Transformer backbone
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, nhead=nhead) for _ in range(depth)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Multi-horizon heads: 3-class classification per horizon
        self.heads = nn.ModuleDict({
            f"logits_{h}": nn.Linear(d_model + g_dim, 3) 
            for h in self.horizons
        })

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: (B,T,L,C_raw)
        
        # Step 1: Cross-level/channel mixing -> (B,T,L,2C)
        x = self.cvmix(x)
        
        # Step 2: Graph summary over levels -> (B,T,g_dim)
        g = self.graph(x)
        g_pool = g.mean(dim=1)  # Time average -> (B,g_dim)
        
        # Step 3: Patch embedding -> (B,N,d_model)
        tok = self.patch(x)
        
        # Step 4: Transformer processing
        for blk in self.blocks:
            tok = blk(tok)
        tok = self.ln(tok)
        
        # Step 5: Token pooling -> (B,d_model)
        pooled = tok.transpose(1, 2)  # (B,d_model,N)
        pooled = self.pool(pooled).squeeze(-1)  # (B,d_model)
        
        # Step 6: Concatenate features -> (B, d_model+g_dim)
        feat = torch.cat([pooled, g_pool], dim=-1)
        
        # Step 7: Multi-horizon classification
        out = {name: head(feat) for name, head in self.heads.items()}
        return out


# ===== Testing utility =====
def test_litcvg_shapes():
    """Sanity check for shape propagation"""
    B, T, L, C = 4, 100, 20, 7
    model = LiTCVG_Lite(T=T, L=L, C=C, pT=10, pL=5, d_model=64, depth=2, 
                        horizons=[10, 50, 100], g_dim=16)
    x = torch.randn(B, T, L, C)
    
    with torch.no_grad():
        out = model(x)
    
    print("✅ LiTCVG_Lite Shape Test:")
    print(f"   Input: {x.shape}")
    for k, v in out.items():
        print(f"   {k}: {v.shape}")
    
    # Verify output shapes
    for k, v in out.items():
        assert v.shape == (B, 3), f"Expected {k} shape (B,3), got {v.shape}"
    
    print("✅ All shape tests passed!")


if __name__ == "__main__":
    test_litcvg_shapes()