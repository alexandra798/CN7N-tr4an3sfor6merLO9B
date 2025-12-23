"""Attention visualization for LiT-CVG model.

This module provides tools to visualize:
1. Level attention (price-level interactions)
2. Patch attention (spatiotemporal patterns)
3. Attention evolution over time
"""
from __future__ import annotations
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns


def extract_graph_attention(model, x: torch.Tensor) -> np.ndarray:
    """Extract attention weights from TinyGraphSummary.
    
    Args:
        model: LiTCVG_Lite model
        x: Input tensor (B,T,L,C)
        
    Returns:
        Attention weights (B,T,heads,L,L)
        
    Note: Requires modifying TinyGraphSummary.forward to return attn weights
    """
    # This requires model modification to return attention
    # For now, we provide the framework
    
    model.eval()
    with torch.no_grad():
        # Step 1: CVMix
        x_mixed = model.cvmix(x)  # (B,T,L,2C)
        
        # Step 2: Extract attention from graph summary
        # Need to modify TinyGraphSummary to return attn
        graph_module = model.graph
        
        B, T, L, C = x_mixed.shape
        q = graph_module.q(x_mixed)
        k = graph_module.k(x_mixed)
        v = graph_module.v(x_mixed)
        
        # Reshape for multi-head
        g_dim = graph_module.g_dim
        heads = graph_module.heads
        d_k = g_dim // heads
        
        q = q.view(B, T, L, heads, d_k).transpose(2, 3)  # (B,T,h,L,d_k)
        k = k.view(B, T, L, heads, d_k).transpose(2, 3)
        
        # Compute attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * graph_module.scale
        attn = torch.softmax(attn, dim=-1)  # (B,T,h,L,L)
        
    return attn.cpu().numpy()


def plot_level_attention_heatmap(attn: np.ndarray, 
                                  timestep: int = 0,
                                  sample_idx: int = 0,
                                  save_path: Optional[str] = None,
                                  figsize: Tuple[int, int] = (15, 4)):
    """Plot attention heatmap across price levels.
    
    Args:
        attn: Attention weights (B,T,heads,L,L)
        timestep: Which timestep to visualize
        sample_idx: Which sample in batch
        save_path: Path to save figure
        figsize: Figure size
    """
    B, T, heads, L, L = attn.shape
    
    fig, axes = plt.subplots(1, heads, figsize=figsize)
    if heads == 1:
        axes = [axes]
    
    # Extract attention for specified sample and timestep
    attn_t = attn[sample_idx, timestep]  # (heads, L, L)
    
    for i, ax in enumerate(axes):
        sns.heatmap(
            attn_t[i], 
            ax=ax, 
            cmap='viridis', 
            cbar=True,
            square=True,
            vmin=0, 
            vmax=attn_t[i].max(),
            xticklabels=5,
            yticklabels=5
        )
        ax.set_title(f'Head {i+1}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Key Level', fontsize=10)
        ax.set_ylabel('Query Level', fontsize=10)
        
        # Add grid for better readability
        ax.grid(False)
    
    plt.suptitle(f'Price Level Attention at t={timestep}', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved level attention heatmap to {save_path}")
    
    return fig


def plot_attention_aggregation(attn: np.ndarray,
                               save_path: Optional[str] = None,
                               figsize: Tuple[int, int] = (12, 6)):
    """Plot aggregated attention patterns over time and levels.
    
    Args:
        attn: Attention weights (B,T,heads,L,L)
        save_path: Path to save figure
    """
    B, T, heads, L, L = attn.shape
    
    # Average over batch and heads
    attn_avg = attn.mean(axis=(0, 2))  # (T, L, L)
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Attention over time (average across levels)
    attn_time = attn_avg.mean(axis=(1, 2))  # (T,)
    axes[0].plot(attn_time, linewidth=2, color='steelblue')
    axes[0].set_xlabel('Timestep', fontsize=11)
    axes[0].set_ylabel('Average Attention', fontsize=11)
    axes[0].set_title('Attention Evolution Over Time', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Level importance (how much each level is attended to)
    # Sum attention received by each level across all queries
    level_importance = attn_avg.mean(axis=0).sum(axis=0)  # (L,)
    
    axes[1].bar(range(L), level_importance, color='coral', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Price Level', fontsize=11)
    axes[1].set_ylabel('Cumulative Attention', fontsize=11)
    axes[1].set_title('Price Level Importance', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Highlight top-5 levels
    top5_idx = np.argsort(level_importance)[-5:]
    for idx in top5_idx:
        axes[1].patches[idx].set_facecolor('darkred')
        axes[1].patches[idx].set_alpha(0.9)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved attention aggregation to {save_path}")
    
    return fig


def plot_attention_difference(attn_up: np.ndarray, 
                              attn_down: np.ndarray,
                              save_path: Optional[str] = None,
                              figsize: Tuple[int, int] = (10, 8)):
    """Compare attention patterns for different prediction outcomes.
    
    Args:
        attn_up: Attention for "up" predictions (B,T,heads,L,L)
        attn_down: Attention for "down" predictions
        save_path: Path to save figure
    """
    # Average over batch, time, heads
    attn_up_avg = attn_up.mean(axis=(0, 1, 2))    # (L, L)
    attn_down_avg = attn_down.mean(axis=(0, 1, 2))
    
    diff = attn_up_avg - attn_down_avg
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot 1: Up pattern
    sns.heatmap(attn_up_avg, ax=axes[0], cmap='Reds', cbar=True, square=True)
    axes[0].set_title('Up Predictions', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Key Level')
    axes[0].set_ylabel('Query Level')
    
    # Plot 2: Down pattern
    sns.heatmap(attn_down_avg, ax=axes[1], cmap='Blues', cbar=True, square=True)
    axes[1].set_title('Down Predictions', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Key Level')
    axes[1].set_ylabel('Query Level')
    
    # Plot 3: Difference
    sns.heatmap(diff, ax=axes[2], cmap='RdBu_r', center=0, cbar=True, square=True)
    axes[2].set_title('Difference (Up - Down)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Key Level')
    axes[2].set_ylabel('Query Level')
    
    plt.suptitle('Attention Pattern Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved attention difference to {save_path}")
    
    return fig


def analyze_attention_by_outcome(model, dataloader, 
                                 output_dir: str = "artifacts/plots/attention"):
    """Analyze attention patterns grouped by prediction outcomes.
    
    Args:
        model: LiTCVG_Lite model
        dataloader: DataLoader with labeled data
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    
    # Collect attention for different outcomes
    attn_by_label = {0: [], 1: [], 2: []}  # down, flat, up
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader):
            if batch_idx >= 10:  # Limit samples for efficiency
                break
            
            # Extract attention
            attn = extract_graph_attention(model, x)  # (B,T,heads,L,L)
            
            # Group by label (use horizon 10 for simplicity)
            labels = y[10].numpy()
            
            for i, label in enumerate(labels):
                attn_by_label[label].append(attn[i])
    
    # Convert to arrays
    for label in [0, 1, 2]:
        if attn_by_label[label]:
            attn_by_label[label] = np.stack(attn_by_label[label])
    
    # Plot comparisons
    if attn_by_label[0] and attn_by_label[2]:
        plot_attention_difference(
            attn_by_label[2],  # up
            attn_by_label[0],  # down
            save_path=os.path.join(output_dir, "attention_up_vs_down.png")
        )
    
    print(f"✅ Attention analysis complete! Results saved to {output_dir}")


def plot_attention_temporal_evolution(attn: np.ndarray,
                                      level_focus: int = 0,
                                      save_path: Optional[str] = None,
                                      figsize: Tuple[int, int] = (12, 8)):
    """Plot how attention from a specific level evolves over time.
    
    Args:
        attn: Attention weights (B,T,heads,L,L)
        level_focus: Which query level to track
        save_path: Path to save figure
    """
    B, T, heads, L, L = attn.shape
    
    # Average over batch
    attn_avg = attn.mean(axis=0)  # (T, heads, L, L)
    
    # Extract attention from level_focus to all other levels over time
    attn_from_level = attn_avg[:, :, level_focus, :]  # (T, heads, L)
    
    fig, axes = plt.subplots(heads, 1, figsize=figsize)
    if heads == 1:
        axes = [axes]
    
    for h, ax in enumerate(axes):
        # Plot time evolution
        im = ax.imshow(attn_from_level[:, h, :].T, 
                      aspect='auto', cmap='viridis', 
                      interpolation='nearest')
        ax.set_ylabel('Target Level', fontsize=10)
        ax.set_title(f'Head {h+1}: Attention from Level {level_focus}', 
                    fontsize=11, fontweight='bold')
        
        # Colorbar
        plt.colorbar(im, ax=ax, label='Attention Weight')
        
        if h == len(axes) - 1:
            ax.set_xlabel('Timestep', fontsize=10)
    
    plt.suptitle(f'Temporal Evolution of Attention from Level {level_focus}',
                fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved temporal evolution to {save_path}")
    
    return fig


# ===== Main visualization pipeline =====

def visualize_model_attention(model, 
                              sample_input: torch.Tensor,
                              output_dir: str = "artifacts/plots/attention",
                              plot_types: List[str] = ["heatmap", "aggregation", "temporal"]):
    """Complete attention visualization pipeline.
    
    Args:
        model: LiTCVG_Lite model
        sample_input: Sample input tensor (B,T,L,C)
        output_dir: Directory to save plots
        plot_types: Which visualizations to generate
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("🔍 Extracting attention weights...")
    attn = extract_graph_attention(model, sample_input)
    print(f"   Attention shape: {attn.shape}")
    
    if "heatmap" in plot_types:
        print("📊 Plotting level attention heatmaps...")
        plot_level_attention_heatmap(
            attn, 
            timestep=50,  # middle of sequence
            save_path=os.path.join(output_dir, "level_attention_heatmap.png")
        )
    
    if "aggregation" in plot_types:
        print("📊 Plotting attention aggregation...")
        plot_attention_aggregation(
            attn,
            save_path=os.path.join(output_dir, "attention_aggregation.png")
        )
    
    if "temporal" in plot_types:
        print("📊 Plotting temporal evolution...")
        plot_attention_temporal_evolution(
            attn,
            level_focus=0,  # best bid level
            save_path=os.path.join(output_dir, "attention_temporal.png")
        )
    
    print(f"✅ All visualizations saved to {output_dir}")


# ===== Testing =====

def main():
    """Demo visualization with synthetic data"""
    print("🎨 Attention Visualization Demo\n")
    
    # Create synthetic attention patterns
    B, T, heads, L = 4, 100, 2, 20
    
    # Simulate attention with some structure
    attn = np.random.rand(B, T, heads, L, L).astype(np.float32)
    
    # Add diagonal dominance (levels attend to nearby levels)
    for b in range(B):
        for t in range(T):
            for h in range(heads):
                for i in range(L):
                    for j in range(L):
                        distance = abs(i - j)
                        attn[b, t, h, i, j] *= np.exp(-distance / 5.0)
    
    # Normalize
    attn = attn / attn.sum(axis=-1, keepdims=True)
    
    output_dir = "artifacts/plots/attention_demo"
    
    # Test heatmap
    plot_level_attention_heatmap(
        attn, 
        timestep=50,
        save_path=os.path.join(output_dir, "demo_heatmap.png")
    )
    
    # Test aggregation
    plot_attention_aggregation(
        attn,
        save_path=os.path.join(output_dir, "demo_aggregation.png")
    )
    
    # Test temporal
    plot_attention_temporal_evolution(
        attn,
        level_focus=0,
        save_path=os.path.join(output_dir, "demo_temporal.png")
    )
    
    print("\n✅ Demo complete! Check artifacts/plots/attention_demo/")


if __name__ == "__main__":
    main()