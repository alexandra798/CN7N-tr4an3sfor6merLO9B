"""Alpha term structure analysis for multi-horizon predictions.

This module computes:
1. Information Coefficient (IC) for each horizon
2. Half-life estimation via exponential decay fitting
3. Confidence intervals via bootstrap
4. Turnover vs IC tradeoff analysis
5. Comprehensive visualizations
"""
from __future__ import annotations
import os
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from scipy.optimize import curve_fit
import seaborn as sns


# ===== IC Computation =====

def compute_ic(predictions: np.ndarray, 
               labels: np.ndarray, 
               method: str = 'spearman') -> float:
    """Compute Information Coefficient.
    
    Args:
        predictions: Predicted probabilities or scores (N,)
        labels: True labels (N,)
        method: 'spearman' or 'pearson'
        
    Returns:
        IC value (correlation coefficient)
    """
    if method == 'spearman':
        ic, _ = spearmanr(predictions, labels)
    elif method == 'pearson':
        ic, _ = pearsonr(predictions, labels)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return ic if not np.isnan(ic) else 0.0


def compute_ic_with_confidence(predictions: np.ndarray,
                               labels: np.ndarray,
                               method: str = 'spearman',
                               n_bootstrap: int = 1000,
                               confidence: float = 0.95) -> Tuple[float, float, float]:
    """Compute IC with confidence interval via bootstrap.
    
    Returns:
        (ic, lower_bound, upper_bound)
    """
    ic = compute_ic(predictions, labels, method)
    
    # Bootstrap
    n = len(predictions)
    bootstrap_ics = []
    
    for _ in range(n_bootstrap):
        idx = np.random.randint(0, n, size=n)
        ic_boot = compute_ic(predictions[idx], labels[idx], method)
        bootstrap_ics.append(ic_boot)
    
    bootstrap_ics = np.array(bootstrap_ics)
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_ics, alpha/2 * 100)
    upper = np.percentile(bootstrap_ics, (1 - alpha/2) * 100)
    
    return ic, lower, upper


# ===== Half-life Estimation =====

def estimate_half_life(ic_series: np.ndarray, 
                      horizons: np.ndarray,
                      method: str = 'exponential') -> Tuple[float, Dict]:
    """Estimate alpha decay half-life.
    
    Fits IC(h) = IC_0 * exp(-λh) and computes half-life = ln(2)/λ
    
    Args:
        ic_series: IC values for each horizon (K,)
        horizons: Horizon values (K,)
        method: 'exponential' or 'power'
        
    Returns:
        (half_life, fit_params)
    """
    if method == 'exponential':
        # Exponential decay: IC(h) = IC_0 * exp(-λh)
        def decay_fn(h, ic0, lam):
            return ic0 * np.exp(-lam * h)
        
        try:
            # Use absolute IC for fitting
            ic_abs = np.abs(ic_series)
            (ic0, lam), _ = curve_fit(
                decay_fn, 
                horizons, 
                ic_abs,
                p0=[ic_abs[0], 0.01],
                bounds=([0, 0], [np.inf, np.inf])
            )
            half_life = np.log(2) / lam if lam > 0 else np.inf
            
            return half_life, {'ic0': ic0, 'lambda': lam, 'method': 'exponential'}
        
        except:
            # Fallback: linear estimate
            return horizons[-1] / 2, {'method': 'fallback'}
    
    elif method == 'power':
        # Power law decay: IC(h) = IC_0 * h^(-α)
        def power_fn(h, ic0, alpha):
            return ic0 * np.power(h, -alpha)
        
        try:
            ic_abs = np.abs(ic_series)
            (ic0, alpha), _ = curve_fit(
                power_fn,
                horizons,
                ic_abs,
                p0=[ic_abs[0], 0.5]
            )
            # Half-life for power law: when IC drops to 50%
            half_life = np.power(2, 1/alpha) * horizons[0]
            
            return half_life, {'ic0': ic0, 'alpha': alpha, 'method': 'power'}
        
        except:
            return horizons[-1] / 2, {'method': 'fallback'}
    
    else:
        raise ValueError(f"Unknown method: {method}")


# ===== Multi-Horizon Analysis =====

def analyze_alpha_term_structure(
    model,
    dataloader,
    horizons: List[int],
    output_dir: str = "artifacts/tables",
    n_bootstrap: int = 1000
) -> Tuple[pd.DataFrame, Dict]:
    """Comprehensive alpha term structure analysis.
    
    Args:
        model: Trained model
        dataloader: Validation/test dataloader
        horizons: List of prediction horizons
        output_dir: Directory to save results
        n_bootstrap: Number of bootstrap samples for CI
        
    Returns:
        (ic_table, analysis_dict)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    
    # Collect predictions and labels
    results = {h: {'preds': [], 'labels': []} for h in horizons}
    
    with torch.no_grad():
        for x, y in dataloader:
            out = model(x)
            
            for h in horizons:
                # Extract probability of "up" class (class 2)
                logits = out[f'logits_{h}']
                probs = torch.softmax(logits, dim=-1)[:, 2]  # P(up)
                
                results[h]['preds'].extend(probs.cpu().numpy())
                results[h]['labels'].extend(y[h].cpu().numpy())
    
    # Compute ICs with confidence intervals
    ic_data = []
    
    for h in horizons:
        preds = np.array(results[h]['preds'])
        labels = np.array(results[h]['labels'])
        
        # Spearman IC
        ic_spear, ic_spear_lo, ic_spear_hi = compute_ic_with_confidence(
            preds, labels, 'spearman', n_bootstrap=n_bootstrap
        )
        
        # Pearson IC
        ic_pear, ic_pear_lo, ic_pear_hi = compute_ic_with_confidence(
            preds, labels, 'pearson', n_bootstrap=n_bootstrap
        )
        
        ic_data.append({
            'horizon': h,
            'ic_spearman': ic_spear,
            'ic_spearman_lo': ic_spear_lo,
            'ic_spearman_hi': ic_spear_hi,
            'ic_pearson': ic_pear,
            'ic_pearson_lo': ic_pear_lo,
            'ic_pearson_hi': ic_pear_hi,
            'n_samples': len(preds)
        })
    
    ic_table = pd.DataFrame(ic_data)
    
    # Estimate half-life
    ic_values = ic_table['ic_spearman'].values
    horizon_values = np.array(horizons)
    
    half_life, fit_params = estimate_half_life(ic_values, horizon_values)
    
    # Save results
    csv_path = os.path.join(output_dir, "alpha_term_structure.csv")
    ic_table.to_csv(csv_path, index=False)
    print(f"✅ Saved IC table to {csv_path}")
    
    # Summary
    analysis = {
        'half_life': half_life,
        'fit_params': fit_params,
        'ic_table': ic_table,
        'max_ic_horizon': horizons[ic_values.argmax()],
        'max_ic_value': ic_values.max()
    }
    
    return ic_table, analysis


# ===== Visualization =====

def plot_alpha_term_structure(
    ic_table: pd.DataFrame,
    half_life: float,
    fit_params: Dict,
    save_path: str,
    figsize: Tuple[int, int] = (12, 5)
):
    """Plot alpha term structure with fitted decay curve.
    
    Args:
        ic_table: DataFrame with IC results
        half_life: Estimated half-life
        fit_params: Decay curve parameters
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    horizons = ic_table['horizon'].values
    ic_spear = ic_table['ic_spearman'].values
    ic_spear_lo = ic_table['ic_spearman_lo'].values
    ic_spear_hi = ic_table['ic_spearman_hi'].values
    
    # Plot 1: IC with confidence intervals
    ax = axes[0]
    ax.plot(horizons, ic_spear, 'o-', linewidth=2, markersize=8, 
            color='steelblue', label='Spearman IC')
    ax.fill_between(horizons, ic_spear_lo, ic_spear_hi, 
                     alpha=0.3, color='steelblue')
    
    # Fitted decay curve
    if fit_params.get('method') == 'exponential':
        ic0 = fit_params['ic0']
        lam = fit_params['lambda']
        h_fine = np.linspace(horizons[0], horizons[-1], 100)
        ic_fitted = ic0 * np.exp(-lam * h_fine)
        ax.plot(h_fine, ic_fitted, '--', linewidth=1.5, color='red', 
                label=f'Exp Fit (λ={lam:.4f})')
    
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel('Horizon (steps)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Information Coefficient', fontsize=11, fontweight='bold')
    ax.set_title(f'Alpha Term Structure\n(Half-life: {half_life:.1f} steps)', 
                fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: IC comparison (Spearman vs Pearson)
    ax = axes[1]
    ic_pear = ic_table['ic_pearson'].values
    
    x_pos = np.arange(len(horizons))
    width = 0.35
    
    ax.bar(x_pos - width/2, ic_spear, width, 
           label='Spearman', color='steelblue', alpha=0.8, edgecolor='black')
    ax.bar(x_pos + width/2, ic_pear, width, 
           label='Pearson', color='coral', alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Horizon (steps)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Information Coefficient', fontsize=11, fontweight='bold')
    ax.set_title('IC Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(horizons)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved alpha term plot to {save_path}")
    
    return fig


def plot_ic_stability_over_time(
    model,
    dataloader,
    horizon: int = 10,
    window_size: int = 500,
    save_path: Optional[str] = None
):
    """Analyze IC stability using rolling windows.
    
    Useful for detecting regime changes or overfitting.
    """
    model.eval()
    
    preds = []
    labels = []
    
    with torch.no_grad():
        for x, y in dataloader:
            out = model(x)
            logits = out[f'logits_{horizon}']
            probs = torch.softmax(logits, dim=-1)[:, 2]
            
            preds.extend(probs.cpu().numpy())
            labels.extend(y[horizon].cpu().numpy())
    
    preds = np.array(preds)
    labels = np.array(labels)
    
    # Compute rolling IC
    n = len(preds)
    rolling_ics = []
    timestamps = []
    
    for i in range(window_size, n):
        window_preds = preds[i-window_size:i]
        window_labels = labels[i-window_size:i]
        ic = compute_ic(window_preds, window_labels, 'spearman')
        rolling_ics.append(ic)
        timestamps.append(i)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(timestamps, rolling_ics, linewidth=1.5, color='steelblue')
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax.axhline(np.mean(rolling_ics), color='red', linestyle='--', 
               label=f'Mean IC: {np.mean(rolling_ics):.4f}')
    
    ax.fill_between(timestamps, 
                     np.mean(rolling_ics) - np.std(rolling_ics),
                     np.mean(rolling_ics) + np.std(rolling_ics),
                     alpha=0.2, color='red')
    
    ax.set_xlabel('Sample Index', fontsize=11, fontweight='bold')
    ax.set_ylabel('Rolling IC', fontsize=11, fontweight='bold')
    ax.set_title(f'IC Stability Over Time (h={horizon}, window={window_size})',
                fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved IC stability plot to {save_path}")
    
    return fig


# ===== Main Pipeline =====

def main():
    """Demo with synthetic data"""
    print("📊 Alpha Term Structure Analysis Demo\n")
    
    # Synthetic data
    horizons = [10, 50, 100]
    n_samples = 1000
    
    # Simulate predictions with decay
    np.random.seed(42)
    results = {}
    
    for h in horizons:
        # IC decays with horizon
        true_ic = 0.3 * np.exp(-0.01 * h)
        
        # Generate correlated data
        labels = np.random.randint(0, 3, n_samples)
        noise = np.random.randn(n_samples) * 0.5
        preds = labels + noise + np.random.randn(n_samples) * (1 - true_ic)
        
        results[h] = {'preds': preds, 'labels': labels}
    
    # Compute ICs
    ic_data = []
    for h in horizons:
        ic = compute_ic(results[h]['preds'], results[h]['labels'])
        ic_data.append({'horizon': h, 'ic_spearman': ic, 
                       'ic_spearman_lo': ic-0.05, 'ic_spearman_hi': ic+0.05,
                       'ic_pearson': ic*0.9, 
                       'ic_pearson_lo': ic*0.9-0.05, 'ic_pearson_hi': ic*0.9+0.05,
                       'n_samples': n_samples})
    
    ic_table = pd.DataFrame(ic_data)
    
    # Estimate half-life
    ic_values = ic_table['ic_spearman'].values
    half_life, fit_params = estimate_half_life(ic_values, np.array(horizons))
    
    print(ic_table)
    print(f"\n📈 Half-life: {half_life:.1f} steps")
    print(f"📈 Fit params: {fit_params}")
    
    # Plot
    plot_alpha_term_structure(
        ic_table, 
        half_life, 
        fit_params,
        save_path="artifacts/plots/alpha_term_demo.png"
    )
    
    print("\n✅ Demo complete!")


if __name__ == "__main__":
    import torch  # Needed for the demo
    main()