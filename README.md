# LiTCVG-HFT:Lightweight Transformer with Cross-level Value Graph for High-Frequency Trading


[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://isocpp.org/)

> **End-to-end deep learning system for microsecond-level limit order book prediction**  
> From research prototype to production-ready deployment, bridging academic innovation with industrial-grade performance

---

## 📌 Project Overview

This project implements a complete high-frequency trading prediction system that analyzes limit order books (LOB) in real-time and forecasts future price movement directions. The system employs Python for model training and research, with C++ implementation for low-latency inference engine achieving production-grade performance.

### Key Features

- **🧠 Novel Architecture**: LiT-CVG Transformer specifically designed for order book microstructure
- **⚡ Ultra-Low Latency**: C++ inference engine with <500µs per prediction
- **🎯 Multi-Horizon Prediction**: Joint forecasting of short (10-step), medium (50-step), and long-term (100-step) price movements
- **📊 End-to-End Pipeline**: From raw LOB data to real-time trading signals
- **🔬 Interpretability**: Attention visualization, alpha decay analysis, information coefficient computation
- **🚀 Production-Ready**: ONNX export, multi-threaded pipeline, performance monitoring

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     TRAINING PIPELINE (Python)               │
├─────────────────────────────────────────────────────────────┤
│  Raw LOB Data → Feature Engineering → Model Training        │
│  ├─ OFI Computation    ├─ CVMix Layer                      │
│  ├─ Normalization      ├─ Graph Attention                   │
│  └─ Windowing          └─ Multi-Horizon Heads               │
└─────────────────────────────────────────────────────────────┘
                            ↓ ONNX Export
┌─────────────────────────────────────────────────────────────┐
│                  INFERENCE ENGINE (C++/ONNX)                │
├─────────────────────────────────────────────────────────────┤
│  Market Data → Feature Extraction → Inference → Signals     │
│  [Thread 1]    [Thread 2]           [Thread 3]              │
│  Ring Buffer → Feature Queue → Prediction Queue             │
└─────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Training Framework** | PyTorch 2.0+ | Model development & training |
| **Inference Engine** | ONNX Runtime + C++20 | Production deployment |
| **Data Processing** | NumPy, Pandas | Feature engineering |
| **Visualization** | Matplotlib, Seaborn | Results analysis |
| **Build System** | CMake 3.20+ | C++ project management |
| **Testing** | pytest, Custom C++ tests | Quality assurance |

---

## 🎯 Model Innovation: LiT-CVG Architecture

### Core Components

#### 1️⃣ **CVMix (Cross-level Value Mixing)**
```python
# Innovation: Cross-price-level feature mixing + channel doubling
Input:  (B, T, L, C)     # L=20 price levels, C=7 feature channels
Output: (B, T, L, 2C)    # Channel doubled, preserving original + mixed features
```
- **Purpose**: Capture interactions between different price levels (e.g., spread compression, liquidity migration)
- **Implementation**: Learnable linear transformation across L dimension + GELU activation
- **Advantage**: Better preservation of price level structure compared to traditional CNNs

#### 2️⃣ **TinyGraphSummary (Graph Attention Aggregation)**
```python
# Multi-head attention captures price level relationships
Query/Key/Value: (B, T, L, 2C) → (B, T, heads, L, d_k)
Attention: (B, T, heads, L, L)  # Inter-level interaction matrix
Output: (B, T, g_dim)           # Aggregated graph embedding
```
- **Purpose**: Learn which price levels are most important for prediction (typically near best bid/ask)
- **Interpretability**: Attention weight visualization reveals market microstructure focus

#### 3️⃣ **Patch Embedding (Spatiotemporal Patchification)**
```python
# Split (T,L) into (pT×pL) patches
T=100, L=20 → patches=(10, 2) with pT=10, pL=10
Each patch: pT×pL×2C → d_model (learnable projection)
```
- **Purpose**: Reduce computational complexity while preserving local spatiotemporal patterns
- **Inspiration**: Adapted from ViT for LOB's spatiotemporal characteristics

#### 4️⃣ **Multi-Horizon Prediction Heads**
```python
# Joint prediction across multiple time horizons
logits_10:  Predict price direction in next 10 steps (short-term)
logits_50:  Predict price direction in next 50 steps (medium-term)
logits_100: Predict price direction in next 100 steps (long-term)
```
- **Advantage**: Single model captures both short-term fluctuations and long-term trends
- **Alpha Decay**: Prediction capability across horizons decays exponentially, aligning with market efficiency hypothesis

### Complete Forward Pass
```
Input LOB (B,T,L,C=7)
    ↓ CVMix
(B,T,L,2C=14)
    ↓ Graph Attention
(B,T,g_dim=16) + (B,N,d_model=64)
    ↓ Transformer Blocks (depth=2)
(B,N,d_model)
    ↓ Pooling + Concat
(B, d_model+g_dim)
    ↓ Multi-Horizon Heads
logits_10, logits_50, logits_100 (each: B×3)
```

---

## 🔬 Feature Engineering

### Raw LOB Features (per price level)
| Feature | Description | Formula |
|---------|-------------|---------|
| `bid_size` | Bid order quantity | - |
| `ask_size` | Ask order quantity | - |
| `OFI` | Order Flow Imbalance | `Δbid_size - Δask_size` |
| `spread` | Bid-ask spread | `ask_p - bid_p` |
| `mid` | Mid-price | `(bid_p + ask_p) / 2` |
| `dmid` | Mid-price change | `mid_t - mid_{t-1}` |
| `cum_vol` | Cumulative volume | `∑ trade_vol` |

### Optional Advanced Features
```python
FeatureConfig(
    include_price=True,             # Normalized price levels
    include_volume_profile=True,    # Volume distribution
    ofi_levels=5                    # Multi-level OFI
)
```

### Data Normalization
- **Method**: Welford online algorithm for mean/std computation
- **Purpose**: Training stabilization and convergence acceleration
- **Storage**: JSON format for easy C++ loading

---

## ⚡ C++ Inference Engine

### Performance Metrics (Single-threaded, CPU)
| Component | P50 Latency | P99 Latency | Throughput |
|-----------|-------------|-------------|------------|
| Feature Extraction | 2.5 µs | 4.8 µs | 400K ops/s |
| Normalization | 0.8 µs | 1.2 µs | 1.2M ops/s |
| ONNX Inference | 350 µs | 520 µs | 2.8K infer/s |
| **End-to-End** | **360 µs** | **530 µs** | **2.7K pred/s** |

### Multi-threaded Pipeline Architecture
```cpp
Thread 1: Market Data Receiver
    ↓ RingBuffer(4096)
Thread 2: Feature Extraction + Normalization
    ↓ RingBuffer(2048)
Thread 3: Model Inference + Signal Generation
```

**Advantages**:
- ✅ Lock-free design minimizes thread contention
- ✅ Pipeline parallelism maximizes throughput
- ✅ Backpressure mechanism prevents memory overflow

### Key Optimization Techniques
1. **Memory Pool**: Pre-allocated ring buffers avoid dynamic allocation
2. **SIMD**: Feature computation leverages vectorized instructions
3. **Cache-Friendly**: Data structures aligned to cache lines
4. **Compiler Optimization**: `-O3 -march=native -flto`

---

## 📊 Experimental Results

### Dataset: FI-2010
- **Description**: Real LOB data from Finnish Stock Exchange
- **Scale**: 254,701 training samples, 139,488 test samples
- **Classes**: 3 classes (down, stationary, up)

### Performance Comparison (from original Kaggle notebook)
| Model | Test Accuracy | Parameters | Inference Speed |
|-------|--------------|------------|-----------------|
| CNN-LSTM-DeepLOB | 73.4% | ~500K | ~5ms |
| **LiT-CVG (This Project)** | **77.1%** | **197K** | **<0.5ms** |

### Alpha Decay Analysis
```
Horizon    IC (Spearman)   Half-life
10-step    0.285           ~45 steps
50-step    0.178           (exponential decay)
100-step   0.092           
```
- **Interpretation**: Short-term prediction strongest, aligning with market microstructure theory
- **Application**: Can be used to dynamically adjust holding periods

---

## 🚀 Quick Start

### Environment Setup

```bash
# Python environment
conda create -n litcvg python=3.10
conda activate litcvg
pip install torch numpy pandas matplotlib seaborn scikit-learn pyyaml

# C++ dependencies (macOS example)
brew install onnxruntime

# Or build from source (Linux)
git clone https://github.com/microsoft/onnxruntime
cd onnxruntime
./build.sh --config Release --parallel
```

### Training Model

```bash
# 1. Generate normalization statistics
python python0/train.py --config config/train.yaml --make-norm

# 2. Train model (supports FI-2010 or synthetic data)
python python0/train.py --config config/train.yaml

# 3. Export to ONNX
python python0/export_onnx.py \
    --config config/export.yaml \
    --ckpt artifacts/ckpt.pt \
    --out artifacts/model.onnx

# 4. Evaluate model
python python0/evaluate.py \
    --config config/export.yaml \
    --ckpt artifacts/ckpt.pt
```

### C++ Inference

```bash
# Compile
cd cpp && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Performance benchmark
./perf_benchmark artifacts/model.onnx artifacts/norm_stats.json

# Multi-threaded streaming inference
./main_stream_threaded ../config/runtime.yaml

# Visualize results
python python0/explain/visualize_perf.py
```

---

## 🔬 Interpretability & Analysis

### 1. Attention Visualization
```bash
python python0/explain/attn_viz.py \
    --config config/train.yaml \
    --ckpt artifacts/ckpt.pt
```
Generates:
- Price level attention heatmaps
- Temporal attention evolution plots
- Attention differences across prediction outcomes

### 2. Alpha Decay Analysis
```bash
python python0/explain/alpha_term.py \
    --config config/train.yaml \
    --ckpt artifacts/ckpt.pt
```
Outputs:
- Information Coefficient (IC) confidence intervals
- Half-life estimation
- IC stability analysis

### 3. Transaction Cost Analysis (TCA)
```cpp
// C++ implementation
TCAConfig cfg{
    .fee = 0.0001,           // 0.01% commission
    .slip_bps = 0.5,         // 0.5 bps slippage
    .half_spread = 0.005,    // 0.5¢ bid-ask spread
    .latency_us = 100        // 100μs latency
};

TCAResult result = backtest_multi_horizon(
    signal_10, signal_50, signal_100, cfg
);
// Outputs: PnL, Sharpe ratio, turnover
```

---

## 📂 Project Structure

```
litcvg-hft/
├── python0/                    # Python training pipeline
│   ├── data/                   # Data processing
│   │   ├── features.py        # Feature engineering (OFI, normalization)
│   │   ├── fi2010.py          # FI-2010 data loader
│   │   ├── loaders.py         # Data pipeline builder
│   │   └── windows.py         # Sliding window
│   ├── model/                  # Model definitions
│   │   ├── lit_cvg.py         # LiT-CVG architecture
│   │   └── losses.py          # Multi-horizon loss functions
│   ├── explain/                # Interpretability
│   │   ├── attn_viz.py        # Attention visualization
│   │   └── alpha_term.py      # Alpha decay analysis
│   ├── train.py               # Training script
│   ├── export_onnx.py         # ONNX export
│   └── evaluate.py            # Model evaluation
├── cpp/                        # C++ inference engine
│   ├── include/               # Header files
│   │   ├── session.hpp        # ONNX Runtime wrapper
│   │   ├── preproc.hpp        # Feature preprocessing
│   │   ├── fe_ofi.hpp         # OFI computation
│   │   ├── ring_buffer.hpp    # Lock-free ring buffer
│   │   └── perf_stats.hpp     # Performance statistics
│   ├── src/                   # Implementation
│   │   ├── main_stream_threaded.cpp  # Multi-threaded inference
│   │   ├── perf_benchmark.cpp        # Performance testing
│   │   └── tca.cpp            # Transaction cost analysis
│   └── CMakeLists.txt         # Build configuration
├── config/                     # Configuration files
│   ├── train.yaml             # Training parameters
│   ├── export.yaml            # Export configuration
│   └── runtime.yaml           # Inference configuration
├── tests/                      # Test suite
│   ├── test_parity_*.py       # Python/C++ parity tests
│   └── latency_smoke.py       # Latency smoke tests
└── artifacts/                  # Output directory
    ├── model.onnx             # Exported model
    ├── norm_stats.json        # Normalization parameters
    ├── ckpt.pt                # PyTorch checkpoint
    ├── plots/                 # Visualization results
    └── perf/                  # Performance reports
```

---

## 🎓 Theoretical Foundation

### Market Microstructure
1. **Order Flow Imbalance (OFI)**: Key metric for measuring buy/sell pressure
   - Reference: Cont et al. (2014) "The Price Impact of Order Book Events"
   
2. **Price Discovery Mechanism**: How LOB reflects market information
   - Theory: Information asymmetry and adverse selection

3. **Alpha Decay**: Predictive signals decay over time
   - Formula: `IC(h) = IC₀ · exp(-λh)`
   - Half-life: `t₁/₂ = ln(2)/λ`

### Deep Learning Methodology
1. **Transformer for Time Series**: 
   - Attention mechanism captures long-range dependencies
   - Patch embedding reduces complexity

2. **Multi-Task Learning**:
   - Joint optimization across horizons improves generalization
   - Adaptive weighting strategy

3. **Domain Adaptation**:
   - Cross-market transfer learning (future work)

---

## 📈 Future Work

### Short-term Plans
- [ ] Add reinforcement learning layer for execution optimization
- [ ] Support more market data sources (futures, options)
- [ ] GPU inference acceleration (CUDA/TensorRT)

### Medium-term Plans
- [ ] Multi-asset portfolio optimization
- [ ] Market impact model integration
- [ ] Real-time risk management

### Long-term Vision
- [ ] Complete trading system (OMS + risk + monitoring)
- [ ] Distributed backtesting framework
- [ ] Production-grade monitoring and alerting

---

## 🤝 Technical Highlights (Interview Focus)

### 1. System Design Capability
- **End-to-End Thinking**: Complete pipeline from research to production
- **Performance Optimization**: Python prototype → C++ production, 100× speedup
- **Multi-threaded Architecture**: Lock-free ring buffer, zero-copy design

### 2. Algorithmic Innovation
- **CVMix Layer**: Novel design tailored for LOB structure
- **Multi-Horizon Prediction**: Capturing multiple market timescales simultaneously
- **Adaptive Loss**: Weighting strategy based on alpha decay theory

### 3. Engineering Practice
- **Code Quality**: Complete unit tests, parity tests
- **Maintainability**: Clear modular design, YAML config management
- **Scalability**: Pluggable data sources, features, models

### 4. Quantitative Finance Knowledge
- **Market Microstructure**: OFI, spread, liquidity
- **Transaction Cost Analysis**: Slippage, market impact modeling
- **Alpha Research**: IC analysis, half-life estimation

---

## 📚 References

1. **DeepLOB**: Zhang et al. (2019) - "DeepLOB: Deep Convolutional Neural Networks for Limit Order Books"
2. **Transformer**: Vaswani et al. (2017) - "Attention Is All You Need"
3. **Vision Transformer**: Dosovitskiy et al. (2020) - "An Image is Worth 16x16 Words"
4. **Order Flow Imbalance**: Cont et al. (2014) - "The Price Impact of Order Book Events"
5. **FI-2010 Dataset**: Ntakaris et al. (2018) - "Benchmark Dataset for Mid-Price Forecasting"

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details

