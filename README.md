# ⚡ Advanced Graph Contrastive Learning for Recommendation Systems

> **Nghiên cứu chuyên sâu về Graph Neural Networks trong bài toán gợi ý Top-K dưới điều kiện dữ liệu thưa**

[![Python](https://img.shields.io/badge/Python-3.10|3.11-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-Academic%20Research-6C757D?style=flat)]()

---

## 📋 Mục Lục

| Phần | Nội dung |
|:---:|:---|
| [🚀](#-quick-start) | Quick Start |
| [📊](#-datasets) | Tập Dữ Liệu |
| [🧮](#-mathematical-foundations) | Cơ Sở Toán Học |
| [🏗️](#-project-structure) | Cấu Trúc Dự Án |
| [⚙️](#-installation) | Hướng Dẫn Cài Đặt |
| [🎯](#-usage) | Cách Sử Dụng |
| [📈](#-dashboard) | Dashboard Streamlit |
| [📚](#-references) | Tài Liệu Tham Khảo |

---

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/.../advanced-machine.git
cd advanced-machine

# Install dependencies
pip install -r requirements.txt

# Prepare data
python scripts/prepare_data.py

# Train models
python scripts/train.py --model lightgcn --epochs 50

# Launch dashboard
streamlit run app/streamlit_app.py
```

---

## 📊 Datasets

### Amazon Electronics (5-core)

| Metric | Value | Description |
|:------:|:-----:|:------------|
| **Users** | 135,996 | Valid users after K-core filtering |
| **Items** | 62,749 | Electronics products |
| **Interactions** | 1,173,135 | Positive feedback edges |
| **Density** | 0.0137% | Extremely sparse graph |
| **Avg. Interactions/User** | 8.63 | Long-tail distribution |

### Data Sources

| Resource | Link |
|:---------|:-----|
| 📥 Reviews (1.68M) | [SNAP Stanford](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz) |
| 📥 Metadata (498K) | [SNAP Stanford](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Electronics.json.gz) |
| 🔗 Kaggle | [Amazon Electronics Reviews](https://www.kaggle.com/datasets/omer2241/amazon-electronics-reviews-5-core) |

---

## 🧮 Mathematical Foundations

### 3.1 Graph Propagation & BPR Loss (LightGCN)

Linear message passing via symmetric normalized adjacency:

$$
E^{(k+1)} = \tilde{A} E^{(k)}, \quad \tilde{A} = D^{-\frac{1}{2}} A D^{-\frac{1}{2}}
$$

Layer combination:

$$
E = \frac{1}{L+1} \sum_{k=0}^L E^{(k)}
$$

BPR Loss:

$$
\mathcal{L}_{\text{BPR}} = \sum_{(u,i,j)} -\ln \sigma(\hat{y}_{ui} - \hat{y}_{uj}) + \lambda \|\Theta_0\|_2^2
$$

---

### 3.2 Edge Dropout Contrastive Learning (SGL)

Two augmented views via edge dropout $p_{\text{drop}} = 0.1$:

$$
z_u^{(1)} = \text{LightGCN}(G_1, u), \quad z_u^{(2)} = \text{LightGCN}(G_2, u)
$$

InfoNCE Loss:

$$
\mathcal{L}_{\text{SSL}} = -\sum_{u} \log \frac{\exp(\text{sim}(z_u^{(1)}, z_u^{(2)}) / \tau)}{\sum_v \exp(\text{sim}(z_u^{(1)}, z_v^{(2)}) / \tau)}
$$

---

### 3.3 Representation Noise Perturbation (SimGCL)

Direct L2-normalized noise injection (no graph augmentation):

$$
e^{(k)'} = e^{(k)} + \epsilon \cdot \frac{\Delta}{\|\Delta\|_2}, \quad \Delta \sim U(0,1), \ \epsilon = 0.1
$$

---

### 3.4 Hypersphere Representation Geometry

Based on Wang & Isola (ICML 2020):

| Metric | Formula | Interpretation |
|:------:|:--------|:---------------|
| **Alignment** | $\mathbb{E}[ \|f(u) - f(i)\|_2^2 ]$ | Lower = closer positive pairs |
| **Uniformity** | $\log \mathbb{E}_{u,v} [ \exp(-2\|f(u)-f(v)\|^2) ]$ | Lower = more uniform distribution |

### 3.5 Beyond-Accuracy Metrics

| Metric | Formula | Purpose |
|:------:|:--------|:--------|
| **ILD@K** | $\frac{1}{\|U\|} \sum_u \frac{2}{K(K-1)} \sum_{i<j} (1 - \cos(e_i, e_j))$ | Measure diversity of recommendations |
| **Novelty** | $\frac{1}{\|U\| \cdot K} \sum_{u,i} -\log_2 \frac{\text{count}(i)}{\|U_{\text{train}}\|}$ | Recommending unpopular items |
| **Coverage** | $\frac{\|\bigcup_u R_u\|}{\|I\|}$ | Catalog coverage |
| **Gini** | $\frac{\sum (2i-1)c_{(i)}}{n\sum c_i}$ | Fairness of recommendation distribution |

### 3.6 Algorithm Complexity

| Algorithm | Forward Pass | Contrastive Loss | Memory |
|:----------|:------------:|:---------------:|:------:|
| **LightGCN** | $O(L \cdot E \cdot d)$ | — | $1 \times$ Adj |
| **SGL** | $O(3L \cdot E \cdot d)$ | $O(B^2 \cdot d)$ | $3 \times$ Adj |
| **SimGCL** | $O(L \cdot E \cdot d)$ | $O(B^2 \cdot d)$ | $1 \times$ Adj |

> **Legend**: $E$ = edges, $L$ = layers, $d$ = embedding dim (64), $B$ = batch size (2048)

---

## 🏗️ Project Structure

```
advanced-machine/
│
├── 📁 configs/                    # YAML configuration files
│   ├── common.yaml               # Shared hyperparameters
│   ├── lightgcn.yaml
│   ├── xsimgcl.yaml
│   └── adaptive_gcl.yaml
│
├── 📁 src/                       # Source code
│   ├── 📁 data/                 # Data processing
│   │   ├── preprocessing.py      # Amazon data preprocessing
│   │   ├── splitter.py          # Train/Val/Test splitting
│   │   ├── sparsity.py          # Sparsity sampling
│   │   ├── validation.py        # Data validation
│   │   └── graph.py            # Graph construction
│   │
│   ├── 📁 models/               # Model implementations
│   │   ├── base.py              # BaseRecommender ABC
│   │   ├── lightgcn.py
│   │   ├── xsimgcl.py
│   │   ├── directau.py
│   │   └── adaptive_gcl.py     # Multimodal Gated GCL
│   │
│   ├── 📁 losses/              # Loss functions
│   │   ├── bpr.py
│   │   ├── contrastive.py
│   │   └── directau.py
│   │
│   ├── 📁 training/            # Training pipeline
│   │   ├── trainer.py           # Main training loop
│   │   ├── early_stopping.py
│   │   └── loss_strategies.py  # Strategy pattern
│   │
│   ├── 📁 evaluation/           # Metrics & analysis
│   │   ├── metrics.py           # Recall, NDCG, MRR
│   │   ├── evaluator.py         # Full-ranking evaluator
│   │   ├── representation.py     # Alignment, Uniformity, SVD
│   │   ├── subgroup.py         # Head/Tail analysis
│   │   └── significance.py      # Statistical tests
│   │
│   └── 📁 utils/                # Utilities
│       ├── config.py
│       ├── checkpoints.py        # Checkpoint management
│       ├── geometry.py           # Shared geometry metrics
│       └── config_schemas.py    # Config validation
│
├── 📁 scripts/                  # Executable scripts
│   ├── prepare_data.py          # Data download & preprocessing
│   ├── train.py                # Single model training
│   ├── train_all_models.py      # Multi-model training
│   ├── benchmark_all.py        # Full benchmark suite
│   └── generate_plots.py        # Research visualizations
│
├── 📁 app/                      # Web application
│   └── streamlit_app.py         # Interactive dashboard
│
├── 📁 results/                 # Output directory
│   ├── checkpoints/            # Model weights
│   ├── history/               # Training logs
│   ├── aggregated/            # Summary CSVs
│   └── figures/               # Generated plots
│
└── 📄 README.md               # This file
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.10 or 3.11
- CUDA 11.8+ (for GPU acceleration)
- 8GB+ RAM

### Setup

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Required Packages

| Package | Version | Purpose |
|:--------|:-------:|:--------|
| torch | 2.0+ | Deep learning framework |
| pandas | latest | Data manipulation |
| numpy | latest | Numerical computing |
| pyyaml | latest | Config loading |
| streamlit | 1.25+ | Web dashboard |
| sentence-transformers | latest | Text encoding |
| scipy | latest | Scientific computing |
| tqdm | latest | Progress bars |

---

## 🎯 Usage

### Data Preparation

```bash
# Download and preprocess Amazon Electronics data
python scripts/prepare_data.py
```

### Model Training

```bash
# Train single model
python scripts/train.py --model lightgcn --sparsity 1.0 --epochs 50

# Train with specific seed and resume capability
python scripts/train.py --model xsimgcl --sparsity 0.75 --seed 2025 --epochs 100 --resume

# Train all models at once
python scripts/train_all_models.py --all_sparsity
```

### Benchmark Suite

```bash
# Quick test (1 seed, 5 epochs)
python scripts/benchmark_all.py --quick

# Full benchmark (3 seeds × 4 sparsity × 50 epochs)
python scripts/benchmark_all.py
```

### Generate Plots

```bash
# Create research visualizations
python scripts/generate_plots.py
```

---

## 📈 Streamlit Dashboard

Launch the interactive dashboard:

```bash
streamlit run app/streamlit_app.py
```

### Dashboard Features

| Tab | Description |
|:---:|:-----------|
| 🎯 **Interactive Recommendation** | User-based Top-10 recommendations with real-time metrics |
| 📊 **Benchmark & Significance** | Comprehensive results with statistical tests |
| 🌐 **Representation Geometry** | Alignment vs Uniformity analysis, SVD spectrum |
| 📉 **Sparsity & Subgroups** | Long-tail vs Head user performance |
| 📘 **Theoretical Foundations** | Mathematical formulations & complexity analysis |

---

## 📚 References

```bibtex
@inproceedings{he2020lightgcn,
  title={LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation},
  author={He, Xiangnan and Deng, Kuan and Wang, Xiang and Li, Yan and Zhang, Yongdong and Wang, Meng},
  booktitle={SIGIR},
  pages={639--648},
  year={2020}
}

@inproceedings{wu2021sgl,
  title={Self-supervised Graph Learning for Recommendation},
  author={Wu, Jiancan and Wang, Xiang and Feng, Fuli and He, Xiangnan and others},
  booktitle={SIGIR},
  pages={726--735},
  year={2021}
}

@inproceedings{yu2022simgcl,
  title={Are Graph Augmentations Necessary? Simple Graph Contrastive Learning for Recommendation},
  author={Yu, Junliang and Yin, Hongzhi and Xia, Xin and others},
  booktitle={SIGIR},
  pages={1294--1303},
  year={2022}
}

@inproceedings{wang2020understanding,
  title={Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere},
  author={Wang, Tongzhou and Isola, Phillip},
  booktitle={ICML},
  pages={9929--9939},
  year={2020}
}
```

---

## 🙏 Acknowledgments

- Dataset: [UCSD/Stanford SNAP Lab](https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_data)
- Base Models: [LightGCN](https://github.com/gusye1234/LightGCN-PyTorch), [XSimGCL](https://github.com/YuWVandy/XSimGCL)

---

<div align="center">

**⭐ Star this repo if you find it useful!**

*Built with ❤️ for the recommendation systems research community*

</div>
