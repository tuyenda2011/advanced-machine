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
| **Users** | 124,895 | Valid users after positive-feedback 5-core filtering |
| **Items** | 44,843 | Electronics products |
| **Interactions** | 1,072,740 | Positive feedback edges |
| **Density** | 0.0192% | Extremely sparse graph |
| **Avg. Interactions/User** | 8.59 | Long-tail distribution |
| **Split** | 80% / 10% / 10% | Exact chronological train/validation/test split |

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

### 3.2 XSimGCL Perturbed Propagation

XSimGCL injects sign-aware normalized noise after each propagation layer and contrasts the aggregated recommendation representation with layer $l^*$:

$$
e^{(k)'} = \tilde{A}e^{(k-1)'} + \epsilon\,\operatorname{sign}(e^{(k)'})\odot\frac{\Delta^{(k)}}{\|\Delta^{(k)}\|_2}
$$

$$
\mathcal{L} = \mathcal{L}_{\text{BPR}} + \lambda_{cl}\mathcal{L}_{\text{InfoNCE}}
$$

### 3.3 DirectAU

DirectAU removes negative sampling and directly optimizes normalized positive-pair alignment and batch uniformity:

$$
\mathcal{L}_{\text{DirectAU}} = \mathcal{L}_{\text{align}} + \gamma\mathcal{L}_{\text{uniform}}
$$

### 3.4 AdaptiveGCL (Proposed Course-Project Model)

AdaptiveGCL combines gated ID/text fusion, user semantic profiles, learnable layer attention, debiased graph-text InfoNCE, explicit hard-negative ranking penalties, and normalized Dirichlet-energy regularization. Main benchmark metrics use the shared warm-start protocol; zero-shot inference remains an auxiliary capability rather than a benchmark claim.

### 3.5 Hypersphere Representation Geometry

Based on Wang & Isola (ICML 2020):

| Metric | Formula | Interpretation |
|:------:|:--------|:---------------|
| **Alignment** | $\mathbb{E}[ \|f(u) - f(i)\|_2^2 ]$ | Lower = closer positive pairs |
| **Uniformity** | $\log \mathbb{E}_{u,v} [ \exp(-2\|f(u)-f(v)\|^2) ]$ | Lower = more uniform distribution |

### 3.6 Beyond-Accuracy Metrics

| Metric | Formula | Purpose |
|:------:|:--------|:--------|
| **ILD@K** | $\frac{1}{\|U\|} \sum_u \frac{2}{K(K-1)} \sum_{i<j} (1 - \cos(e_i, e_j))$ | Measure diversity of recommendations |
| **Novelty** | $\frac{1}{\|U\| \cdot K} \sum_{u,i} -\log_2 \frac{\text{count}(i)}{\|U_{\text{train}}\|}$ | Recommending unpopular items |
| **Coverage** | $\frac{\|\bigcup_u R_u\|}{\|I\|}$ | Catalog coverage |
| **Gini** | $\frac{\sum (2i-1)c_{(i)}}{n\sum c_i}$ | Fairness of recommendation distribution |

### 3.7 Algorithm Complexity

| Algorithm | Forward Pass | Contrastive Loss | Memory |
|:----------|:------------:|:---------------:|:------:|
| **LightGCN** | $O(L \cdot E \cdot d)$ | — | $1 \times$ Adj |
| **XSimGCL** | $O(L \cdot E \cdot d)$ | $O(B^2 \cdot d)$ | $1 \times$ Adj |
| **DirectAU** | $O(L \cdot E \cdot d)$ | $O(B^2 \cdot d)$ uniformity | $1 \times$ Adj |
| **AdaptiveGCL** | $O(L \cdot E \cdot d)$ | $O(B^2 \cdot d)$ semantic SSL | $1 \times$ Adj + text |

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

# Full benchmark (3 seeds × 4 sparsity × 100 epochs)
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
- Base Models: [LightGCN](https://github.com/gusye1234/LightGCN-PyTorch), [XSimGCL/SELFRec](https://github.com/Coder-Yu/SELFRec), [DirectAU](https://github.com/THUwangcy/DirectAU)

---

<div align="center">

**⭐ Star this repo if you find it useful!**

*Built with ❤️ for the recommendation systems research community*

</div>
