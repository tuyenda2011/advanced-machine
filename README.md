# ⚡ Nghiên Cứu Chuyên Sâu Graph Contrastive Learning Cho Hệ Thống Gợi Ý Đồ Điện Tử Dưới Điều Kiện Dữ Liệu Thưa

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/pytest-19%2F19%20Passing-brightgreen.svg)](https://docs.pytest.org/)
[![License](https://img.shields.io/badge/License-Academic%20Research-green.svg)]()

Dự án nghiên cứu thực nghiệm và phân tích toán học chuyên sâu so sánh ba phương pháp Graph Neural Networks tiêu biểu trong bài toán Gợi ý Top-K (**Top-K Electronics Recommendation**) trên tập dữ liệu chuẩn **Amazon Electronics (5-core)** dưới các mức độ thưa dữ liệu khác nhau (100%, 75%, 50%, 25%):
1. **LightGCN** (SIGIR '20): Graph Collaborative Filtering chuẩn hóa tối giản (loại bỏ phi tuyến, ma trận trọng số và self-connections).
2. **SGL (Self-Supervised Graph Learning)** (SIGIR '21): Học tương phản đồ thị dựa trên **Edge Dropout** tạo 2 đồ thị con augmented ($G_1, G_2$).
3. **SimGCL (Simple Graph Contrastive Learning)** (SIGIR '22): Học tương phản đồ thị dựa trên **Embedding Uniform Noise Perturbation** trực tiếp trên không gian biểu diễn ẩn mà không cần tái tạo cấu trúc đồ thị.

---

## 📌 Mục Lục
- [1. Câu Hỏi Nghiên Cứu (Research Questions)](#1-câu-hỏi-nghiên-cứu-research-questions)
- [2. Tập Dữ Liệu & Link Tải Dữ Liệu](#2-tập-dữ-liệu--link-tải-dữ-liệu)
- [3. Cơ Sở Toán Học & Các Hàm Mục Tiêu](#3-cơ-sở-toán-học--các-hàm-mục-tiêu)
  - [3.1 Lan Truyền Đồ Thị & BPR Loss (LightGCN)](#31-lan-truyền-đồ-thị--bpr-loss-lightgcn)
  - [3.2 Edge Dropout Contrastive Learning (SGL)](#32-edge-dropout-contrastive-learning-sgl)
  - [3.3 Representation Noise Perturbation (SimGCL)](#33-representation-noise-perturbation-simgcl)
  - [3.4 Hình Học Biểu Diễn trên Hypersphere](#34-hình-học-biểu-diễn-trên-hypersphere)
  - [3.5 Bộ Chỉ Số Toàn Diện Beyond-Accuracy](#35-bộ-chỉ-số-toàn-diện-beyond-accuracy)
  - [3.6 So Sánh Độ Phức Tạp Thuật Toán](#36-so-sánh-độ-phức-tạp-thuật-toán)
- [4. Cấu Trúc Dự Án](#4-cấu-trúc-dự-án)
- [5. Hướng Dẫn Cài Đặt & Thực Thi](#5-hướng-dẫn-cài-đặt--thực-thi)
- [6. Tiện Ích Chuyển Đổi Dữ Liệu (Excel / CSV / SQLite)](#6-tiện-ích-chuyển-đổi-dữ-liệu-excel--csv--sqlite)
- [7. Dashboard Streamlit 5 Tab Tương Tác](#7-dashboard-streamlit-5-tab-tương-tác)
- [8. Tài Liệu Tham Khảo (References)](#8-tài-liệu-tham-khảo-references)

---

## 1. Câu Hỏi Nghiên Cứu (Research Questions)

- **RQ1 (Ranking Accuracy)**: LightGCN, SGL và SimGCL khác nhau thế nào về chất lượng gợi ý Top-K (`Recall@10`, `NDCG@10`, `MRR@10`) trên tập dữ liệu đầy đủ và khi gặp dữ liệu thưa?
- **RQ2 (Representation Geometry)**: Cơ chế Contrastive Learning giải quyết hiện tượng sụp đổ chiều (Dimensional Collapse) và phân bố đều (Uniformity vs Alignment trên Hypersphere theo Wang & Isola, ICML 2020) ra sao?
- **RQ3 (Beyond-Accuracy Metrics)**: Sự khác biệt giữa 3 mô hình về **Intra-List Diversity (ILD)**, **Novelty (Self-Information)**, **Catalog Coverage** và **Hệ số Bất bình đẳng Gini**?
- **RQ4 (Degree-Stratified Robustness)**: Khi mức độ thưa tăng (từ 100% xuống 75%, 50%, 25%), mô hình nào duy trì hiệu năng tốt nhất trên nhóm người dùng **Tail (Cold-Start)** so với **Head (Active)**?
- **RQ5 (Spectral SVD & Over-smoothing)**: Phổ trị riêng ma trận embedding $\sigma_k$ và Effective Rank của các mô hình thể hiện khả năng kháng over-smoothing khi tăng số layer ra sao?
- **RQ6 (Statistical Rigor)**: Sự vượt trội của SimGCL/SGL so với LightGCN có ý nghĩa thống kê thực sự hay không (thông qua Paired t-test và Wilcoxon Signed-Rank Test với $p < 0.05, 0.01, 0.001$)?

---

## 2. Tập Dữ Liệu & Link Tải Dữ Liệu

Dự án sử dụng bộ dữ liệu chuẩn mực **Amazon Reviews (Electronics 5-core)** do nhóm nghiên cứu UCSD / Stanford SNAP phát hành:

* **Trang chủ dự án gốc**: [UCSD Amazon Product Data](https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_data) (GS. Julian McAuley)
* **Link tải trực tiếp từ Stanford SNAP Server (Dạng file nén `.json.gz`)**:
  * 📥 **Tương tác Reviews (1.68M đánh giá)**: [reviews_Electronics_5.json.gz](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz) *(495 MB)*
  * 📥 **Metadata Sản phẩm (498K items & Brand/Category)**: [meta_Electronics.json.gz](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Electronics.json.gz) *(186 MB)*
* **Link tải trên Kaggle**:
  * 🔗 [Amazon Electronics Reviews (5-core) trên Kaggle](https://www.kaggle.com/datasets/omer2241/amazon-electronics-reviews-5-core)
  * 🔗 [Amazon Product Reviews Dataset trên Kaggle](https://www.kaggle.com/datasets/saurav9786/amazon-product-reviews)

### Thống Kê Dữ Liệu Sau Tiền Xử Lý:
- **Implicit Feedback Threshold**: Tương tác dương $R_{u,i} = 1$ khi $\text{rating} \ge 4.0$.
- **K-core Filter**: Giữ lại các user có tối thiểu 5 tương tác dương ($\text{min\_interactions} \ge 5$).
- **Phân chia Per-User Chronological Split**: 80% Train, 10% Validation, 10% Test (chống rò rỉ dữ liệu Anti-Data Leakage).

| Chỉ số đồ thị | Ký hiệu | Giá trị thực tế | Ý nghĩa thực nghiệm |
| :--- | :---: | :---: | :--- |
| **Số người dùng** | $N_u$ | **135,996** | Số lượng user hợp lệ trong hệ thống |
| **Số sản phẩm** | $N_i$ | **62,749** | Số lượng mặt hàng đồ điện tử |
| **Tổng tương tác dương** | $|\mathcal{E}|$ | **1,173,135** | Số cạnh kết nối trong đồ thị hai phía |
| **Mật độ đồ thị** | $\text{Density}$ | **0.000137** (0.0137%) | Mức độ thưa thớt cực lớn của đồ thị thực tế |
| **Tương tác TB / User** | $\bar{d}_u$ | **8.63** (Trung vị: 6.0) | Phân phối tương tác lệch đuôi dài (Long-tail) |

---

## 3. Cơ Sở Toán Học & Các Hàm Mục Tiêu

### 3.1 Lan Truyền Đồ Thị & BPR Loss (LightGCN)

Lan truyền biểu diễn tuyến tính qua ma trận kề chuẩn hóa đối xứng:

$$
E^{(k+1)} = \tilde{A} E^{(k)}, \quad \text{where} \quad \tilde{A} = D^{-\frac{1}{2}} A D^{-\frac{1}{2}}
$$

Tổng hợp biểu diễn cuối cùng qua $L$ layer:

$$
E = \frac{1}{L+1} \sum_{k=0}^L E^{(k)}
$$

Hàm mất mát BPR (Bayesian Personalized Ranking) tối ưu hóa khoảng cách tương tác dương - âm:

$$
\mathcal{L}_{\text{BPR}} = \sum_{(u,i,j) \in \mathcal{D}} -\ln \sigma(\hat{y}_{ui} - \hat{y}_{uj}) + \lambda \lVert \Theta_0 \rVert_2^2
$$

Trong đó $\hat{y}_{ui} = e_u^\top e_i$ là tích vô hướng dự đoán điểm sở thích giữa người dùng $u$ và sản phẩm $i$, và $\Theta_0 = [E_u^{(0)}, E_i^{(0)}]$ là ma trận embedding ban đầu.

---

### 3.2 Edge Dropout Contrastive Learning (SGL)

SGL tạo 2 đồ thị con $G_1, G_2$ thông qua cơ chế loại bỏ cạnh ngẫu nhiên (Edge Dropout) với xác suất $p_{\text{drop}} = 0.1$:

$$
z_u^{(1)} = \text{LightGCN}(G_1, u), \quad z_u^{(2)} = \text{LightGCN}(G_2, u)
$$

Hàm mất mát tự giám sát InfoNCE tối đa hóa sự tương đồng giữa 2 view của cùng một nút:

$$
\mathcal{L}_{\text{SSL}} = -\sum_{u \in \mathcal{B}} \log \frac{\exp(\text{sim}(z_u^{(1)}, z_u^{(2)}) / \tau)}{\sum_{v \in \mathcal{B}} \exp(\text{sim}(z_u^{(1)}, z_v^{(2)}) / \tau)}
$$

Hàm mục tiêu tổng hợp của SGL:

$$
\mathcal{L}_{\text{SGL}} = \mathcal{L}_{\text{BPR}} + \lambda_{\text{ssl}} \mathcal{L}_{\text{SSL}} + \lambda \lVert \Theta_0 \rVert_2^2
$$

---

### 3.3 Representation Noise Perturbation (SimGCL)

SimGCL loại bỏ hoàn toàn việc sinh đồ thị con tốn kém, thay vào đó bơm trực tiếp nhiễu ngẫu nhiên đều $\Delta \sim U(0,1)$ đã được chuẩn hóa L2 vào biểu diễn ẩn tại mỗi layer:

$$
e^{(k)\prime} = e^{(k)} + \epsilon \cdot \frac{\Delta}{\lVert \Delta \rVert_2}, \quad \text{with} \quad \epsilon = 0.1
$$

Biểu diễn của cùng một nút qua 2 lần bơm nhiễu độc lập $e', e''$ tạo thành cặp dương:

$$
\mathcal{L}_{\text{CL}} = -\sum_{u \in \mathcal{B}} \log \frac{\exp(e_u^{\prime \top} e_u^{\prime\prime} / \tau)}{\sum_{v \in \mathcal{B}} \exp(e_u^{\prime \top} e_v^{\prime\prime} / \tau)}
$$

Hàm mục tiêu tổng thể của SimGCL:

$$
\mathcal{L}_{\text{SimGCL}} = \mathcal{L}_{\text{BPR}} + \lambda_{\text{cl}} \mathcal{L}_{\text{CL}} + \lambda \lVert \Theta_0 \rVert_2^2
$$

---

### 3.4 Hình Học Biểu Diễn trên Hypersphere

Theo lý thuyết của Wang & Isola (ICML 2020), chất lượng biểu diễn tương phản trên mặt cầu đơn vị $\mathcal{S}^{d-1}$ được đặc trưng bởi hai đặc tính đối ngẫu:

1. **Alignment Loss ($\mathcal{L}_{\text{align}}$)**: Đo khoảng cách Euclidean kỳ vọng giữa các cặp tương tác dương (càng nhỏ càng tốt):

$$
\mathcal{L}_{\text{align}} = \mathbb{E}_{(u,i) \sim p_{\text{pos}}} \left[ \lVert \bar{f}(u) - \bar{f}(i) \rVert_2^2 \right]
$$

2. **Uniformity Loss ($\mathcal{L}_{\text{uniform}}$)**: Đo mức độ phân bố đều của các vector biểu diễn trên toàn bộ mặt cầu nhằm tối đa hóa thông tin và chống sụp đổ chiều (càng âm/càng nhỏ càng tốt):

$$
\mathcal{L}_{\text{uniform}} = \log \mathbb{E}_{u, v \sim p_{\text{data}}} \left[ \exp\left(-2 \lVert \bar{f}(u) - \bar{f}(v) \rVert_2^2\right) \right]
$$

3. **Effective Rank & SVD Entropy**: Đánh giá độ phân rã phổ trị riêng $\sigma_k$ của ma trận embedding $E \in \mathbb{R}^{N \times d}$:

$$
\text{Effective Rank} = \exp\left( -\sum_{i=1}^d \bar{\sigma}_i \ln \bar{\sigma}_i \right), \quad \text{where} \quad \bar{\sigma}_i = \frac{\sigma_i}{\sum_{j=1}^d \sigma_j}
$$

---

### 3.5 Bộ Chỉ Số Toàn Diện Beyond-Accuracy

1. **Intra-List Diversity (ILD)**: Khoảng cách Cosine trung bình giữa các cặp sản phẩm trong danh sách Top-K:

$$
\text{ILD}@K = \frac{1}{|U|} \sum_{u \in U} \frac{2}{K(K-1)} \sum_{i < j \in R_u} (1 - \cos(e_i, e_j))
$$

2. **Novelty (Self-Information)**: Đo khả năng gợi ý các sản phẩm ít phổ biến (tránh thiên kiến phổ biến Popularity Bias):

$$
\text{Novelty}@K = \frac{1}{|U| \cdot K} \sum_{u \in U} \sum_{i \in R_u} -\log_2 \left( \frac{\text{count}(i) + 1}{|U_{\text{train}}|} \right)
$$

3. **Catalog Coverage & Gini Index**: Đánh giá tỷ lệ bao phủ toàn bộ danh mục sản phẩm và mức độ công bằng phân phối lượt hiển thị:

$$
\text{Coverage}@K = \frac{|\bigcup_{u \in U} R_u|}{|I|}, \quad \text{Gini}@K = \frac{\sum_{i=1}^{|I|} (2i - |I| - 1) \cdot c_{(i)}}{|I| \sum_{i=1}^{|I|} c_{(i)}}
$$

---

### 3.6 So Sánh Độ Phức Tạp Thuật Toán

| Thuật toán | Xây dựng đồ thị | Lan truyền Forward | Tính Contrastive Loss | Chi phí Bộ nhớ Đồ thị |
| :--- | :---: | :---: | :---: | :---: |
| **LightGCN** | $O(E)$ | $O(L \cdot E \cdot d)$ | Không ($0$) | $1 \times \text{Adj}$ (Thấp) |
| **SGL** | $O(3 \cdot E)$ | $O(3L \cdot E \cdot d)$ | $O(B^2 \cdot d)$ | $3 \times \text{Adj}$ (Rất cao) |
| **SimGCL** | $O(E)$ | $O(3L \cdot E \cdot d)$ | $O(B^2 \cdot d)$ | $1 \times \text{Adj}$ (Thấp) |

*Ghi chú*: $E = |\mathcal{E}|$ là số cạnh đồ thị, $L$ là số layer GNN, $d$ là số chiều embedding ($d=64$), $B$ là kích thước mini-batch ($B=2048$).

---

## 4. Cấu Trúc Dự Án

```text
advanced-machine/
├── README.md                 # Tài liệu nghiên cứu khoa học tổng hợp
├── WORK_LOG.md               # Nhật ký chi tiết tiến độ dự án (Append-only)
├── pyproject.toml            # Cấu hình dự án & Pytest
├── requirements.txt          # Danh sách thư viện Python
├── configs/                  # File cấu hình YAML
│   ├── common.yaml           # Cấu hình dữ liệu, siêu tham số chung
│   ├── lightgcn.yaml         # Cấu hình riêng LightGCN
│   ├── sgl.yaml              # Cấu hình SGL (ssl_weight, drop_ratio)
│   └── simgcl.yaml           # Cấu hình SimGCL (contrastive_weight, epsilon)
├── data/
│   ├── raw/                  # Dữ liệu nén Amazon Electronics (.json.gz)
│   ├── processed/            # Parquet splits & ma trận ánh xạ (.pkl)
│   └── exported/             # Dữ liệu xuất sang .csv, .xlsx, .sqlite
├── src/
│   ├── data/                 # Loader, Preprocessing, Graph, Sparsity, Splitter
│   ├── models/               # LightGCN, SGL, SimGCL modules
│   ├── losses/               # BPR Loss, InfoNCE Loss
│   ├── training/             # Trainer loop, Early Stopping, Multi-metric tracking
│   ├── evaluation/           # Metrics, Representation, Subgroup, Significance
│   │   ├── metrics.py        # Recall, NDCG, MRR, ILD Diversity, Novelty, Coverage, Gini
│   │   ├── representation.py # Alignment, Uniformity, SVD Spectrum, Over-smoothing
│   │   ├── subgroup.py       # Phân tầng bậc nút (Head, Torso, Tail)
│   │   ├── significance.py   # Paired t-test, Wilcoxon, LaTeX Table Generator
│   │   └── evaluator.py      # Full-ranking Top-K Evaluator
│   └── utils/                # Seed, Device, Logger, Config loader
├── scripts/
│   ├── prepare_data.py       # Tải & tiền xử lý dữ liệu SNAP Amazon (kèm tqdm)
│   ├── train.py              # CLI huấn luyện mô hình đơn lẻ
│   ├── benchmark_all.py      # Tự động hóa suite 36 runs + Thống kê + LaTeX
│   ├── generate_plots.py     # Sinh đồ thị nghiên cứu, Radar, Alignment/Uniformity
│   └── export_data.py        # Xuất dữ liệu sang Excel, CSV, SQLite, RecSys TXT
├── app/
│   └── streamlit_app.py      # Dashboard nghiên cứu tương tác 5 tab
├── tests/                    # 19 Unit tests pytest (100% PASS)
└── results/                  # Thư mục lưu trữ toàn bộ đầu ra (Output Hub)
    ├── checkpoints/          # File trọng số mô hình (.pt) & Global Best Models
    ├── history/              # File CSV tiến trình từng Epoch (Loss, Val NDCG)
    ├── aggregated/           # File CSV tổng hợp từng Model & Bảng LaTeX
    ├── raw/                  # File JSON chi tiết từng lần chạy
    └── figures/              # Đồ thị khoa học & Biểu đồ đường cong học tập
```


---

## 5. Hướng Dẫn Cài Đặt & Thực Thi

### 1. Cài Đặt Môi Trường
```bash
git clone https://github.com/.../advanced-machine.git
cd advanced-machine
pip install -r requirements.txt
```

### 2. Tiền Xử Lý Dữ Liệu (Amazon Electronics 5-core)
```bash
python scripts/prepare_data.py
```
*(Script có thanh tiến trình `tqdm` tải dữ liệu, giải nén và lọc 1.17 triệu tương tác)*.

### 3. Chạy Kiểm Thử Toàn Bộ (Pytest)
```bash
pytest -v
```
*(Chạy 19 test cases kiểm tra toán học, loss function, độ đa dạng, kiểm định thống kê và phân tầng)*.

### 4. Huấn Luyện Mô Hình Đơn Lẻ (50 Epochs)
```bash
# Huấn luyện LightGCN với 100% dữ liệu (50 epochs)
python scripts/train.py --model lightgcn --sparsity 1.0 --seed 42 --epochs 50

# Huấn luyện SGL với 75% dữ liệu (50 epochs)
python scripts/train.py --model sgl --sparsity 0.75 --seed 42 --epochs 50

# Huấn luyện SimGCL với 50% dữ liệu (50 epochs)
python scripts/train.py --model simgcl --sparsity 0.50 --seed 42 --epochs 50
```

### 5. Huấn Luyện & So Sánh Cả 3 Mô Hình Cùng Lúc (Mặc Định 50 Epochs)
```bash
# 1. Chạy mặc định cả 3 mô hình ở mức 100% dữ liệu:
python scripts/train_all_models.py

# 2. Chạy cả 3 mô hình ở một mức thưa cụ thể (ví dụ 75% hoặc 25% data):
python scripts/train_all_models.py --sparsity 0.75
python scripts/train_all_models.py --sparsity 0.25

# 3. Chạy 1 lệnh quét qua nhiều mức thưa:
python scripts/train_all_models.py --sparsity 1.0 0.50

# 4. Chạy toàn bộ 4 mức độ thưa (100%, 75%, 50%, 25%):
python scripts/train_all_models.py --all_sparsity

# 5. Tiếp tục huấn luyện nếu bị gián đoạn:
python scripts/train_all_models.py --resume
```
*(Script sẽ tự động in bảng so sánh đa chỉ số trên terminal, lưu bảng `results/aggregated/models_all_sparsity_seed42.csv` và tự động vẽ biểu đồ nghiên cứu)*.

### 6. Chạy Tự Động Toàn Diện Benchmark Suite (36 Runs Matrix - 50 Epochs/Run)
```bash
# Chế độ kiểm tra nhanh (Quick mode: 1 seed, 5 epochs, 100% data)
python scripts/benchmark_all.py --quick

# Chế độ Full Benchmark Nghiên Cứu (36 runs: 3 models x 4 sparsity x 3 seeds x 50 epochs)
python scripts/benchmark_all.py
```
*(Tự động tính Paired t-test, Wilcoxon test $p$-values và sinh mã bảng LaTeX tại `results/aggregated/benchmark_table.tex`)*.

### 6. Sinh Đồ Thị Nghiên Cứu Khoa Học
```bash
python scripts/generate_plots.py
```
*(Sinh biểu đồ Radar 6 chiều, 2D Pareto Alignment-Uniformity, Subgroup Tail vs Head, và Sparsity Curves)*.

### 7. Khởi Chạy Dashboard Tương Tác Streamlit
```bash
streamlit run app/streamlit_app.py
```

---

## 6. Hướng Dẫn Xem & Xuất Dữ Liệu (Excel / CSV / SQLite / Terminal)

Dự án cung cấp bộ công cụ toàn diện để bạn xem nhanh dữ liệu trên Terminal hoặc xuất sang các định dạng bảng tính phổ biến:

### 6.1 Xuất Dữ Liệu Sang File (Excel, CSV, SQLite, RecSys .inter)
Sử dụng script [`scripts/export_data.py`](file:///d:/advanced-machine/scripts/export_data.py) (có cờ `--with_meta` tự động ghép Tên sản phẩm, Thương hiệu và Ngành hàng):

```bash
# 1. Xuất sang file .CSV (mở trực tiếp bằng Microsoft Excel / Google Sheets):
python scripts/export_data.py --format csv --with_meta
# -> File được tạo tại: data/exported/train.csv, val.csv, test.csv

# 2. Xuất workbook Excel .XLSX (gồm 3 sheet Train, Val, Test có định dạng):
python scripts/export_data.py --format excel --with_meta --sample_size 50000
# -> File được tạo tại: data/exported/amazon_electronics_dataset.xlsx

# 3. Xuất cơ sở dữ liệu SQLite (.sqlite) để truy vấn SQL:
python scripts/export_data.py --format sqlite --with_meta
# -> File được tạo tại: data/exported/amazon_electronics.sqlite

# 4. Xuất chuẩn RecSys (.inter) tương thích RecBole / DaisyRec:
python scripts/export_data.py --format recsys_txt

# 5. Xuất tất cả định dạng cùng một lúc:
python scripts/export_data.py --format all --with_meta
```

### 6.2 Lệnh Xem Nhanh Dữ Liệu Trực Tiếp Trên Terminal
Bạn có thể chạy các lệnh một dòng sau để kiểm tra cấu trúc dữ liệu mà không cần mở file:

```bash
# Xem 5 dòng đầu tập Train (u_idx, i_idx, timestamp):
python -c "import pandas as pd; print(pd.read_parquet('data/processed/train.parquet').head(5))"

# Xem thống kê tổng quan (Số user, số item, số tương tác, mật độ đồ thị):
python -c "import pickle; print(pickle.load(open('data/processed/mappings.pkl', 'rb'))['stats'])"

# Xem thông tin chi tiết của 2 sản phẩm mẫu (Tên, Hãng, Danh mục):
python -c "import pickle; m=pickle.load(open('data/processed/mappings.pkl', 'rb')); print(list(m['item_metadata'].items())[:2])"
```

### 6.3 Duyệt Dữ Liệu Trực Quan Trên Web (Streamlit UI)
Khởi chạy giao diện web để duyệt thông tin sản phẩm và lịch sử mua sắm bằng mắt:
```bash
streamlit run app/streamlit_app.py
```
*(Mở trình duyệt vào Tab 1: **🎯 Interactive Recommendation** để chọn User ID và xem toàn bộ lịch sử mua sắm kèm sản phẩm gợi ý)*.

---

## 7. Dashboard Streamlit 5 Tab Tương Tác

1. **🎯 Interactive Recommendation**: Chọn User ID bất kỳ, xem lịch sử sản phẩm đã mua và so sánh danh sách Top-10 gợi ý song song giữa LightGCN, SGL và SimGCL cùng thời gian phản hồi latency (ms), điểm Diversity (ILD) và điểm Novelty (Self-info) tính theo thời gian thực.
2. **📊 Benchmark & Statistical Significance**: Bảng tổng hợp chi tiết tất cả các chỉ số (Recall, NDCG, MRR, Diversity, Novelty, Coverage, Gini), ma trận $p$-values kiểm định thống kê và bộ trích xuất mã LaTeX chuẩn IEEE/ACM.
3. **🌐 Representation Geometry & SVD Phổ**: Khám phá 2D Pareto Frontier giữa Alignment và Uniformity trên Hypersphere, đồ thị Radar 6 chiều và phân tích Rank ma trận.
4. **📉 Sparsity & Long-tail Subgroups**: Đồ thị trực quan sự suy giảm hiệu năng khi độ thưa tăng từ 100% đến 25% và so sánh nhóm người dùng ít tương tác (Tail) vs nhiều tương tác (Head).
5. **📘 Theoretical Foundations & Complexity**: Hệ thống hóa toàn bộ công thức toán học, cơ chế bơm nhiễu, phân tích độ phức tạp thời gian $O(\cdot)$ và chi phí bộ nhớ.

---

## 8. Tài Liệu Tham Khảo (References)

```bibtex
@inproceedings{he2020lightgcn,
  title={LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation},
  author={He, Xiangnan and Deng, Kuan and Wang, Xiang and Li, Yan and Zhang, Yongdong and Wang, Meng},
  booktitle={Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={639--648},
  year={2020}
}

@inproceedings{wu2021sgl,
  title={Self-supervised Graph Learning for Recommendation},
  author={Wu, Jiancan and Wang, Xiang and Feng, Fuli and He, Xiangnan and Chen, Liang and Lian, Jianxun and Xie, Xing},
  booktitle={Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={726--735},
  year={2021}
}

@inproceedings{yu2022simgcl,
  title={Are Graph Augmentations Necessary? Simple Graph Contrastive Learning for Recommendation},
  author={Yu, Junliang and Yin, Hongzhi and Xia, Xin and Chen, Tong and Li, Lizhen and Huang, Zi},
  booktitle={Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={1294--1303},
  year={2022}
}

@inproceedings{wang2020understanding,
  title={Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere},
  author={Wang, Tongzhou and Isola, Phillip},
  booktitle={International Conference on Machine Learning},
  pages={9929--9939},
  year={2020}
}

@inproceedings{mcauley2015image,
  title={Image-based recommendations on styles and substitutes},
  author={McAuley, Julian and Targett, Christopher and Shi, Qinfeng and Van Den Hengel, Anton},
  booktitle={Proceedings of the 38th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={43--52},
  year={2015}
}
```
