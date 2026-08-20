# Nghiên cứu Chuyên sâu So sánh LightGCN, SGL và SimGCL: Hình học Biểu diễn, Đa chiều Beyond-Accuracy và Khả năng Chống chịu Dữ liệu Thưa

Dự án nghiên cứu thực nghiệm và phân tích toán học chuyên sâu so sánh ba phương pháp Graph Neural Networks tiêu biểu trong bài toán Gợi ý Top-K (**Top-K Electronics Recommendation**) trên tập dữ liệu **Amazon Electronics (5-core)** dưới các mức độ thưa dữ liệu khác nhau ($100\%, 75\%, 50\%, 25\%$):
1. **LightGCN** (SIGIR '20): Graph Collaborative Filtering chuẩn hóa tối giản (loại bỏ phi tuyến, ma trận trọng số và tự khuyên).
2. **SGL** (SIGIR '21): Self-Supervised Graph Learning dựa trên **Edge Dropout** tạo 2 đồ thị con augmented ($G_1, G_2$).
3. **SimGCL** (SIGIR '22): Simple Graph Contrastive Learning dựa trên **Embedding Uniform Noise Perturbation** trực tiếp trên không gian biểu diễn mà không cần tái tạo cấu trúc đồ thị.

---

## 1. Câu hỏi Nghiên cứu Chuyên sâu (Research Questions - RQs)

- **RQ1 (Ranking Accuracy)**: LightGCN, SGL và SimGCL khác nhau thế nào về chất lượng gợi ý Top-K (`Recall@10`, `NDCG@10`, `MRR@10`) trên tập dữ liệu đầy đủ và dữ liệu thưa?
- **RQ2 (Representation Geometry)**: Cơ chế Contrastive Learning giải quyết hiện tượng sụp đổ chiều (Dimensional Collapse) và phân bố đều (Uniformity vs Alignment trên Hypersphere theo Wang & Isola, ICML 2020) ra sao?
- **RQ3 (Beyond-Accuracy Metrics)**: Sự khác biệt giữa 3 mô hình về **Intra-List Diversity (ILD)**, **Novelty (Self-Information)**, **Catalog Coverage** và **Hệ số Bất bình đẳng Gini**?
- **RQ4 (Degree-Stratified Robustness)**: Khi mức độ thưa tăng ($100\% \rightarrow 75\% \rightarrow 50\% \rightarrow 25\%$), mô hình nào duy trì hiệu năng tốt nhất trên nhóm người dùng **Tail (Cold-Start)** so với **Head (Active)**?
- **RQ5 (Spectral SVD & Over-smoothing)**: Phổ trị riêng ma trận embedding $\sigma_k$ và Effective Rank của các mô hình thể hiện khả năng kháng over-smoothing khi tăng số layer ra sao?
- **RQ6 (Statistical Rigor)**: Sự vượt trội của SimGCL/SGL so với LightGCN có ý nghĩa thống kê thực sự hay không (thông qua Paired t-test và Wilcoxon Signed-Rank Test với $p < 0.05, 0.01, 0.001$)?

---

## 2. Bản chất Toán học & Các Hàm Mục tiêu

### 2.1 Lan truyền Đồ thị & Loss BPR Cơ sở
- **Lan truyền Đồ thị Tuyến tính**:
  $$E^{(k+1)} = \tilde{A} E^{(k)}, \quad \text{với } \tilde{A} = D^{-\frac{1}{2}} A D^{-\frac{1}{2}}$$
- **Tổng hợp Biểu diễn**: $E = \frac{1}{L+1} \sum_{k=0}^L E^{(k)}$
- **Hàm mất mát BPR**:
  $$\mathcal{L}_{\text{BPR}} = \sum_{(u,i,j) \in \mathcal{D}} -\ln \sigma(\hat{y}_{ui} - \hat{y}_{uj}) + \lambda \|\Theta_0\|_2^2$$

### 2.2 SGL (Edge Dropout Graph Contrastive Learning)
- Tạo 2 đồ thị con $G_1, G_2$ với xác suất loại bỏ cạnh $p_{\text{drop}} = 0.1$:
  $$\mathcal{L}_{\text{SGL}} = \mathcal{L}_{\text{BPR}} + \lambda_{\text{ssl}} \mathcal{L}_{\text{InfoNCE}}(z^{(1)}, z^{(2)}) + \lambda \|\Theta_0\|_2^2$$

### 2.3 SimGCL (Embedding Noise Perturbation Contrastive Learning)
- Bơm nhiễu ngẫu nhiên đều $\Delta \sim U(0,1)$ được chuẩn hóa vào biểu diễn ẩn tại mỗi layer:
  $$e^{(k)\prime} = e^{(k)} + \epsilon \cdot \frac{\Delta}{\|\Delta\|_2}, \quad \epsilon = 0.1$$
  $$\mathcal{L}_{\text{SimGCL}} = \mathcal{L}_{\text{BPR}} + \lambda_{\text{cl}} \mathcal{L}_{\text{InfoNCE}}(e', e'') + \lambda \|\Theta_0\|_2^2$$

### 2.4 Hình học Biểu diễn (Representation Geometry Metrics)
- **Alignment Loss** ($\mathcal{L}_{\text{align}}$):
  $$\mathcal{L}_{\text{align}} = \mathbb{E}_{(u,i) \sim p_{\text{pos}}} \left[ \|\bar{f}(u) - \bar{f}(i)\|_2^2 \right]$$
- **Uniformity Loss** ($\mathcal{L}_{\text{uniform}}$):
  $$\mathcal{L}_{\text{uniform}} = \log \mathbb{E}_{u, v \sim p_{\text{data}}} \left[ e^{-2 \|\bar{f}(u) - \bar{f}(v)\|_2^2} \right]$$
- **Effective Rank / SVD Entropy**:
  $$\text{Effective Rank} = \exp\left( -\sum_{i=1}^d \bar{\sigma}_i \ln \bar{\sigma}_i \right), \quad \text{với } \bar{\sigma}_i = \frac{\sigma_i}{\sum \sigma_j}$$

### 2.5 Bộ Chỉ số Beyond-Accuracy
- **Intra-List Diversity (ILD)**: Khoảng cách Cosine trung bình giữa các item trong danh sách Top-K:
  $$\text{ILD}@K = \frac{1}{|U|} \sum_{u} \frac{2}{K(K-1)} \sum_{i < j} (1 - \cos(e_i, e_j))$$
- **Novelty (Self-Information)**: Đánh giá khả năng gợi ý các mặt hàng ít phổ biến:
  $$\text{Novelty}@K = \frac{1}{|U| \cdot K} \sum_{u} \sum_{i \in R_u} -\log_2 \left( \frac{\text{count}(i) + 1}{|U_{\text{train}}|} \right)$$
- **Catalog Coverage & Gini Index**: Đo lường tỷ lệ bao phủ danh mục và mức độ công bằng phân phối.

---

## 3. Tập dữ liệu & Link Tải Dữ liệu (Dataset Download Links)

Dự án sử dụng bộ dữ liệu chuẩn mực **Amazon Reviews (Electronics 5-core)** do nhóm nghiên cứu UCSD / Stanford SNAP phát hành:
- **Trang chủ dự án gốc**: [UCSD Amazon Product Data](https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_data) (GS. Julian McAuley)
- **Link tải trực tiếp từ Stanford SNAP Server**:
  - 📥 **Reviews (1.68M tương tác)**: [reviews_Electronics_5.json.gz](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz) *(495 MB)*
  - 📥 **Metadata sản phẩm (498K items)**: [meta_Electronics.json.gz](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Electronics.json.gz) *(186 MB)*
- **Link tải từ Kaggle**:
  - 🔗 [Amazon Electronics Reviews (5-core) trên Kaggle](https://www.kaggle.com/datasets/omer2241/amazon-electronics-reviews-5-core)
  - 🔗 [Amazon Product Reviews Dataset trên Kaggle](https://www.kaggle.com/datasets/saurav9786/amazon-product-reviews)
- **Quy tắc tiền xử lý**:
  - **Implicit Positive Feedback**: Lọc $R_{u,i} = 1$ khi `rating >= 4.0`.
  - **K-core Filter**: Giữ lại các người dùng có tối thiểu 5 tương tác dương (`min_user_interactions = 5`).
  - **Quy mô sau xử lý**: **135,996 users**, **62,749 items**, **1,173,135 interactions** (độ thưa $0.0137\%$).

---

## 4. Cấu trúc Dự án

```text
advanced-machine/
├── README.md                 # Tài liệu nghiên cứu khoa học chi tiết
├── pyproject.toml            # Thông tin gói và dependencies
├── requirements.txt          # Danh sách thư viện Python
├── configs/                  # File cấu hình YAML
│   ├── common.yaml           # Cấu hình dùng chung
│   ├── lightgcn.yaml         # Config LightGCN
│   ├── sgl.yaml              # Config SGL
│   └── simgcl.yaml           # Config SimGCL
├── data/
│   ├── raw/                  # Dữ liệu gốc Amazon Electronics
│   └── processed/            # Parquet splits & metadata
├── src/
│   ├── data/                 # Loader, Preprocessing, Splitter, Graph, Sparsity
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
│   ├── prepare_data.py       # Tải & tiền xử lý dữ liệu SNAP Amazon
│   ├── train.py              # CLI huấn luyện mô hình đơn lẻ
│   ├── benchmark_all.py      # Tự động hóa suite 36 runs + Thống kê + LaTeX
│   └── generate_plots.py     # Sinh đồ thị nghiên cứu, Radar, Alignment/Uniformity
├── app/
│   └── streamlit_app.py      # Dashboard nghiên cứu tương tác 5 tab
├── tests/                    # 19 Unit tests pytest
└── results/
    ├── aggregated/           # CSV tổng hợp, thống kê p-value, bảng LaTeX
    └── figures/              # Đồ thị khoa học độ phân giải cao
```

---

## 5. Hướng dẫn Thực thi

### 1. Cài đặt Môi trường
```bash
pip install -r requirements.txt
```

### 2. Tiền xử lý Dữ liệu
```bash
python scripts/prepare_data.py
```

### 3. Chạy Kiểm thử Toàn bộ (Pytest)
```bash
pytest -v
```

### 4. Chạy Benchmark Suite & Kiểm định Thống kê
```bash
# Chế độ kiểm tra nhanh (Quick mode)
python scripts/benchmark_all.py --quick

# Chế độ Full Benchmark (36 runs: 3 models x 4 sparsity x 3 seeds)
python scripts/benchmark_all.py
```

### 5. Sinh Toàn bộ Đồ thị Nghiên cứu
```bash
python scripts/generate_plots.py
```

### 6. Khởi chạy Research Suite Dashboard (Streamlit)
```bash
streamlit run app/streamlit_app.py
```

---

## 6. Dashboard Streamlit 5 Tab Tương tác

1. **🎯 Interactive Top-K Recommendation**: Chọn User ID, xem lịch sử và so sánh danh sách Top-10 cùng thời gian thực Latency, Diversity (ILD) và Novelty (Self-info).
2. **📊 Benchmark & Statistical Significance**: Bảng tổng hợp toàn diện, ma trận kiểm định $p$-value ($*, **, ***$), và nút trích xuất bảng LaTeX chuẩn IEEE/ACM.
3. **🌐 Representation Geometry & SVD Phổ**: Khám phá 2D Pareto Frontier giữa Alignment và Uniformity, đồ thị Radar 6 chiều, và phân tích Rank ma trận.
4. **📉 Sparsity & Long-tail Subgroup Analysis**: Đồ thị hiệu năng suy giảm theo độ thưa và phân tích so sánh nhóm người dùng ít tương tác (Tail) vs nhiều tương tác (Head).
5. **📘 Theoretical Foundations & Complexity**: Chi tiết công thức toán học, cơ chế bơm nhiễu, phân tích độ phức tạp thời gian $O(\cdot)$ và bộ nhớ.
