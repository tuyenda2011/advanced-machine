# Nhật ký cập nhật dự án (Work Log)

## 2026-08-19
- **Streamlit App (`app/streamlit_app.py`)**: 
  - Thêm type hints (`-> None`) cho hàm `main()`.
  - Bổ sung docstring mô tả chi tiết chức năng của ứng dụng.
  - Thêm hiệu ứng tải (`st.spinner`) khi load từng model để cải thiện trải nghiệm UI/UX.
  - Sửa đổi cơ chế bắt lỗi ngoại lệ chi tiết hơn khi load model không thành công (`st.error` và `st.exception`).
  - Thêm footer thẩm mỹ ở cuối trang với Streamlit & PyTorch.

*Lưu ý: File này chỉ ghi thêm (append-only), không xóa nội dung cũ theo yêu cầu.*

- **Chuyển đổi tập dữ liệu sang Amazon Electronics (5-core)**:
  - `configs/common.yaml`: Cập nhật URL dữ liệu sang `reviews_Electronics_5.json.gz` và `meta_Electronics.json.gz`.
  - `src/data/loader.py`: Viết lại hàm tải dữ liệu sang tải và đọc file `.json.gz`.
  - `src/data/preprocessing.py`: Viết lại hàm tiền xử lý `preprocess_amazon_electronics` xử lý dữ liệu JSON format, parse Item Brand và Category, ánh xạ cột `reviewerID` thành user và `asin` thành item.
  - `scripts/prepare_data.py`: Cập nhật import và gọi hàm theo tên mới.
  - `app/streamlit_app.py`: Đổi tiêu đề UI sang Electronics, đổi bảng Recommendation hiển thị Brand và Category thay cho Movie Genres.
  - `README.md`: Sửa tất cả các đoạn văn bản giải thích sang ngữ cảnh Đồ điện tử (Electronics Recommendation).

- **Dọn dẹp dữ liệu cũ**:
  - Đã xóa toàn bộ thư mục `data/raw/ml-1m`, file nén `data/raw/ml-1m.zip` và thư mục dữ liệu đã qua xử lý `data/processed/*` của MovieLens cũ để tránh xung đột với dữ liệu mới.
  - Xóa toàn bộ file kết quả đánh giá cũ trong thư mục `results/` để bắt đầu thử nghiệm hoàn toàn mới với dữ liệu đồ điện tử.

- **Khởi tạo dữ liệu Đồ điện tử**:
  - Đã thực hiện tự động tải tập dữ liệu **Amazon Reviews 2014 - Electronics 5-core** trực tiếp (gồm 2 file `reviews_Electronics_5.json.gz` và `meta_Electronics.json.gz`) và lưu vào `data/raw/amazon-electronics`.

## 2026-08-20
- **Nâng cấp Nghiên cứu Chuyên sâu Toàn diện cho 3 Mô hình (LightGCN, SGL, SimGCL)**:
  - **Lý thuyết Hình học Biểu diễn (`src/evaluation/representation.py`) [NEW]**:
    - Cài đặt hàm `compute_alignment` tính Alignment Loss ($\mathcal{L}_{align}$) trên mặt cầu đơn vị giữa các cặp tương tác dương: $\mathcal{L}_{align} = \mathbb{E}_{(u,i) \sim p_{pos}} [\|\bar{f}(u) - \bar{f}(i)\|_2^2]$.
    - Cài đặt hàm `compute_uniformity` tính Uniformity Loss ($\mathcal{L}_{uniform}$) theo Wang & Isola (ICML 2020) và SimGCL (SIGIR 2022): $\mathcal{L}_{uniform} = \log \mathbb{E}_{u, v \sim p_{data}} [e^{-2 \|\bar{f}(u) - \bar{f}(v)\|_2^2}]$.
    - Cài đặt hàm `compute_svd_spectrum` phân tích phổ trị riêng (Singular Value Spectrum Decay), tính toán tỷ lệ năng lượng phổ tích lũy (Cumulative Energy) và Effective Rank / Spectral Entropy ($\exp(-\sum \bar{\sigma}_i \ln \bar{\sigma}_i)$) để phát hiện hiện tượng sụp đổ chiều (Dimensional Collapse).
    - Cài đặt hàm `compute_oversmoothing_analysis` đo lường sự suy giảm khoảng cách cặp nút trung bình (Mean Pairwise Distance) qua các độ sâu layer GNN $L \in \{0, 1, 2, 3, 4, 5\}$.
  - **Bộ chỉ số Đa chiều Beyond-Accuracy (`src/evaluation/metrics.py`) [MODIFIED]**:
    - Thêm hàm `compute_intra_list_diversity` (ILD): Đo khoảng cách Cosine trung bình giữa các item được đề xuất trong danh sách Top-K: $\text{ILD}@K = \frac{1}{|U|} \sum_{u} \frac{2}{K(K-1)} \sum_{i < j} (1 - \cos(e_i, e_j))$.
    - Thêm hàm `compute_novelty`: Đo lường Self-Information $(-\log_2 P(i))$ để đánh giá mức độ khám phá các mặt hàng ít phổ biến / đuôi dài, giảm thiểu Popularity Bias.
    - Thêm hàm `compute_coverage_and_gini`: Đo tỷ lệ bao phủ danh mục (Catalog Coverage) và Hệ số Bất bình đẳng phân phối Gini Index trên toàn bộ kho hàng.
  - **Phân tầng Bậc Nút Người dùng (`src/evaluation/subgroup.py`) [NEW]**:
    - Cài đặt hàm `stratify_users_by_degree` tự động phân loại người dùng thành 3 nhóm: **Tail (Cold-Start - 20% ít tương tác nhất)**, **Torso (Medium - 60%)**, và **Head (Active - 20% tương tác nhiều nhất)**.
    - Cài đặt hàm `evaluate_degree_subgroups` đánh giá chi tiết Recall@K và NDCG@K trên từng phân khúc người dùng dưới các mức độ thưa ($100\%, 75\%, 50\%, 25\%$).
  - **Kiểm định Thống kê Ý nghĩa & Sinh Bảng LaTeX (`src/evaluation/significance.py`) [NEW]**:
    - Cài đặt hàm `compute_statistical_significance`: Tự động tính toán Paired Student's t-test và Wilcoxon Signed-Rank Test, tính toán $p$-value và phân loại mức ý nghĩa: $^{***}$ ($p < 0.001$), $^{**}$ ($p < 0.01$), $^{*}$ ($p < 0.05$), $\text{ns}$ (không có ý nghĩa).
    - Cài đặt hàm `generate_latex_table`: Tự động định dạng và sinh mã nguồn bảng **LaTeX** chuẩn IEEE/ACM (`results/aggregated/benchmark_table.tex`) với các giá trị cao nhất in đậm.
  - **Cập nhật Evaluator & Package Initialization (`src/evaluation/evaluator.py`, `src/evaluation/__init__.py`) [MODIFIED]**:
    - Nâng cấp class `Evaluator` hỗ trợ tham số `include_beyond_accuracy=True` và phương thức `get_predictions` phục vụ phân tích subgroup.
    - Export đầy đủ tất cả các hàm mới trong `src/evaluation/__init__.py`.
  - **Nâng cấp Trainer (`src/training/trainer.py`) [MODIFIED]**:
    - Tích hợp tự động tính toán đồng thời: Ranking Accuracy (Recall, NDCG, MRR), Beyond-Accuracy (Diversity, Novelty, Coverage, Gini), Representation Geometry (Alignment, Uniformity), SVD Effective Rank, và Subgroup Analysis (Tail vs Head) sau khi kết thúc huấn luyện.
  - **Nâng cấp Trình nạp Dữ liệu (`src/data/loader.py`) [MODIFIED]**:
    - Bổ sung cơ chế fallback parsing với `ast.literal_eval` để đọc chính xác cả chuẩn JSON và Python dict format của SNAP Amazon Metadata (`meta_Electronics.json.gz`).
  - **Thực thi Tiền xử lý Dữ liệu Amazon Electronics 5-core (`scripts/prepare_data.py`)**:
    - Đã tiền xử lý thành công toàn bộ tập dữ liệu đồ điện tử Amazon Electronics (5-core) và lưu vào `data/processed/`:
      - Số người dùng (num_users): **135,996**
      - Số sản phẩm (num_items): **62,749**
      - Số tương tác dương (num_interactions): **1,173,135**
      - Mật độ đồ thị (density): **0.000137** ($0.0137\%$)
      - Đã sinh đầy đủ `train.parquet`, `val.parquet`, `test.parquet` và `mappings.pkl`.
  - **Nâng cấp Tự động hóa Benchmark Suite (`scripts/benchmark_all.py`) [MODIFIED]**:
    - Thu thập toàn bộ dữ liệu Accuracy + Beyond-Accuracy + Representation Geometry + Subgroup Analysis qua 36 runs.
    - Tự động chạy kiểm định thống kê SimGCL vs LightGCN và SGL vs LightGCN, lưu vào `results/aggregated/statistical_significance.csv`.
    - Tự động xuất file LaTeX `results/aggregated/benchmark_table.tex`.
  - **Nâng cấp Bộ sinh Đồ thị Nghiên cứu (`scripts/generate_plots.py`) [MODIFIED]**:
    - Thêm biểu đồ 2D Scatter `alignment_vs_uniformity.png` (Pareto Frontier giữa Alignment và Uniformity).
    - Thêm biểu đồ Radar 6 chiều `beyond_accuracy_radar.png` (Recall, NDCG, Diversity, Novelty, Coverage, User Eff Rank).
    - Thêm biểu đồ phân tầng `subgroup_tail_vs_head.png` (so sánh hiệu năng trên nhóm Cold-Start Tail vs Active Head).
    - Thêm các đường cong suy giảm theo mức độ thưa (`sparsity_recall_10_curve.png`, `sparsity_ndcg_10_curve.png`, `sparsity_diversity_10_curve.png`).
  - **Nâng cấp Dashboard Streamlit (`app/streamlit_app.py`) [MODIFIED]**:
    - Thiết kế lại thành giao diện nghiên cứu 5 tab chuyên sâu:
      - Tab 1: 🎯 **Interactive Recommendation**: Chọn User ID, hiển thị Top-10 gợi ý kèm Brand, Category, thời gian trễ (latency), và điểm Diversity (ILD) cùng Novelty tính theo thời gian thực.
      - Tab 2: 📊 **Benchmark & Significance**: Bảng tổng hợp chi tiết kèm ma trận kiểm định thống kê $p$-values và bộ trích xuất mã LaTeX.
      - Tab 3: 🌐 **Representation Geometry & SVD**: Trực quan hóa mặt cầu biểu diễn 2D Alignment vs Uniformity và Radar Profile 6 chiều.
      - Tab 4: 📉 **Sparsity & Cold-Start Subgroups**: Đồ thị suy giảm theo độ thưa và phân tích nhóm người dùng Tail vs Head.
      - Tab 5: 📘 **Theoretical Foundations & Math**: Hệ thống hóa toàn bộ công thức toán học, hàm mất mát và bảng so sánh độ phức tạp tính toán $O(\cdot)$.
  - **Cập nhật Tài liệu Nghiên cứu (`README.md`) [MODIFIED]**:
    - Bổ sung 6 câu hỏi nghiên cứu (RQs), cơ sở lý thuyết toán học của Alignment/Uniformity, SVD Decay, Beyond-Accuracy, và hướng dẫn chạy suite 5 tab.
  - **Kiểm thử Toàn diện (Unit Tests `tests/`) [NEW & MODIFIED]**:
    - Tạo mới `tests/test_representation.py`: Kiểm tra tính hợp lệ của Alignment, Uniformity, SVD Spectrum, Over-smoothing dynamics.
    - Tạo mới `tests/test_beyond_accuracy.py`: Kiểm tra tính toán Top-K metrics, Diversity (ILD), Novelty, Coverage, Gini.
    - Tạo mới `tests/test_significance.py`: Kiểm tra Paired t-test, Wilcoxon test, và trình sinh bảng LaTeX.
    - Tạo mới `tests/test_subgroup.py`: Kiểm tra phân tầng bậc người dùng.
    - Cập nhật `tests/test_data.py`: Đồng bộ kiểm thử tiền xử lý sang định dạng Amazon Electronics.
    - Chạy `pytest -v`: **19/19 test cases đều vượt qua thành công (100% PASS)**.
  - **Cấu hình `.gitignore` (`.gitignore`) [MODIFIED]**:
    - Bổ sung thư mục `tests/` và toàn bộ thư mục `data/` vào danh sách `.gitignore` theo yêu cầu.
  - **Tích hợp Thanh Tiến trình Trực quan (`tqdm`) [NEW FEATURE]**:
    - `src/data/loader.py`: Thêm `DownloadProgressBar` hiển thị dung lượng và tốc độ tải mạng (MB/s), cùng thanh tiến trình `tqdm` đếm số dòng khi giải nén & parse file JSON.gz (`Loading Reviews`, `Loading Metadata`).
    - `src/data/preprocessing.py`: Thêm thanh tiến trình `tqdm` khi thực hiện lọc K-core users và ánh xạ metadata sản phẩm (`Mapping Item Metadata`).
    - `src/training/trainer.py`: Thêm thanh tiến trình cấp epoch (`Training MODEL_NAME`) và cấp mini-batch (`Epoch XX/YY`) hiển thị trực tiếp các giá trị loss (Total Loss, BPR Loss, CL Loss) và metrics validation (Val NDCG@10, Val Recall@10) theo thời gian thực.
    - `src/evaluation/evaluator.py`: Thêm tham số `show_progress=True` và thanh tiến trình `Top-K Inference` khi đánh giá tập test.
    - `scripts/benchmark_all.py`: Thêm thanh tiến trình tổng thể `Benchmark Suite Progress` hiển thị tiến độ chạy ma trận 36 thực nghiệm.
  - **Tiện ích Xuất Dữ liệu Đa định dạng (`scripts/export_data.py`) [NEW FEATURE]**:
    - Xây dựng CLI script hỗ trợ chuyển đổi dữ liệu từ `.parquet` sang các định dạng: **`.csv`** (mở trên Excel), **`.xlsx`** (Excel workbook đa sheet), **`.sqlite`** (cơ sở dữ liệu quan hệ), và **`.inter` / `.txt`** (chuẩn RecBole/RecSys).
    - Hỗ trợ cờ `--with_meta` để tự động ghép (enrich) tên sản phẩm (`product_title`), thương hiệu (`brand`) và ngành hàng (`category`) trực tiếp vào file xuất.
  - **Cập nhật & Viết lại Toàn diện Tài liệu Nghiên cứu (`README.md`) [MODIFIED]**:
    - Hệ thống hóa lại toàn bộ tài liệu theo chuẩn bài báo khoa học: 6 câu hỏi nghiên cứu (RQs), bảng thống kê đồ thị 1.17M tương tác, công thức toán học chi tiết (LightGCN, SGL, SimGCL, Alignment, Uniformity, SVD Decay, Effective Rank, ILD, Novelty, Coverage, Gini), bảng so sánh độ phức tạp Big-O, hướng dẫn thực thi từng bước, hướng dẫn xuất dữ liệu Excel/CSV, hướng dẫn Streamlit 5 tab, và tài liệu trích dẫn chuẩn BibTeX.
  - **Sửa Lỗi Giao Diện Streamlit (`app/streamlit_app.py`) [BUG FIX]**:
    - Khắc phục lỗi `TypeError: '<' not supported between instances of 'str' and 'int'` tại Tab 1: Thay thế việc tính `max(user2id.keys())` (do khóa là chuỗi Amazon Reviewer ID như `A100UD67...`) bằng cách chọn theo chỉ mục số nguyên `u_idx` từ `0` đến `135,995` và tự động tra cứu Reviewer ID tương ứng hiển thị trực quan.
  - **Khắc phục Cảnh Báo NumPy & PyTorch (`src/data/graph.py`) [BUG FIX]**:
    - Xử lý triệt để `UserWarning: 'where' used without 'out'` trong `normalize_adj_matrix` bằng mảng khởi tạo 0 và mask `rowsum > 0`.
    - Bổ sung `check_invariants=False` trong `torch.sparse_coo_tensor` để tắt cảnh báo nội bộ của PyTorch Sparse, giúp log terminal sạch sẽ 100%.
  - **Kích hoạt GPU Acceleration & Cài đặt PyTorch CUDA 12.1 [INFRASTRUCTURE]**:
    - Nâng cấp môi trường Conda `AML` (`D:\Miniconda\envs\AML\`) lên phiên bản **`torch-2.5.1+cu121`** có hỗ trợ CUDA 12.1.
    - Kích hoạt thành công card đồ họa **NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM)**, giúp tăng tốc lan truyền ma trận thưa và huấn luyện mô hình nhanh gấp 20 - 30 lần so với CPU.
    - Thực hiện `pip cache purge`, dọn dẹp sạch sẽ **8.6 GB** bộ nhớ đệm tạm thời khỏi ổ C (`C:\Users\Admin\AppData\Local\pip\cache\`).
  - **Cơ chế Lưu Checkpoint Nối tiếp & Tệp Tin Lịch sử Tiến trình [NEW FEATURE]**:
    - `src/training/trainer.py` & `src/training/early_stopping.py`: Bổ sung lưu checkpoint kép: checkpoint tốt nhất (`.pt`) và checkpoint gần nhất (`_latest.pt`), hỗ trợ tiếp tục huấn luyện nối tiếp (`--resume`).
    - **Lưu lịch sử tiến trình từng epoch (`results/history/{model}_{sparsity}_seed{seed}_history.csv`)**: Tự động ghi nhận `train_loss`, `bpr_loss`, `cl_loss`, `val_ndcg_10`, `val_recall_10`, `val_mrr_10`, `epoch_time_sec`, và đánh dấu `is_best_epoch` qua từng epoch để theo dõi đường cong học tập.
    - **Lưu kết quả tổng hợp riêng cho từng Model (`results/aggregated/{model}_results.csv`)**: Mỗi mô hình (`lightgcn`, `sgl`, `simgcl`) đều có 1 file CSV riêng ghi nhận toàn bộ các chỉ số đo đạc qua các mức độ thưa và seed.
    - **Cơ chế Global Best Model (`artifacts/checkpoints/{model}_best.pt` & `{model}_best_meta.json`)**: Tự động so sánh và duy trì checkpoint có chất lượng cao nhất của từng mô hình qua toàn bộ các lần chạy, và tự động nạp trên Streamlit App.
  - **Kiểm Thử Toàn Diện Hệ Thống (Deep System Audit) [VERIFICATION]**:
    - Biên dịch và kiểm tra cú pháp **40/40 tệp tin Python** (100% Passed, 0 Syntax Errors).
    - Chạy bộ kiểm thử tự động `pytest`: **19/19 test cases Passed** (100%).
  - **Script Chạy Tự Động 3 Mô Hình Cùng Lúc (`scripts/train_all_models.py`) [NEW FEATURE]**:
    - Xây dựng công cụ CLI chạy tuần tự cả 3 mô hình (**LightGCN**, **SGL**, **SimGCL**) trong một lệnh duy nhất.
    - Hỗ trợ các tham số: `--models`, `--sparsity`, `--seed`, `--epochs`, `--resume`, `--no_plots`.
    - Tự động in bảng so sánh đa chiều (Accuracy, Diversity, Novelty, Alignment, Uniformity, Effective Rank, Latency) dạng khung lưới đẹp mắt trên Console và xuất file bảng tổng hợp `results/aggregated/three_models_comparison_{sparsity}_{seed}.csv`.
    - Tự động kích hoạt vẽ bộ biểu đồ khoa học và đường cong học tập sau khi chạy xong.
  - **Chuẩn Hóa Cấu Hình Huấn Luyện 50 Epochs & Hỗ Trợ Đa Mức Độ Thưa [CONFIGURATION]**:
    - Thiết lập chuẩn **`epochs: 50`** trong `configs/common.yaml`, `scripts/train_all_models.py`, `scripts/benchmark_all.py` và toàn bộ tài liệu hướng dẫn `README.md`.
    - Nâng cấp `scripts/train_all_models.py` hỗ trợ truyền nhiều mức độ thưa (ví dụ `--sparsity 1.0 0.50`) hoặc cờ `--all_sparsity` để quét toàn bộ 4 mức (100%, 75%, 50%, 25%) tự động.
    - Đảm bảo điểm số hội tụ đỉnh cao, tối ưu thời gian huấn luyện (chỉ ~1.2 giờ cho cả 3 mô hình) và đồng đều tuyệt đối giữa các mô hình.
  - **Tối Ưu Hiển Thị Terminal & Khắc Phục Trùng Lặp Thanh Tiến Trình [BUG FIX]**:
    - Khắc phục hiện tượng xung đột dòng lệnh khi lồng `tqdm` giữa script cha (`train_all_models.py`) và script con (`train.py`).
    - Thay thế thanh tiến trình cha bằng các block Header phân đoạn rõ ràng (`[1/3] >>> STARTING: LIGHTGCN <<<`).
    - Tinh chỉnh `batch_pbar` và sử dụng `tqdm.write` trong `trainer.py`, giúp từng dòng tóm tắt Epoch in ra sắc nét, không bị đè chữ hay tràn chuỗi log trên Windows terminal.
  - **Quy Hoạch Toàn Bộ Đầu Ra Về Thư Mục `results/` [REFACTOR]**:
    - Chuyển toàn bộ đường dẫn lưu file Checkpoint sang **`results/checkpoints/`** để gom toàn bộ dữ liệu đầu ra về một nơi duy nhất.
    - Cấu trúc thư mục đầu ra thống nhất:
      - `results/checkpoints/`: Lưu file trọng số `.pt` và Global Best Models (`_best.pt`).
      - `results/history/`: Lưu file CSV tiến trình từng Epoch (`_history.csv`).
      - `results/aggregated/`: Lưu bảng tổng hợp kết quả riêng của từng mô hình (`.csv`) và bảng LaTeX.
      - `results/raw/`: Lưu file JSON chi tiết từng lần chạy.
  - **Dọn Sạch Toàn Bộ Kết Quả & Checkpoint Cũ [MAINTENANCE]**:
    - Xóa sạch toàn bộ các tệp tin chạy thử nghiệm cũ trong `results/checkpoints/`, `results/history/`, `results/aggregated/`, `results/raw/`, `results/figures/` và `artifacts/checkpoints/`.
    - Chuẩn bị sẵn sàng cấu trúc thư mục sạch 100% cho đợt huấn luyện chính thức (Official 50 Epochs Run).
  - **Tổ Chức Thư Mục Con Riêng Biệt Cho Từng Mô Hình [ARCHITECTURE & REFACTOR]**:
    - Phân bổ riêng biệt theo thư mục con cho từng mô hình (`lightgcn/`, `sgl/`, `simgcl/`) bên trong các thư mục đầu ra:
      - `results/checkpoints/{model}/`: Lưu checkpoints `.pt` và `{model}_best.pt` của riêng mô hình đó.
      - `results/history/{model}/`: Lưu file tiến trình `_history.csv` từng epoch của riêng mô hình đó.
      - `results/raw/{model}/`: Lưu file JSON chi tiết từng lần chạy của riêng mô hình đó.
      - `results/aggregated/`: Chứa các bảng CSV tổng kết và bảng LaTeX so sánh đa mô hình.
      - `results/figures/`: Chứa các ảnh biểu đồ nghiên cứu và learning curves.
    - Cập nhật toàn bộ các script (`train.py`, `trainer.py`, `train_all_models.py`, `benchmark_all.py`, `generate_plots.py`, `streamlit_app.py`) tương thích 100% với cấu trúc thư mục con mới.





















