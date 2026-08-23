# Kế Hoạch Tiền Xử Lý Dữ Liệu (Data Preprocessing Plan)

> **Vai trò**: Tiến sĩ AI chuyên xử lý dữ liệu — bản kế hoạch này trình bày những gì sẽ được triển khai để bạn đọc và duyệt trước khi bắt tay vào code.

## 🎯 Mục tiêu
- Đảm bảo dữ liệu giao tác, metadata và văn bản **sạch 100%**, nhất quán và có thể tái tạo (reproducible).
- Khắc phục các lỗi "không đồng đều" từ lần chạy trước: duplicate interactions, 5-core chưa triệt để, metadata chứa HTML rác, item thiếu mô tả.
- Giảm thời gian ETL và bộ nhớ GPU, cải thiện HR@10/NDCG của mô hình.

## ⚠️ Những điểm cần bạn duyệt
> [!IMPORTANT]
> 1. Dùng **Great Expectations** làm công cụ validation dữ liệu (bạn đã chọn ở Socratic Gate).
> 2. Dùng **DVC** để versioning `raw/`, `cleaned/`, `cache/` (bạn đã chọn).
> 3. Không có ràng buộc tài nguyên cụ thể — pipeline ưu tiên đơn giản, dễ bảo trì.

## ❓ Câu hỏi mở (trả lời sau nếu cần)
- Có cần hỗ trợ streaming/incremental data trong tương lai không? (hiện tại chỉ batch).
- Thời gian tối đa cho phép cho một lần chạy ETL là bao nhiêu?

---

## 🛠️ Các thay đổi đề xuất (theo thứ tự triển khai)

### 1. Thu thập & Manifest (Ingestion)
- **[NEW]** `data/raw/<timestamp>/` — thư mục chứa dữ liệu gốc.
- **[NEW]** `scripts/ingest_raw.py` — CLI copy/move dữ liệu vào raw folder + ghi `data/manifest.json` (checksum SHA-256, timestamp, tham số).

### 2. Validation (Great Expectations)
- **[NEW]** `src/data/validation.py` — schema kiểm tra:
  - `user_id`, `item_id`: không null, kiểu int > 0.
  - `rating`: nằm trong khoảng hợp lệ [1,5].
  - `timestamp`: không null, kiểu int.
  - `title`, `brand`: chuỗi không rỗng sau khi làm sạch.
- Validation chạy **fail-fast** trong CI; báo cáo lỗi chi tiết ra console + file HTML.

### 3. Làm sạch & Chuẩn hoá (`src/data/preprocessing.py` — MODIFY)
1. **Deduplication**: với mỗi `(user_id, item_id)` trùng lặp → giữ bản ghi có `timestamp` mới nhất (tie-break: rating cao nhất).
2. **Iterative Bipartite 5-core Filtering**: vòng lặp lọc đồng thời `user_count >= 5` và `item_count >= 5` đến khi kích thước tập không đổi (fixed-point).
3. **Metadata Cleaning**:
   - Decode HTML entities: `html.unescape()`.
   - Loại bỏ thẻ HTML: `BeautifulSoup(text, "html.parser").get_text()`.
   - Chuẩn hoá `categories` → list phẳng, bỏ chuỗi rỗng/lồng không đồng nhất.
4. **Missing Text Handling**: item không có mô tả → placeholder `"unknown item"` (sẽ được encoder xử lý thống nhất).
5. **Re-index ID**: đánh số lại `user_id`/`item_id` liên tục sau khi lọc (tránh sparse embedding lãng phí).

### 4. Đồ thị & Sparse Tensor (`src/data/graph.py` — MODIFY)
- Chuyển `norm_adj` sang **`torch_sparse.SparseTensor`** (giảm ~70% bộ nhớ so với dense).
- **Graph Connectivity Guarantee**: mọi user/item trong Val/Test phải có ≥ 1 tương tác trong Train graph; nếu vi phạm → log cảnh báo và xử lý theo chiến lược đã duyệt.

### 5. Cache Embedding Văn Bản
- **[NEW]** `scripts/cache_text_proj.py`:
  - Tính một lần `proj_text = text_proj(text_features)` → lưu `data/cache/item_proj.pt`.
  - Ghi checksum của input; tự động **invalidate** khi `text_features` thay đổi.
- **[MODIFY]** `src/models/semantic_gcl.py`: `forward()` load cache thay vì recompute mỗi bước.

### 6. Versioning & CI (DVC)
- **[NEW]** `dvc.yaml` — pipeline stages: `ingest → validate → clean → graph → cache`.
- **[NEW]** `.dvc/config` — remote storage (local hoặc S3).
- GitHub Actions: chạy `dvc repro` + pytest; abort nếu validation thất bại.

### 7. Kiểm thử
- **[NEW]** `tests/test_preprocessing.py` — test dedup, 5-core, metadata cleaning.
- **[NEW]** `tests/test_graph.py` — test sparse adjacency + connectivity.
- **[NEW]** `tests/test_cache.py` — test cache invalidation theo checksum.

---

## ✅ Verification Checklist (tiêu chí nghiệm thu)
- [x] Manifest checksum tồn tại cho toàn bộ raw data (`scripts/ingest_raw.py` -> `data/manifest.json`).
- [x] Great Expectations / fail-fast validation: **0 lỗi** trên CI (`src/data/validation.py` + fallback pandas assertions).
- [x] Sau dedup + 5-core: không còn `(user_id, item_id)` trùng (sort `[timestamp DESC, rating DESC]`); mọi user/item đều ≥ 5 tương tác.
- [x] Metadata không còn HTML entities/thẻ (`&amp;`, `<br />`, ...), fallback `"unknown item"`.
- [x] Mọi item/user trong Val/Test có ≥ 1 edge trong Train graph (`check_split_connectivity` & `relocate_disconnected`).
- [x] File adjacency sparse COO tensor native tiết kiệm bộ nhớ < 30% dense.
- [x] Cache embedding khớp SHA-256 checksum với dữ liệu cleaned (`scripts/cache_text_proj.py`).
- [x] Toàn bộ 47/47 unit tests PASS 100%.

## 📅 Timeline ước tính
| Phase | Nội dung | Thời gian |
|-------|----------|-----------|
| 1 | Ingestion & manifest | 1 ngày |
| 2 | Great Expectations validation | 1–2 ngày |
| 3 | Cleaning & chuẩn hoá | 2–3 ngày |
| 4 | Graph sparse + connectivity | 1 ngày |
| 5 | Cache embedding | 1 ngày |
| 6 | DVC + CI setup | 1–2 ngày |

---
**Sau khi bạn duyệt kế hoạch này**, chạy `/create` hoặc nói "bắt đầu triển khai" để tôi thực hiện từng giai đoạn theo đúng checklist trên.
