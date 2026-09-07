# Sửa đánh giá và báo cáo kết quả quick-test

Đính chính DirectAU (2026-09-06):

- [x] Evaluator nhận `score_fn`; Trainer nối `model.get_user_rating_scores` cho validation và test. Phân tích nhóm dùng lại test evaluator nên cũng dùng cùng scorer. DirectAU dùng cosine khi chọn checkpoint/early stopping và báo cáo cuối; LightGCN, XSimGCL, AdaptiveGCL giữ dot product. Gọi evaluator riêng với embedding cần truyền `score_fn=model.get_user_rating_scores` để chọn scorer của model.
- [x] Regression tests tái hiện item norm làm đổi thứ hạng, kiểm tra batch/user ID, candidate/history masks, bất biến khi scale embedding dương, và một epoch DirectAU thực có lưu/đọc checkpoint. Toàn bộ suite đạt **181 PASS, 10 SKIP** trong AML; các skip cần Pydantic tùy chọn. Test mới được ngoại lệ khỏi `.gitignore` để đi cùng bản sửa.
- [x] Run mới ghi `evaluation_protocol=model_scoring_shared_minilm_diversity_v3` vào JSON và CSV từng model; dashboard nhận diện giao thức mới. Fingerprint tự đổi theo evaluator/trainer/script nên kết quả/checkpoint cũ không được coi là tương thích với lần chạy mới.
- Chưa chạy lại benchmark thật. Ví dụ tổng hợp cho NDCG@10 cosine = 1, dot product = 0 chỉ chứng minh lỗi đường chấm điểm có thể ảnh hưởng kết quả; chưa chứng minh đây là nguyên nhân chính của điểm thấp. Checkpoint cũ được chọn bằng validation dot product; đánh giá lại cùng checkpoint chỉ đo ảnh hưởng scorer, còn benchmark đúng giao thức cần chọn checkpoint lại bằng validation cosine.

Các sửa đổi đánh giá trước đó:

Phạm vi: sửa cách đo/trình bày, không đổi loss hoặc cấu hình AdaptiveGCL, không dùng test để chọn siêu tham số.

- [x] Một lượt chạy: standard deviation và p-value không có cơ sở được ghi N/A, kèm số cặp và trạng thái kiểm định. Kiểm định không xác định không bị đổi thành `p=1`.
- [x] Diversity dùng cache MiniLM chung, đã được xác minh qua loader cho cả bốn model; không dùng embedding riêng của model. Baseline chỉ dùng text lúc đánh giá, không đưa text vào training.
- [x] Mỗi danh sách chỉ tính cặp item có text usable; user không đủ hai item usable không tham gia trung bình ILD. `DiversityValidTextFraction@K` báo tỷ lệ item gợi ý có text; lấy mẫu tối đa 1.000 user bằng RNG riêng seed 42, không làm đổi RNG huấn luyện. Không có reference thì metric không khả dụng, không âm thầm đổi định nghĩa.
- [x] Đổi `Tail (Cold-Start)` thành `Tail (Low-Activity)` ở code, bảng đọc, UI và biểu đồ. Nhóm theo phân vị bậc user, không phải user/item mới; không cố định ngưỡng 5 tương tác.
- [x] Regression tests và biểu đồ với std không khả dụng: toàn bộ **171 PASS, 10 SKIP** trong AML (19,32 giây). Các skip cần Pydantic tùy chọn. Bộ test riêng cho thay đổi đánh giá đạt 14/14, bao gồm sinh biểu đồ vào thư mục tạm; đã xem ảnh nhóm Tail/Head để xác nhận nhãn. Ruff phần thống kê/test mới và kiểm tra cú pháp/tên các file tích hợp PASS; MyPy thống kê PASS; CLI benchmark `--help` PASS. UI đã sửa code nhưng chưa kiểm thử bằng trình duyệt.

Các CSV/JSON/checkpoint/ảnh quick-test đã có được giữ nguyên, không sửa số đo Diversity cũ thành số đo mới. Dashboard cảnh báo bảng cũ. Các kết quả chạy mới ghi `evaluation_protocol=shared_minilm_diversity_v2`; fingerprint đánh giá đổi nên không được trộn với kết quả cũ. Chưa chạy lại benchmark hoặc ablation Amazon trong lượt này; chưa có bằng chứng cải thiện NDCG.

Phần thống kê mới áp dụng cho bảng sinh lại từ code mới. Bôi đậm số trung bình lớn nhất trong LaTeX không có nghĩa đạt ý nghĩa thống kê. Các hình mới không vẽ thanh sai số cho lượt thiếu std. Metadata missing có thể làm tập cặp được đánh giá khác nhau giữa các danh sách; cần đọc cùng tỷ lệ text hợp lệ, không coi missing là độ đa dạng cao.
