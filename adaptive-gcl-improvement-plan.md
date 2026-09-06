# Cải tiến AdaptiveGCL có thể kiểm chứng

Phạm vi 06/09/2026: đồ án môn học; giữ backbone và mặc định hiện tại, không đổi dữ liệu/split/baseline. Mục tiêu là điều khiển và kiểm chứng từng thành phần, chưa tuyên bố tăng accuracy.

- [x] Thêm `use_item_text`, `user_semantic_weight`, `layer_aggregation`; `ssl_reg=0` bỏ hẳn tính SSL. Kiểm tra nhánh tắt không truyền gradient và cấu hình cũ giữ kết quả.
- [x] Thêm `mlp_weight_decay` riêng cho ma trận trọng số MLP, không phạt thêm ID/bias/trọng số tầng; mặc định 0. Kiểm tra nhóm optimizer và cập nhật thực tế.
- [x] Đồng bộ config/train/demo/pilot, từ chối cấu hình sai; chuẩn bị runner ablation validation-only lưu riêng từng cấu hình, không ghi đè benchmark.
- [x] Ghi đúng tổng hợp tầng dùng chung toàn đồ thị; zero-shot chưa được đánh giá cold-start. Ghi hướng dẫn đối chứng và giới hạn.
- [x] Chạy regression/unit/integration tests, CLI dry-run và lint. Không tự chạy sweep trên Amazon trong lượt này.

Hoàn tất triển khai không đồng nghĩa hoàn tất thực nghiệm: lựa chọn hệ số/regularization cần validation, xác nhận nhiều seed trước khi kết luận. Các artifact pilot trước được giữ nguyên; fingerprint mới sẽ yêu cầu checkpoint tương ứng code/config mới.

Xác minh: `python -m pytest -q` → 144 PASS (69,01 giây); Ruff toàn bộ runner/test mới và kiểm tra lỗi cú pháp/tên trên các file tích hợp PASS; MyPy model + runner PASS. CLI help/dry-run PASS, không train Amazon. Bandit còn cảnh báo deserialize pickle (chỉ dùng mappings cục bộ đáng tin cậy). UI chưa kiểm thử bằng trình duyệt.

Đọc [hướng dẫn ablation](docs/ADAPTIVE_GCL_ABLATION.md). Để xem trước 6 lượt đối chứng mặc định: `python scripts/ablate_adaptive.py`; chỉ thêm `--run` khi muốn bắt đầu train. Đã hoàn tất triển khai, chưa lựa chọn cấu hình thắng bằng validation.
