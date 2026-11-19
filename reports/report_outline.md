# Đề cương báo cáo

**Luồng học từ ảnh mới:** dùng [IMAGE_LEARNING_GUIDE.md](IMAGE_LEARNING_GUIDE.md) để trình bày đối chiếu tư thế lớp ảnh nguồn 9/7/6/5, đồng bộ 824 ảnh, HOG 1764 chiều, SVM/KNN/Random Forest và FSM. Số liệu thực tế ở `reports/image_learning/metrics.json`; nhóm tên file của ảnh nguồn chưa phải người/phiên thật, không công bố điểm test là độ chính xác webcam/người mới. Phần 66 landmark bên dưới là luồng tham khảo cũ.

Số liệu và hình trong báo cáo lấy từ file do các script sinh ra. Đừng gõ tay.

## Chương 1. Tổng quan
Lý do chọn đề tài, mục tiêu, phạm vi (chế độ Morse), sản phẩm. Nêu rõ: nhận diện vài cử chỉ ngón tay không phải dịch ngôn ngữ ký hiệu.

## Chương 2. Cơ sở lý thuyết
Mã Morse (`src/morse_decoder.py`), MediaPipe Hands và 21 landmark, khoảng cách Euclid, KNN, SVM, Random Forest, máy trạng thái hữu hạn.

## Chương 3. Dữ liệu
- Nguồn và kịch bản thu: `data/dataset_description.md`.
- Thống kê: `reports/dataset_statistics.json`, `reports/figures/*_distribution.png`.
- Làm sạch: `data/processed/cleaning_report.json`.
- Nhãn hai tầng (cử chỉ và tư thế), 66 đặc trưng, chia tập theo phiên.

## Chương 4. Thiết kế
Luồng: webcam, MediaPipe, 66 đặc trưng, mô hình phân loại tư thế, bỏ phiếu đa số, máy trạng thái, bộ đệm Morse, tra bảng. Giải thích cơ chế chống nhập lặp và vì sao một khung hình không đủ để phân biệt DOT với DASH.

## Chương 5. Thực nghiệm
- Chọn siêu tham số: `reports/validation_results.csv`.
- Kết quả test: `reports/test_results.md`, `reports/<algo>_classification_report.txt`.
- Ma trận nhầm lẫn: `reports/figures/<algo>_confusion_matrix.png`. Phân tích cặp nhãn hay nhầm, đối chiếu với dữ liệu thực tế của nhóm.
- Máy trạng thái: các test mô phỏng trong `tests/test_morse_fsm.py` (nhập "SOS", giữ tay không lặp).

## Chương 6. Kết luận
Kết quả, hạn chế (ánh sáng, góc nghiêng, chưa có chạy webcam trực tiếp), hướng mở rộng.

## Phụ lục
Danh sách nhãn, bảng Morse, hướng dẫn chạy (README).
