# Đề cương báo cáo

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
