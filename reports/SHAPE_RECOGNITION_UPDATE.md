# Nhận diện tay nghỉ và tay thao tác — 19/11/2026

Nhánh webcam hiện dùng mô hình giám sát học hình dạng từ ảnh: ảnh có nhãn → MediaPipe → 21 điểm → 210 khoảng cách cặp điểm chuẩn hóa theo lòng bàn tay → SVM. Mô hình tự học bốn nhãn; không thay SVM bằng quy tắc ngưỡng chạm ngón. Đặc trưng bỏ ảnh hưởng vị trí, kích thước, lật ngang và xoay trong mặt phẳng, không đảm bảo bất biến với mọi góc nghiêng hoặc che ngón. HOG được giữ làm dự phòng cho yêu cầu ảnh không gửi landmark.

## Dữ liệu và kết quả

- Đã mở lớn các ảnh lỗi còn lại, chuyển ba ảnh tay mở từ PINCH_THUMB về UNKNOWN và loại một ảnh chuyển tiếp mờ. Giữ bản gốc và nhật ký tại `data/calibration_review/2026-11-19_shape_audit/`. Việc chỉnh nhãn trước đó chưa chính xác; nhật ký chỉnh mới có hiệu lực thay thế đối với bốn ảnh này.
- Còn 862 ảnh, gồm 38 ảnh webcam hiệu chỉnh. Hiệu chỉnh chỉ nằm trong train, trọng số mẫu 4; không trộn sang validation/test.
- Mô hình hình dạng chọn theo validation: Shape SVM C=1.0. Test 116/116 đúng, nhưng chỉ xét ảnh có phát hiện được bàn tay; 78 ảnh không phát hiện tay được ghi riêng trong `shape_metrics.json`. Nhóm ảnh công khai theo tên tệp chưa xác minh danh tính nên chưa phải đánh giá độc lập theo người.
- Kiểm tra lại 38 ảnh hiệu chỉnh: 38 dự đoán đúng và đạt ngưỡng. Đây là ảnh train, không phải đánh giá tổng quát hóa.
- Phát lại dự đoán ảnh hiệu chỉnh qua FSM: DOT, DASH, CONFIRM, SPACE, DELETE; kết quả cuối là A. Lưu riêng tại `image_learning/calibration_replay.json`, không gọi đây là kiểm thử camera trực tiếp.
- 78 kiểm thử Python và kiểm tra web client đạt. API thật đã được kiểm tra chọn mô hình hình dạng cho đủ bốn tư thế ở chế độ không ghi lệnh.

## Sử dụng và đánh giá tiếp

Máy chủ cổng 8001 nạp mô hình mới. Nhấn Ctrl+F5 để tải app.js có gửi landmark, bật camera và mở khóa nhập. Dòng mô hình cần hiển thị “Nhận diện hình dạng học từ ảnh”; dòng dự đoán cần có “Shape SVM”. Tay mở được trả về IDLE; ảnh không đủ chắc chắn vẫn là UNKNOWN, không dùng để tự nhập lệnh. Lịch sử 18.000 khung tại `/api/diagnostics/history` có thể dùng đối chiếu phiên thử mới.

Vẫn cần một phiên webcam độc lập để xác nhận trên người dùng, góc tay và điều kiện ánh sáng thực tế. Nếu MediaPipe mất dấu bàn tay hoặc xác định sai các điểm thì mô hình mới cũng có thể không nhận đúng; điểm test ảnh tĩnh không loại trừ hạn chế này.
