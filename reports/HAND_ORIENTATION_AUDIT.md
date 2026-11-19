# Kiểm tra chiều bàn tay và dữ liệu hiệu chỉnh — 19/11/2026

## Cập nhật xử lý phiên nhập và DOT

Lần đọc nhật ký tiếp theo ghi nhận phiên nhập bị khóa do `hand_not_detected`; các khung cuối không thấy tay. Bộ đệm chẩn đoán cũ chỉ có 60 khung, API chỉ trả 12 khung cuối nên không thể khôi phục đầy đủ lần thử DOT/DELETE đã kết thúc. Không dùng các khung này để kết luận mô hình phân loại sai ngón út.

Bản mới giữ quyền nhập khi mất dấu tay hoặc ảnh không chắc chắn, chỉ hủy cử chỉ đang dở. Trạng thái LOCKED sau một lệnh được giữ qua khoảng mất dấu, chỉ mở khi nhận tư thế nghỉ; không suy diễn thao tác thả từ ảnh thiếu tay. Tắt camera/reset/cancel chủ động vẫn khóa nhập. Ngưỡng DASH của web tăng thành 0,65 giây để dành thêm thời gian cho DOT. MediaPipe dùng detection 0,5 và tracking 0,35; ngưỡng chấp nhận mô hình ảnh không bị hạ. Lịch sử tăng lên 18.000 khung, đọc qua `/api/diagnostics/history`, có xác suất từng lớp và nguyên nhân hủy; chỉ lưu trong bộ nhớ phiên máy chủ, không lưu ảnh camera.

24 kiểm thử máy chủ và kiểm tra web client đạt sau thay đổi; cần xác nhận bằng phiên webcam mới, đặc biệt khả năng nhận PINCH_PINKY. Nội dung các lần kiểm tra cũ bên dưới được giữ làm lịch sử và có thể dùng ngưỡng cũ.

> Cập nhật sau khi sửa: nội dung kiểm tra ban đầu bên dưới mô tả phiên bản trước sửa. Đã sửa 6 nhãn, loại 4 ảnh chuyển tiếp, giữ bản gốc và nhật ký trong `data/calibration_review/2026-11-19/`. Còn 39 ảnh hiệu chỉnh, được cố định chỉ dùng train. Huấn luyện lại chọn SVM C=1: test 131/132 (99,24%), cùng ảnh lật ngang 130/132 (98,48%). Không so sánh trực tiếp với điểm cũ vì việc cố định nhóm hiệu chỉnh làm thay đổi cách chia nhóm công khai. Các điểm này không phải độ chính xác trên phiên webcam mới.
>
> Bộ xử lý web giảm cửa sổ làm mượt từ 5 xuống 3 khung, xác nhận thả từ 3 xuống 2 khung, mở rộng chạm đôi từ 0,45 lên 0,65 giây. Tư thế nghỉ IDLE đã được mô hình chấp nhận dùng trực tiếp để xác nhận thả, tránh kéo dài chạm ngắn do bộ lọc. UNKNOWN không đủ tin cậy vẫn không tự tạo lệnh. 68 kiểm thử Python và kiểm tra web client đã đạt; đã nạp bản mới tại cổng 8001. Cần thử camera thật để đánh giá độ nhạy thực tế.

## Kết quả mô hình đang chạy tại cổng 8001

- SVM C=10, ngưỡng chấp nhận 0,55; 867 ảnh, gồm 43 ảnh webcam P01/S01.
- Validation: 120/122 = 98,36%, cho cả chiều gốc và chiều lật ngang.
- Test: 132/132 = 100%, cho cả chiều gốc và chiều lật ngang. Đây là cùng tập ảnh được lật, không phải hai tập kiểm thử độc lập.
- P01/S01 nằm trong train. Dự đoán lại 43 ảnh này đúng nhãn lưu 43/43 ở cả hai chiều; kết quả này chỉ kiểm tra khả năng khớp dữ liệu huấn luyện, không xác nhận nhãn đúng hoặc độ chính xác webcam.
- Bộ ảnh công khai chia theo nhóm tên tệp, chưa có danh tính người chụp được xác minh. Không thể kết luận tổng quát hóa sang người mới từ điểm test này.

## Chiều ảnh

Video và canvas phủ điểm bàn tay được lật ngang bằng CSS để hiển thị như gương. Ảnh dùng thu dữ liệu và suy luận đều lấy từ video gốc, không dùng phép lật CSS. Nhánh phân loại ảnh dùng HOG từ điểm ảnh, không dùng nhãn Left/Right của MediaPipe để chọn ngón. Train có tăng cường lật ngang. Chưa tìm thấy bằng chứng lỗi đảo chiều trái/phải trong nhánh này; kiểm thử ảnh tĩnh không loại trừ mọi góc xoay hoặc che khuất khi dùng camera.

## Nhãn hiệu chỉnh có vấn đề

Đã kiểm tra trực quan bảng 43 ảnh: `reports/diagnostics/all_calibration.jpg`; ánh xạ số ảnh sang đường dẫn: `reports/diagnostics/calibration_index.txt`. Các số dưới đây là số trên bảng, không phải tên ảnh gốc.

- 22, 23, 25: tay mở nhưng lưu nhãn PINCH_RING/PINCH_THUMB.
- 20: khung hình chủ yếu là thân người/cánh tay, lưu nhãn PINCH_RING.
- 37, 39, 40: có tư thế chạm/cụp ngón cái–trỏ nhưng lưu nhãn UNKNOWN (tay nghỉ).
- 8, 15, 38: tư thế chuyển tiếp hoặc ảnh mờ, cần xem lại trước khi dùng huấn luyện.

Các nhãn mâu thuẫn có thể làm sai nhận diện lúc bắt đầu và thả cử chỉ. DOT cần chạm cái–trỏ ngắn rồi thả; DASH được phát khi giữ từ 0,4 giây. SPACE cần thả cái–áp út; CONFIRM cần hai lần chạm. Vì vậy nhận được DASH chưa chứng minh các bước thả hoặc chạm nhanh đã hoạt động.

## Giới hạn và bước tiếp theo

Máy chủ ghi nhận ba sự kiện DASH, bộ đệm `---`; các khung gần nhất không thấy bàn tay. Nhật ký hiện tại không đủ để xác định lỗi cụ thể của lần thử DOT/SPACE/DELETE. Chưa thay đổi hoặc xóa ảnh, chưa huấn luyện lại trong lần kiểm tra này. Cần sửa nhãn/loại ảnh chuyển tiếp khỏi tập train (giữ bản gốc), rồi đánh giá bằng một phiên webcam riêng không đưa vào train và ghi chuỗi chạm–thả để kiểm tra thời gian.
