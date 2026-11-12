# Mô tả bộ dữ liệu

Tài liệu này mô tả cách thu, gán nhãn, làm sạch và chia dữ liệu. Mục "Số liệu thực tế" ở cuối để trống cho đến khi bạn thu xong; chạy `python3 src/data_analysis.py` để lấy số.

## 1. Nguồn
- Tự thu bằng webcam laptop với `src/data_collection.py`. Không dùng bộ dữ liệu ngoài.
- Chỉ lưu 21 landmark (x, y, z) của MediaPipe. Không lưu ảnh hay video của người tham gia.
- Người tham gia đặt mã ẩn danh (`user_01`, `user_02`, ...) và đồng ý trước khi quay.

## 2. Nhãn

Mô hình chỉ nhìn một khung hình, nên chỉ phân biệt được tư thế. Một khung hình không cho biết người dùng đang chạm nhanh hay giữ, hay đây là lần chạm thứ mấy. Vì vậy nhãn chia hai tầng:

| Nhãn cử chỉ (người quay chọn) | Nhãn tư thế (cột `pose`, mô hình học) | Máy trạng thái quyết định |
|---|---|---|
| DOT | PINCH_THUMB | trỏ + cái, giữ dưới 0,40 s: DOT |
| DASH | PINCH_THUMB | trỏ + cái, giữ từ 0,40 s: DASH |
| CONFIRM | PINCH_RING | trỏ + áp út, chạm 2 lần trong 0,45 s: CONFIRM |
| SPACE | PINCH_RING | trỏ + áp út, chạm 1 lần: SPACE |
| DELETE | PINCH_PINKY | trỏ + út, chạm 1 lần: DELETE |
| UNKNOWN | UNKNOWN | tay nghỉ hoặc tư thế ngoài quy ước: không sinh sự kiện |

Cột `gesture` giữ nhãn cử chỉ gốc để thống kê. Cột `pose` là nhãn huấn luyện. Khi thu DOT và DASH, hãy giữ chụm ngón thay vì gõ nhịp, vì với mô hình hai nhãn này giống hệt nhau.

Đây là cử chỉ điều khiển do đề tài tự quy ước, không thuộc hệ ngôn ngữ ký hiệu nào.

## 3. Kịch bản thu

Mỗi phiên cố định một bộ điều kiện và thu đủ 6 nhãn. Thay đổi điều kiện giữa các phiên.

| Yếu tố | Giá trị | Tùy chọn trong lệnh |
|---|---|---|
| Người | tối thiểu 4 người | `--subject` |
| Phiên | 2–3 phiên mỗi người | `--session` |
| Ánh sáng | ban ngày, đèn phòng, tối | `daylight`, `room_light`, `dim_light` |
| Góc quay | chính diện, nghiêng 30° | `front_0deg`, `angle_30deg` |
| Khoảng cách | 35, 50, 70 cm | `near_35cm`, `mid_50cm`, `far_70cm` |
| Nền | tường trơn, nền lộn xộn | `plain_wall`, `cluttered` |

Gợi ý: khoảng 200 mẫu mỗi nhãn mỗi phiên, thu cả tay trái lẫn tay phải. Góc nghiêng rất quan trọng với cặp PINCH_RING và PINCH_PINKY vì hai ngón này nằm sát nhau.

Cột trong file CSV thô:
`subject, session, gesture, pose, handedness, lighting, angle, distance, background, timestamp, x0, y0, z0, ..., x20, y20, z20`

## 4. Loại mẫu lỗi

Lúc thu, khung hình bị bỏ (và đếm lại để báo cáo) khi:
- không thấy tay, hoặc độ tin cậy MediaPipe dưới 0,8;
- có landmark nằm ngoài khung hình quá 2%, tức tay bị cắt hoặc che;
- cổ tay dịch hơn 0,08 so với khung trước, tức tay mờ do chuyển động nhanh.

Lúc tiền xử lý, `preprocessing.py` bỏ thêm: dòng thiếu giá trị, nhãn lạ, dòng trùng, tay ngoài khung hình, tay lỗi (mọi landmark trùng nhau). Số dòng bị bỏ ở từng bước nằm trong `data/processed/cleaning_report.json`.

Hệ thống không đo độ mờ của ảnh vì ảnh không được lưu. Bộ lọc tốc độ cổ tay và độ tin cậy MediaPipe là cách thay thế.

## 5. Đặc trưng (66 giá trị)
- 63 giá trị: landmark dời gốc về cổ tay, lật gương nếu tay trái, chia cho khoảng cách xa nhất tới cổ tay.
- 3 giá trị: khoảng cách Euclid trỏ–cái, trỏ–áp út, trỏ–út, chia cho độ dài lòng bàn tay (cổ tay tới gốc ngón giữa).

## 6. Chia tập
70% train, 15% validation, 15% test, chia theo phiên quay (mặc định) hoặc theo người (`--split-by subject`). Mỗi phiên hay mỗi người nằm trọn trong một tập. Cần ít nhất 3 phiên (hoặc 3 người). Cờ `--allow-frame-split` chỉ dành cho chạy thử.

## 7. Số liệu thực tế (điền sau khi thu)

| Mục | Giá trị |
|---|---|
| Số người | |
| Số phiên | |
| Tổng số mẫu sau làm sạch | |
| Mẫu mỗi nhãn tư thế | |
| Tỷ lệ lệch max/min | |
| Phân bố ánh sáng / góc / khoảng cách / nền | |
| Các cặp tư thế hay nhầm (xem ma trận nhầm lẫn) | |
