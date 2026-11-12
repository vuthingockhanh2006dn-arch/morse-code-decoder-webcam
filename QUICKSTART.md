# Giao diện nhận diện cử chỉ tay

## Cài đặt và chạy trên Windows

Dùng **Python 3.9–3.12** (khuyến nghị 3.11). MediaPipe 0.10.14 trong dự án sử dụng API `solutions.hands`; Python 3.14 không phù hợp với bộ phụ thuộc này.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src/app.py
```

Giao diện desktop dùng Tkinter có sẵn trong bản Python Windows thông thường. Không cần huấn luyện mô hình hoặc tải dataset để thử bộ nhận diện hình học. Chọn camera `0` (hoặc `1` nếu có webcam ngoài), sau đó bấm **Bật camera**. Webcam chỉ được xử lý trên máy; ứng dụng không lưu ảnh/video.

## Cách nhập

| Lệnh | Cử chỉ |
| --- | --- |
| Dot `.` | Ngón trỏ chạm ngón cái nhanh rồi thả, dưới 0,40 giây |
| Dash `-` | Ngón trỏ chạm ngón cái và giữ ít nhất 0,40 giây |
| Space | Ngón trỏ chạm ngón áp út một lần rồi thả; chờ cửa sổ chạm đúp kết thúc |
| Chốt chữ | Ngón trỏ chạm ngón áp út hai lần nhanh, cách nhau dưới 0,45 giây |
| Delete | Ngón trỏ chạm ngón út; xóa tín hiệu cuối, hoặc chữ cuối nếu chuỗi tín hiệu trống |

Mở tay giữa các lệnh để mở khóa chống lặp. Space chốt chữ đang nhập và thêm khoảng trắng. Ví dụ nhập `...`, chốt chữ → `S`. Giữ nguyên cử chỉ không lặp lệnh. Có thể dừng camera và dùng các nút để thử bộ giải mã, sao chép văn bản hoặc xóa toàn bộ.

## Thuật toán và điều chỉnh

`src/gesture_detector.py` đo khoảng cách 3D từ đầu ngón trỏ đến đầu ngón cái, áp út và út. Tọa độ được sửa tỷ lệ khung hình, rồi chia cho kích thước lòng bàn tay (lớn hơn giữa chiều ngang lòng bàn tay và đoạn cổ tay–gốc ngón giữa). Nhờ đó ngưỡng ít phụ thuộc khoảng cách camera, tay trái/phải và xoay tay trong mặt phẳng ảnh.

Ngưỡng chạm mặc định 0,25; ngưỡng thả cao hơn 0,10 để chống rung. Chọn cặp gần nhất; khi hai cặp gần như bằng nhau, trả về UNKNOWN để giảm nhầm áp út/út. FSM hiện có làm mượt nhãn, lọc chạm quá ngắn, đo thời gian dot/dash và chạm đúp, khóa chống lặp. Mất bàn tay hoặc đổi tay sẽ hủy cử chỉ dở dang để tránh nhập nhầm khi mất tracking.

Thanh tiến trình hiển thị thời gian giữ đến dash. Hai thanh điều chỉnh ngưỡng chạm và thời gian dash có hiệu lực khi bật lại camera. Tăng ngưỡng nếu khó bắt chạm, giảm nếu nhận chạm khi ngón còn cách nhau. Các con số THUMB/RING/PINKY bên dưới webcam là khoảng cách đã chuẩn hóa, không phải độ tin cậy dự đoán.

Đặt bàn tay đủ sáng, thấy đủ các đầu ngón; tránh để áp út và út che nhau. Chuẩn hóa hình học không giải quyết hoàn toàn che khuất hoặc sai số độ sâu của MediaPipe. Cần thử webcam thực tế với nhiều người để điều chỉnh và đo độ chính xác; các kiểm thử hình học không thay thế đánh giá này.

## Kiểm thử không cần webcam

```powershell
python -m unittest discover -s tests
```
