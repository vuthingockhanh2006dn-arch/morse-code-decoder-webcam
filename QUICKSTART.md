# Chạy giao diện web Morse

## Cài đặt trên Windows

Dùng Python 3.9–3.12, khuyến nghị 3.11. Nếu đã tạo `.venv` và cài thư viện, chỉ cần chạy dòng cuối.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -X utf8 src/web_server.py
```

Giữ terminal hoạt động và mở **http://localhost:8000** trong trình duyệt. Không mở `web/index.html` bằng `file://`. Nếu cổng bận, thêm `--port 8001` rồi mở http://localhost:8001. Nhấn Ctrl+C để dừng máy chủ.

Ứng dụng dùng thư viện chuẩn `http.server`; không cần FastAPI hay Uvicorn. Giao diện desktop `app.py` và kiểm thử Tkinter đã được bỏ. Xem bố cục ở `web/index.html`, mã tương tác ở `web/app.js`.

## Cử chỉ

Ngón cái là ngón chạm chính, thống nhất với dữ liệu huấn luyện và giao diện:

| Lệnh | Thao tác |
| --- | --- |
| Dot `.` | Cái–trỏ chạm nhanh rồi thả |
| Dash `-` | Cái–trỏ chạm giữ ít nhất 0,40 giây |
| Space | Cái–áp út chạm một lần rồi thả, chờ hết cửa sổ chạm đúp |
| Chốt chữ | Cái–áp út chạm hai lần nhanh, trong cửa sổ 0,45 giây |
| Delete | Cái–út chạm: xóa tín hiệu cuối hoặc chữ cuối nếu chuỗi tín hiệu trống |

Mở tay trước khi bắt đầu và giữa các thao tác. Chốt chữ giữ lại chữ trong văn bản để nhập chữ tiếp theo; Space chốt chữ rồi thêm dấu cách. Ví dụ: dot, dash, chốt chữ → A; tiếp tục dash, dot, chốt chữ → AN.

Khi camera tắt, có thể nhấn giữ/thả các nút mô phỏng. Các khung nghỉ vẫn được gửi sau khi thả nút để xử lý dot và chạm đơn/chạm đúp. Không trộn camera với nút mô phỏng. Nút nghỉ thả cử chỉ; nút Xóa hết xóa cả mã và văn bản. Dừng camera hoặc mất tracking chỉ hủy cử chỉ đang thực hiện, giữ văn bản đã nhập.

## Một luồng xử lý

1. MediaPipe trong trình duyệt lấy 21 điểm bàn tay, gửi tọa độ và kích thước ảnh về `/api/predict`.
2. `gesture_detector.py` sửa tỷ lệ ảnh, chuẩn hóa theo lòng bàn tay, kiểm tra cặp ngón ổn định 70 ms và dùng ngưỡng chạm/thả riêng. Các cặp quá sát hoặc đổi ngón khi giữ bị loại.
3. Nếu chọn mô hình ML, máy chủ đối chiếu dự đoán với cặp ngón hình học; không chấp nhận hai kết quả trái nhau. Tự động chọn `best_model` trong cấu hình, hoặc mô hình sẵn có, hoặc hình học khi chưa huấn luyện. Mô hình chưa có sẽ bị vô hiệu hóa trong danh sách.
4. `fsm_controller.py` là nơi duy nhất đo thời gian, chống lặp và sinh lệnh. `morse_decoder.py` là nơi duy nhất lưu bảng Morse và văn bản. JavaScript chỉ nhận kết quả để hiển thị/phát âm báo.

Chỉnh ngưỡng trong `models/runtime_config.json` rồi khởi động lại máy chủ. Các khóa: `touch_threshold` (0.25), `release_threshold` (0.35), `ambiguity_margin` (0.08), `contact_stable_time` (0.07), `dash_time` (0.40), `double_tap_window` (0.45). Tệp cấu hình chưa tồn tại thì dùng mặc định trong `src/config.py`.

Máy chủ chỉ lắng nghe trên máy cục bộ. Hiện bộ đệm dùng chung cho các tab của cùng máy chủ: nên dùng một tab để tương tác. Văn bản giữ trong bộ nhớ đến khi xóa hoặc tắt máy chủ, chưa lưu xuống đĩa.

Không lưu ảnh/video. MediaPipe được tải từ CDN nên lần mở trang cần Internet; nhận diện trên trình duyệt và gửi tọa độ về Python cục bộ. Độ chính xác thực tế phụ thuộc ánh sáng, che khuất và góc tay; kiểm thử tọa độ không thay thế đánh giá webcam với người dùng.

## Kiểm thử

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests
node tests/test_web_client.js
```

`hand_tracker.py` vẫn cần cho `data_collection.py`. Các tệp dữ liệu, chia tập, huấn luyện và đánh giá vẫn cần cho đề tài; không phải phần dư của giao diện.
