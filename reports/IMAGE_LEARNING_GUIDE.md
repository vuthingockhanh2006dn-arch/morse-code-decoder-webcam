# Học từ ảnh, giữ nguyên cử chỉ chạm ngón

## 1. Dữ liệu hiện có

Hai thư mục `CDIOII/Morse_Webcam_Dataset/data/raw/images/sign_language_digits` và `datasets/data/raw/images/sign_language_digits` là hai bản sao của 2.062 ảnh từ [Sign Language Digits Dataset](https://github.com/ardamavi/Sign-Language-Digits-Dataset). Sau khi xem ảnh người dùng chỉ ra và các lớp tương ứng, có thể dùng lại tư thế thực tế: **9 → cái–trỏ, 7 → cái–áp út, 6 → cái–út, 5 → tay mở/nghỉ**. Đây là nhãn tư thế dẫn xuất từ nội dung ảnh, không phải quy ước số 1 = dot, số 2 = dash. Các lớp còn lại chưa nhập để tránh mơ hồ.

Đã đồng bộ 824 ảnh (204/206/207/207 cho bốn lớp) vào `data/images/morse/`. Không sửa hoặc xóa ảnh nguồn. `source_manifest.csv` giữ đường dẫn gốc, lớp số, nhãn tư thế, nhóm và hash. `dataset_protocol.json` ghi rõ cách ánh xạ cùng giới hạn. Để tái tạo:

```powershell
.\.venv\Scripts\python.exe -X utf8 src/import_pose_images.py
.\.venv\Scripts\python.exe -X utf8 src/train_image_models.py --split-by session
```

Bản dữ liệu này thiếu mã người/phiên được xác nhận. Vì vậy ảnh nguồn được nhóm theo khối 100 số thứ tự tên file, nhằm hạn chế các ảnh tên gần nhau lọt vào nhiều tập. Đó chỉ là **nhóm thay thế theo tên file**, không phải chia độc lập theo người hay phiên quay thật. Không cho phép báo cáo `--split-by subject` cho các ảnh công khai chưa có ID. Số ảnh độc lập: train 570, validation 122, test 132. Các số này không phải độ chính xác webcam hay người mới; có thể lạc quan do ảnh nguồn tương tự nhau. Không chỉnh mô hình theo kết quả test. Đọc chỉ số mới nhất từ metrics thay vì dùng điểm 100% của mô hình cũ một hướng tay.

## 2. Mô hình học gì?

**Đính chính kiểm kê:** ngoài bộ ảnh số, `CDIOII` còn có IPN HandS: ba bộ landmark và nhãn hành động theo đoạn thời gian, gồm click, double-click, zoom, grab. Đây là nguồn dữ liệu hành động có thể tận dụng. Tuy nhiên tên khung `.jpg` bên trong JSON không phải ảnh RGB; cần tìm video/ảnh gốc tương ứng để huấn luyện từ điểm ảnh. Xem mục `ipn_hands` trong báo cáo kiểm kê. Theo [tài liệu IPN HandS](https://github.com/GibranBenitez/IPN-hand/blob/master/README_HandS.md), phải trích từ video gốc 640×480 ở 30 fps để khớp chỉ số nhãn; không trộn với bộ khung 320×240 có timeline khác.

Không kết luận phải tự thu toàn bộ dữ liệu. Nếu ảnh/video có sẵn chứa đúng tư thế chạm ngón và có nhãn đã kiểm chứng thì dùng lại chúng. Cần xem khung RGB để xác nhận click/double-click tương ứng cặp ngón nào; tên hành động chưa chứng minh có đủ riêng ba tư thế cái–trỏ, cái–áp út, cái–út. Chỉ thu bổ sung những lớp còn thiếu sau bước đối chiếu.

```text
Webcam → MediaPipe tìm vùng tay → ảnh cắt vuông 128×128
       → grayscale 64×64 + CLAHE → HOG 1764 chiều
       → bộ phân loại đã học → tư thế → FSM → lệnh Morse
```

Đoạn trên mô tả nhánh HOG dự phòng. Nhánh webcam mới dùng `image_shape_learning.py`: MediaPipe trích 21 điểm từ **ảnh có nhãn**, tạo 210 khoảng cách chuẩn hóa rồi huấn luyện SVM/Random Forest. Webcam dùng cùng đặc trưng hình dạng để phân loại, không dùng quy tắc ngưỡng khoảng cách gán nhãn. Đây là học có giám sát trên đặc trưng trích từ ảnh, không phải tự huấn luyện CNN từ đầu. MediaPipe là bộ trích đặc trưng đã được huấn luyện sẵn; SVM mới là mô hình được huấn luyện bằng dữ liệu của đề tài.

Chạy riêng mô hình mới: `.\.venv\Scripts\python.exe -X utf8 src/image_shape_learning.py` sau khi đã có `metrics.json` từ nhánh HOG để dùng cùng cách chia nhóm. Nút huấn luyện trên web cập nhật cả hai mô hình. Mô hình mới lưu tại `models/image_shape_model.joblib`; báo cáo tại `reports/image_learning/shape_metrics.json` và danh sách split tại `shape_split_manifest.json`. Ảnh không phát hiện được bàn tay bị loại khỏi huấn luyện/đánh giá hình dạng và được ghi trong `undetected_images`; điểm test có điều kiện này không phải độ chính xác toàn hệ thống. Các nhóm webcam hiệu chỉnh chỉ nằm trong train, có trọng số mẫu 4 để cân bằng với bộ ảnh công khai. Các mục HOG còn lại dưới đây chỉ áp dụng nhánh HOG.

Mô hình học bốn nhãn: `PINCH_THUMB` (cái–trỏ), `PINCH_RING` (cái–áp út), `PINCH_PINKY` (cái–út), `UNKNOWN` (tay mở/nghỉ/tư thế khác). Không gán DOT/DASH hoặc SPACE/CONFIRM thành các ảnh tĩnh khác lớp khi chúng cùng tư thế. FSM xử lý những khác biệt về thời gian này.

## 3. Thu ảnh bổ sung trên web (không bắt buộc thu lại toàn bộ)

### Hiệu chỉnh khi camera nhận nhầm cử chỉ

Nhật ký thử trực tiếp ngày 19/11 ghi nhận mô hình phát DELETE do nhận cử chỉ thành PINCH_PINKY; ô Morse trống vì chưa có DOT/DASH, không phải lỗi vẽ ô. Dữ liệu webcam khác nền/ánh sáng cần các ảnh có nhãn đúng để hiệu chỉnh.

Chọn **Thu ảnh có nhãn**, nhập người `webcam01`, phiên `calib01`, tích **Hiệu chỉnh camera: phiên này chỉ dùng cho train**. Bật camera, chọn lần lượt bốn nhãn (nghỉ/mở tay, cái–trỏ, cái–áp út, cái–út), giữ mỗi tư thế và lưu 10–15 ảnh. Dừng lưu trước khi đổi nhãn. Giữ nguyên mã người/phiên cho cả bốn nhãn. Dừng camera rồi huấn luyện theo nhóm phiên/file.

Những nhóm được chọn hiệu chỉnh được ghi vào `data/images/morse/calibration_groups.json` và bắt buộc vào train. Chúng được loại khỏi thuật toán phân chia nhóm đánh giá trước khi gán train/val/test, nên các nhóm ảnh nguồn vẫn giữ cách chia cũ. Mô hình vẫn học từ pixel HOG và nhãn người dùng, không tự đổi predicted label thành nhãn đúng. Điểm validation/test không đo chất lượng trên người/điều kiện vừa hiệu chỉnh; cần phiên webcam khác để đánh giá thật. `metrics.json` ghi danh sách nhóm hiệu chỉnh ở `calibration_train_only_groups`.

1. Chạy `.\.venv\Scripts\python.exe -X utf8 src/web_server.py`, mở http://localhost:8000 và chọn **Thu ảnh có nhãn để huấn luyện**.
2. Nhập mã người `P01`, phiên `S01`, bật camera. Chọn nhãn đúng rồi giữ tư thế tương ứng.
3. Bấm **Bắt đầu lưu ảnh**. Hệ thống lưu khoảng 1 ảnh/giây. Xoay nhẹ, thay đổi góc/khoảng cách. Dừng lưu trước khi đổi nhãn. `UNKNOWN` cần có ảnh tay mở và các tư thế ngoài ba cặp chạm.
4. Thu đủ 4 nhãn trong mỗi phiên; phiên mới phải thực sự thay đổi điều kiện. Khuyến nghị 50–100 ảnh khác nhau/nhãn/phiên, ít nhất 3 phiên độc lập; tốt hơn là nhiều người và nhiều phiên. Không đổi tên phiên cho cùng một đợt ảnh để vượt điều kiện chia tập.

Ảnh chỉ lưu khi chủ động bật lưu: `data/images/morse/<người>/<phiên>/<nhãn>/<hash>.png`. Người thu gán nhãn, không tự gán bằng hình học. Ảnh trùng hoàn toàn bị bỏ. Các ảnh gần giống vẫn có thể tồn tại nên phải chia theo nhóm, không chia ngẫu nhiên các khung liền nhau.

## 4. Huấn luyện

Dừng camera rồi bấm **Huấn luyện từ ảnh đã thu**, hoặc:

```powershell
.\.venv\Scripts\python.exe -X utf8 src/train_image_models.py --split-by session
```

Có ít nhất 3 người có ID thật thì có thể dùng `--split-by subject`. Tối thiểu 30 ảnh khác nhau mỗi lớp và 3 nhóm; cả train/val/test phải đủ 4 lớp. Đây là điều kiện tối thiểu để chạy, không đảm bảo chất lượng. Tỷ lệ mục tiêu 70/15/15 có thể khác tỷ lệ thực tế do giữ nguyên nhóm. Sau chia nhóm, chỉ tập train được tạo thêm ảnh lật gương và biến đổi ánh sáng, nhiễu, xoay nhẹ. 570 ảnh train gốc tạo 2.280 biến thể học; không trình bày đây là 2.280 ảnh độc lập do người dùng thu.

So sánh SVM RBF (`C=1`, `C=10`), KNN và Random Forest bằng macro F1 trên validation gốc và phiên bản lật gương của cùng các ảnh đó. Những biến thể validation không đưa vào model.fit và không tăng số mẫu độc lập. Chọn mô hình và ngưỡng chấp nhận trên validation; chỉ báo cáo test sau khi chọn xong, có báo thêm kết quả ảnh test lật gương. Sản phẩm:

- `models/image_pose_model.joblib`: mô hình, ngưỡng, phiên bản HOG, báo cáo.
- `reports/image_learning/metrics.json`: kết quả validation, Accuracy, macro F1, từng lớp, confusion matrix, coverage và accuracy trên phần được chấp nhận.
- `reports/image_learning/split_manifest.csv`: từng ảnh, nhãn, nhóm và tập để kiểm tra rò rỉ dữ liệu.

Huấn luyện bằng nút chạy nền và nạp mô hình khi hoàn tất. Huấn luyện từ CLI thì khởi động lại máy chủ. Không thu ảnh đồng thời với huấn luyện để cố định dữ liệu.

## 5. Nhận diện

**Nếu tư thế đổi nhưng ô Morse trống:** kiểm tra dòng `Đã/Chưa mở khóa nhập`, FSM và sự kiện cuối. Tư thế PINCH_THUMB chưa đồng nghĩa đã phát DOT/DASH: hệ thống cần sẵn sàng rồi nhận đủ nhịp giữ/thả. Nếu mô hình không nhận được tay mở để tự mở khóa, bật camera và bấm **Mở khóa nhập**. Nút này chỉ kích hoạt tiếp nhận, giữ nguyên văn bản và không tự sinh dot/dash. Sau đó chạm cái–trỏ và thả hoặc giữ; sự kiện cuối phải hiện DOT/DASH, ô Morse phải thêm ký hiệu. Nếu sự kiện cuối là CONFIRM/SPACE thì mã đã được chuyển vào văn bản; DELETE xóa tín hiệu. Mất tay hoặc thiếu tin cậy kéo dài vẫn có thể hủy cử chỉ để tránh nhập sai; xem dòng chẩn đoán.

**Chẩn đoán video 19/11/2026:** bản ghi cho thấy MediaPipe vẫn bám tay nhưng lớp dự đoán có confidence khoảng 34–44%, dưới ngưỡng 55%; tay mở cũng có lúc bị dự đoán là cái–trỏ. Đây là vấn đề phân loại ảnh, không phải thiếu điểm bàn tay. Mô hình cũ đúng 121/122 ảnh validation gốc nhưng chỉ 34/122 khi lật gương. Đã huấn luyện cả hai hướng tay và nhiễu/ánh sáng trên train; mô hình mới được chọn là KNN, validation gốc và validation lật gương đều đúng 119/122. Test gốc 128/132 (96,97%), test lật gương 129/132 (97,73%), không phải hai tập độc lập và chưa phải điểm webcam thật. Không dùng ảnh chụp màn hình có vẽ bộ xương như dữ liệu huấn luyện; chúng khác ảnh pixel đầu vào sạch.

**Sửa lỗi webcam 19/11/2026:** huấn luyện hiện cắt vùng tay trên ảnh công khai bằng cùng công thức vuông, padding 1,45 như trình duyệt. Cache ở `data/processed/image_crops/`, chỉ là kết quả có thể tái tạo, không phải ảnh độc lập mới. Mô hình cập nhật đã dùng 730 ảnh định vị được bàn tay và 94 ảnh giữ toàn khung do MediaPipe không tìm thấy tay; xem `image_framing` trong metrics. Khi thử ảnh tải lên, máy chủ cũng định vị và cắt tay trước dự đoán. Ảnh webcam khác nền/ánh sáng vẫn có thể bị từ chối; chưa có đo độ chính xác webcam người dùng thực tế.

Khung không chắc chắn dưới 0,2 giây tạm giữ FSM, không sinh lệnh và không tính khoảng đó vào thời gian giữ dash. Khi có khung hợp lệ lại, xử lý tiếp; khi thiếu tin cậy kéo dài hoặc không thấy tay thì hủy cử chỉ và yêu cầu mở tay. Điều này tránh mất dot ngay khi ngón chuyển từ chạm sang mở và vẫn giữ chống lặp delete.

Dòng chẩn đoán mới hiển thị lớp có xác suất cao nhất, confidence, ngưỡng và lý do từ chối/chờ mở tay. Nút **Lưu ảnh đầu vào để kiểm tra** tải đúng vùng ảnh mà mô hình nhận; không tự thêm ảnh này vào dữ liệu huấn luyện. Nếu vẫn UNKNOWN, dùng ảnh đó để kiểm tra lỗi mô hình thay vì hạ ngưỡng tùy ý.

Chọn **Nhận diện bằng mô hình ảnh đã học**. Mở tay trước, rồi dùng cử chỉ cũ. Kết quả dưới ngưỡng bị từ chối và hủy cử chỉ dở dang; lớp UNKNOWN được dự đoán đủ tin cậy là nghỉ hợp lệ. Không dùng detector hình học để quyết định nhãn trong chế độ này. FSM/bộ đệm cũ xử lý dot/dash/chốt/space/delete và giữ chữ đã nhập.

**Thử một ảnh** dự đoán ảnh riêng, không nhập lệnh vào văn bản. Dùng ảnh đã cắt vùng tay giống ảnh huấn luyện. Chế độ **Tham khảo: landmark/hình học** dùng riêng để so sánh và thử nút mô phỏng; đó không phải học từ ảnh. Nếu thiếu mô hình ảnh, chế độ ảnh báo lỗi rõ ràng, không âm thầm chuyển sang hình học.

## 6. Trình bày báo cáo

Đã có mô hình học từ 824 ảnh nguồn thật sau khi đối chiếu nhãn tư thế. Không cần video để học bốn tư thế tĩnh; FSM xử lý thời gian ở lúc chạy. Ảnh tự thu giúp bổ sung ánh sáng, nền, tay mở/tư thế lạ và người dùng webcam. Kiểm thử ảnh tổng hợp chỉ chạy trong thư mục tạm để kiểm tra cơ chế lưu/nạp, không phải số liệu thực nghiệm. Các điểm test ảnh thật nằm trong `reports/image_learning/metrics.json`, cần ghi kèm giới hạn nhóm tên file nói trên.

Chia theo phiên không đồng nghĩa chia theo người. Cần báo cáo cả macro F1, confusion matrix, coverage và thử với người chưa xuất hiện trong train. Confidence không bảo đảm đúng khi ảnh khác nền, ánh sáng hay người dùng. Văn bản vẫn lưu RAM, chưa thành cơ sở dữ liệu.
