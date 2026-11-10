# Hệ Thống Nhận Diện Cử Chỉ Tay & Giải Mã Mã Morse Qua Webcam
**(Real-Time Webcam Hand Gesture Recognition & Morse Code Decoder)**

> 🇻🇳 **[Phiên bản Tiếng Việt](#phiên-bản-tiếng-việt)** | 🇬🇧 **[English Version](#english-version)**

---

<a name="phiên-bản-tiếng-việt"></a>
# 🇻🇳 PHIÊN BẢN TIẾNG VIỆT

## Mục Lục
- [Chương 1: Giới Thiệu Đề Tài & Phạm Vi](#chương-1-giới-thiệu-đề-tài--phạm-vi)
- [Chương 2: Sơ Đồ Bàn Tay & Bảng Mã Morse](#chương-2-sơ-đồ-bàn-tay--bảng-mã-morse)
- [Chương 3: Quy Định Nhãn Dữ Liệu (Data Labeling) & Nghiên Cứu Nhãn](#chương-3-quy-định-nhãn-dữ-liệu-data-labeling--nghiên-cứu-nhãn)
- [Chương 4: Tiền Xử Lý Dữ Liệu & Các Mô Hình Học Máy (Machine Learning)](#chương-4-tiền-xử-lý-dữ-liệu--các-mô-hình-học-máy-machine-learning)
- [Chương 5: Cơ Chế Xử Lý Thời Gian, Trạng Thái (FSM) & Giải Mã](#chương-5-cơ-chế-xử-lý-thời-gian-trạng-thái-fsm--giải-mã)
- [Chương 6: Đánh Giá Mô Hình & Hệ Thống (Evaluation)](#chương-6-đánh-giá-mô-hình--hệ-thống-evaluation)
- [Chương 7: Cấu Trúc Dự Án & Hướng Dẫn Cài Đặt, Chạy Thử](#chương-7-cấu-trúc-dự-án--hướng-dẫn-cài-đặt-chạy-thử)
- [Chương 8: Đề Cương Báo Cáo & Danh Mục Sản Phẩm](#chương-8-đề-cương-báo-cáo--danh-mục-sản-phẩm)

---

## Chương 1: Giới Thiệu Đề Tài & Phạm Vi

### 1.1. Mục tiêu đề tài
Dự án xây dựng một hệ thống **thị giác máy tính (Computer Vision)** kết hợp **học máy (Machine Learning)** đơn giản và hiệu quả. Hệ thống sử dụng webcam thông thường để:
1. Bắt chuyển động bàn tay và xác định vị trí các ngón tay qua **MediaPipe Hands**.
2. Nhận diện các cử chỉ tay quy ước bằng các thuật toán học máy quen thuộc (**KNN, SVM, Random Forest**).
3. Chuyển đổi cử chỉ chạm ngón tay thành các ký hiệu **mã Morse (chấm `.` và gạch `-`)**, sau đó ghép lại và giải mã thành chữ cái, từ ngữ hoàn chỉnh.

### 1.2. Hai chế độ hoạt động (Dùng chung Webcam)
Hệ thống hỗ trợ 2 chế độ độc lập:

| Chế độ | Đầu vào từ người dùng | Cách hệ thống xử lý | Kết quả đầu ra |
| :--- | :--- | :--- | :--- |
| **1. Chế độ Morse (Chính)** | Thực hiện các cử chỉ chạm ngón tay quy ước | Phân loại tư thế tay + đo thời gian giữ/số lần chạm $\rightarrow$ Ghép mã `.` và `-` | Ký tự (A–Z, 0–9) và ghép thành câu văn bản |
| **2. Chế độ Ký hiệu tĩnh** | Đưa từng tư thế bàn tay tĩnh (ví dụ: chữ cái ký hiệu A, B, C...) | Dùng mô hình học máy phân loại trực tiếp từng khung hình | Chữ cái tương ứng của tư thế đó |

> [!NOTE]
> **Phân biệt phạm vi đề tài:**
> Nhận diện một vài cử chỉ ngón tay quy ước để nhập mã Morse **chưa phải là dịch toàn bộ ngôn ngữ ký hiệu (Sign Language)**. Ngôn ngữ ký hiệu ngoài đời thực cần cả chuyển động liên tục của hai tay, nét mặt và tư thế cơ thể. Đề tài này tập trung giải quyết bài toán vừa sức: **nhận diện cử chỉ rời rạc để nhập dữ liệu qua mã Morse**.

---

## Chương 2: Sơ Đồ Bàn Tay & Bảng Mã Morse

### 2.1. Sơ đồ 21 điểm mốc bàn tay (MediaPipe Hands)
MediaPipe xác định 21 điểm mốc (landmarks) trên bàn tay, mỗi điểm có tọa độ $(x, y, z)$:

```text
                 8 (Đầu ngón trỏ)    12 (Đầu ngón giữa) 16 (Đầu ngón áp út)
                     ●                   ●                ●
                     |                   |                |
                     7                   11               15          20 (Đầu ngón út)
                     |                   |                |            ●
                     6                   10               14           |
                     |                   |                |           19
    4 (Đầu ngón cái) 5 (Khớp gốc trỏ)    9 (Gốc giữa)     13 (Gốc áp)  |
        ●            ●-------------------●----------------●           18
         \          /                                      \           |
          3        /                                        \         17 (Khớp gốc út)
           \      /                                          \-------●
            2    /                                                  /
             \  /                  LÒNG BÀN TAY                    /
              1                                                   /
               \                                                 /
                \--------------------- 0 -----------------------/
                                   (CỔ TAY)
```

**Các điểm mốc mấu chốt được dùng trong đề tài:**
- `0`: Cổ tay (Wrist) — dùng làm mốc gốc để chuẩn hóa tọa độ.
- `4`: Đầu ngón cái (Thumb Tip).
- `8`: Đầu ngón trỏ (Index Tip) — ngón chính chạm vào các ngón khác.
- `16`: Đầu ngón áp út (Ring Tip).
- `20`: Đầu ngón út (Pinky Tip).

---

### 2.2. Bảng tra cứu Mã Morse chuẩn (A–Z và 0–9)

| Chữ | Mã Morse | Chữ | Mã Morse | Chữ | Mã Morse | Số | Mã Morse |
| :-: | :--- | :-: | :--- | :-: | :--- | :-: | :--- |
| **A** | `.-` | **J** | `.---` | **S** | `...` | **0** | `-----` |
| **B** | `-...` | **K** | `-.-` | **T** | `-` | **1** | `.----` |
| **C** | `-.-.` | **L** | `.-..` | **U** | `..-` | **2** | `..---` |
| **D** | `-..` | **M** | `--` | **V** | `...-` | **3** | `...--` |
| **E** | `.` | **N** | `-.` | **W** | `.--` | **4** | `....-` |
| **F** | `..-.` | **O** | `---` | **X** | `-..-` | **5** | `.....` |
| **G** | `--.` | **P** | `.--.` | **Y** | `-.--` | **6** | `-....` |
| **H** | `....` | **Q** | `--.-` | **Z** | `--..` | **7** | `--...` |
| **I** | `..` | **R** | `.-.` | | | **8** | `---..` |
| | | | | | | **9** | `----.` |

---

## Chương 3: Quy Định Nhãn Dữ Liệu (Data Labeling) & Nghiên Cứu Nhãn

Đây là phần trọng tâm của khâu chuẩn bị dữ liệu học máy.

### 3.1. Bảng quy định 5 nhãn điều khiển hệ thống
Dữ liệu được quy ước theo 5 loại cử chỉ tay để điều khiển toàn bộ quá trình nhập liệu:

| STT | Nhãn (Label) | Cử chỉ người dùng thực hiện | Thao tác thời gian | Ý nghĩa điều khiển |
| :-: | :--- | :--- | :--- | :--- |
| **1** | `DOT` | Ngón trỏ (`8`) chạm ngón cái (`4`) | Chạm nhanh rồi thả ($< 0.4$ giây) | Nhập tín hiệu chấm `.` vào chuỗi Morse |
| **2** | `DASH` | Ngón trỏ (`8`) chạm ngón cái (`4`) | Giữ ngón tay chạm nhau ($\ge 0.4$ giây) | Nhập tín hiệu gạch `-` vào chuỗi Morse |
| **3** | `CONFIRM`| Ngón trỏ (`8`) chạm ngón áp út (`16`) | Chạm 2 lần liên tiếp (double-tap) | Kết thúc chuỗi, giải mã Morse thành chữ |
| **4** | `SPACE` | Ngón trỏ (`8`) chạm ngón áp út (`16`) | Chạm 1 lần duy nhất | Thêm dấu cách (khoảng trắng) giữa các từ |
| **5** | `DELETE` | Ngón trỏ (`8`) chạm ngón út (`20`) | Chạm 1 lần | Xóa 1 ký hiệu vừa bấm (hoặc xóa 1 chữ) |
| *Phụ*| `IDLE` | Bàn tay mở tự nhiên, các ngón không chạm nhau | Giữ trạng thái nghỉ | Trạng thái sẵn sàng, không nhập gì |

---

### 3.2. Mối liên hệ giữa "Nhãn tư thế tĩnh" và "Nhãn điều khiển Morse"
Một khung hình webcam tĩnh (ảnh đơn) **không thể tự biết** là người dùng đang "chạm nhanh" (`DOT`) hay "chạm giữ" (`DASH`), cũng như không biết là "chạm 1 lần" (`SPACE`) hay "chạm lần 1 của cú chạm đúp" (`CONFIRM`).

Vì vậy, hệ thống chia việc học máy thành 2 tầng rất rõ ràng và hợp lý:
1. **Tầng Mô hình Học máy (Phân loại tư thế ngón tay):**
   Mô hình học máy sẽ nhìn vào 21 điểm mốc của bàn tay tại khung hình đó và phân loại vào 4 tư thế:
   - `IDLE`: Bàn tay mở tự nhiên.
   - `PINCH_THUMB`: Ngón trỏ đang chạm ngón cái.
   - `PINCH_RING`: Ngón trỏ đang chạm ngón áp út.
   - `PINCH_PINKY`: Ngón trỏ đang chạm ngón út.
2. **Tầng Xử lý Logic & Thời gian (Sinh ra nhãn Morse cuối cùng):**
   - Nếu thấy `PINCH_THUMB` giữ dưới $0.4\text{s}$ rồi thả $\rightarrow$ Sinh nhãn **`DOT`**.
   - Nếu thấy `PINCH_THUMB` giữ trên $0.4\text{s}$ $\rightarrow$ Sinh nhãn **`DASH`**.
   - Nếu thấy `PINCH_RING` chạm 1 lần $\rightarrow$ Sinh nhãn **`SPACE`**; nếu chạm 2 lần nhanh $\rightarrow$ Sinh nhãn **`CONFIRM`**.
   - Nếu thấy `PINCH_PINKY` $\rightarrow$ Sinh nhãn **`DELETE`**.

---

### 3.3. Nguyên tắc gán nhãn dữ liệu (Data Labeling Principles)
Khi thu thập dữ liệu huấn luyện, nhóm tuân thủ 5 nguyên tắc:
1. **Một mẫu - một nhãn duy nhất:** Mỗi vector tọa độ bàn tay chỉ được gán đúng 1 nhãn tư thế tương ứng.
2. **Đúng cử chỉ quy ước:** Khi thu nhãn nào thì chỉ thực hiện dứt khoát cử chỉ đó, giữ các ngón tay còn lại mở tự nhiên.
3. **Đa dạng môi trường thu thập:** Thu thập dữ liệu từ nhiều người khác nhau, ở nhiều góc quay (thẳng, hơi nghiêng $15^\circ - 30^\circ$), khoảng cách khác nhau tới camera (gần 35cm, xa 70cm) và điều kiện ánh sáng khác nhau (đèn học, ban ngày).
4. **Loại bỏ dữ liệu lỗi (Data Cleaning):** Tự động loại bỏ các khung hình MediaPipe không nhận đủ 21 điểm mốc hoặc tay bị che khuất một phần.
5. **Thống nhất tên nhãn:** Tên nhãn viết hoa đồng nhất xuyên suốt trong toàn bộ file code và thư mục dữ liệu.

---

### 3.4. Nghiên cứu dữ liệu & Các cặp nhãn dễ nhầm lẫn (Confusable Labels)
Trong báo cáo môn học, đây là phần phân tích được đánh giá rất cao:
- **Cặp nhãn dễ nhầm nhất:** Tư thế chạm ngón áp út (`PINCH_RING` - dùng cho SPACE/CONFIRM) và chạm ngón út (`PINCH_PINKY` - dùng cho DELETE).
  - *Nguyên nhân:* Hai ngón này nằm sát nhau. Khi người dùng quay nghiêng bàn tay, ngón áp út có thể che mất ngón út, khiến MediaPipe dự đoán tọa độ hai đầu ngón này sát nhau dẫn đến mô hình đoán nhầm nhãn.
  - *Giải pháp thực tế:* Nhóm bổ sung các mẫu thu thập ở nhiều góc nghiêng, đồng thời bổ sung đặc trưng tính khoảng cách tương đối giữa các đầu ngón tay để mô hình phân biệt tốt hơn.
- **Phân bố nhãn (Class Distribution):** Cần đảm bảo số lượng mẫu thu thập giữa các tư thế tương đối đồng đều (ví dụ mỗi tư thế khoảng 300 – 500 mẫu), tránh trường hợp nhãn `IDLE` quá nhiều còn nhãn `PINCH_PINKY` quá ít làm lệch mô hình.

---

### 3.5. Quy tắc chia tập dữ liệu: 70% Train - 15% Val - 15% Test
Để đánh giá mô hình học máy một cách trung thực và chính xác:
- **Không chia ngẫu nhiên từng dòng:** Nếu một đoạn video quay tay liên tục mà cắt ảnh ra rồi chia ngẫu nhiên thì ảnh ở tập Train và Test gần như y hệt nhau $\rightarrow$ độ chính xác Test sẽ cao giả tạo (99%), nhưng khi người mới dùng thật thì lại nhận diện kém.
- **Chia theo phiên quay (Session-wise) hoặc theo người (Subject-wise):** Toàn bộ các mẫu trong 1 lần quay phải nằm trọn vẹn trong tập Train, hoặc tập Val, hoặc tập Test. Người thử nghiệm ở tập Test nên là người chưa từng xuất hiện ở tập Train.

---

## Chương 4: Tiền Xử Lý Dữ Liệu & Các Mô Hình Học Máy (Machine Learning)

### 4.1. Tại sao dùng tọa độ MediaPipe thay vì đưa cả ảnh vào CNN?
- Nếu đưa thẳng bức ảnh webcam vào mạng Deep Learning (như CNN): Mô hình sẽ rất nặng, dễ bị ảnh hưởng bởi phông nền đằng sau (tường, quần áo, người đi lại) và cần máy tính có card đồ họa rời (GPU).
- **Giải pháp của đề tài:** Dùng MediaPipe tách trước 21 điểm mốc xương bàn tay. Sau đó chỉ đưa **63 con số tọa độ** vào các thuật toán học máy cổ điển.
- **Lợi ích:** Mô hình cực nhẹ (chưa tới vài Megabyte), học xong trong vài giây, chạy siêu mượt trên CPU laptop bình thường với tốc độ trên 30 khung hình/giây (FPS).

### 4.2. Tiền xử lý 63 đặc trưng (Preprocessing)
Từ 21 điểm $(x, y, z)$, hệ thống thực hiện 3 bước đơn giản:
1. **Dời gốc tọa độ về cổ tay:** Lấy tọa độ của tất cả các điểm trừ đi tọa độ cổ tay (điểm số `0`). Sau bước này, dù bạn để tay ở góc trái, góc phải hay giữa màn hình thì hình dáng bàn tay đều có cùng một gốc tọa độ.
2. **Chuẩn hóa tỷ lệ kích thước:** Chia tất cả tọa độ cho khoảng cách xa nhất từ cổ tay tới đầu ngón tay. Nhờ vậy, đưa tay lại gần camera (tay to) hay để tay ra xa (tay nhỏ) thì kích thước bàn tay đều được đưa về cùng một chuẩn từ 0 đến 1.
3. **Xử lý tay trái / tay phải:** Nếu phát hiện tay trái, hệ thống lật trục ngang để đưa về tọa độ giống như tay phải, giúp một mô hình nhận diện tốt cả 2 tay.
4. **Kết quả:** Thu được một mảng (vector) gồm 63 số biểu diễn chính xác tư thế bàn tay.

---

### 4.3. Ba thuật toán học máy được so sánh

Hệ thống huấn luyện và so sánh 3 thuật toán phổ biến:

#### 1. $K$-Nearest Neighbors ($K$-NN)
- **Cách hoạt động:** Khi có một tư thế tay mới, KNN tính khoảng cách giữa 63 con số của tay hiện tại với tất cả các mẫu trong tập dữ liệu đã lưu. Sau đó nó chọn ra $K$ mẫu gần giống nhất (ví dụ $K=5$) và xem đa số các mẫu đó thuộc nhãn nào thì đoán nhãn đó.
- **Ưu điểm:** Cực kỳ trực quan, dễ hiểu, không cần bước huấn luyện phức tạp.
- **Nhược điểm:** Nếu tập dữ liệu lớn thì tốc độ so sánh sẽ chậm dần.

#### 2. Support Vector Machine (SVM)
- **Cách hoạt động:** SVM tìm một "ranh giới" (đường phân chia tối ưu) trong không gian 63 chiều để ngăn cách rõ ràng giữa các nhóm tư thế (`IDLE`, `PINCH_THUMB`, `PINCH_RING`, `PINCH_PINKY`). Khi dùng nhân RBF, SVM có thể uốn cong ranh giới để chia các nhóm phức tạp.
- **Ưu điểm:** Nhận diện rất chính xác, chạy dự đoán cực nhanh (chỉ mất khoảng 1 mili-giây mỗi khung hình).
- **Nhược điểm:** Cần chuẩn hóa dữ liệu cẩn thận và chọn đúng tham số $C$ và $\gamma$.

#### 3. Random Forest (Rừng ngẫu nhiên)
- **Cách hoạt động:** Kết hợp nhiều "cây quyết định" (Decision Trees). Mỗi cây sẽ đặt ra một loạt câu hỏi dạng Nếu - Thì (ví dụ: "Khoảng cách ngón cái và ngón trỏ có nhỏ hơn 0.2 không?", "Ngón út có duỗi thẳng không?"). Cuối cùng, tất cả các cây cùng bỏ phiếu để chọn ra nhãn chiến thắng.
- **Ưu điểm:** Rất ổn định, ít bị học vẹt (overfitting).
- **Nhược điểm:** Kích thước file mô hình lớn hơn SVM.

---

## Chương 5: Cơ Chế Xử Lý Thời Gian, Trạng Thái (FSM) & Giải Mã

### 5.1. Vấn đề nhập lặp khi dùng Webcam (Debouncing)
Webcam quay liên tục khoảng 30 khung hình mỗi giây (30 FPS). Nếu người dùng chạm ngón tay giữ trong 1 giây mà không có cơ chế chặn:
$\implies$ Hệ thống sẽ nhận diện liên tục và nhập tới **30 dấu chấm (`...............`)**!

**Quy tắc bắt buộc của hệ thống:**
*Mỗi lần thực hiện cử chỉ chỉ được ghi nhận đúng 1 lần. Sau khi nhập xong một tín hiệu, người dùng phải mở bàn tay về trạng thái nghỉ (`IDLE`) thì hệ thống mới cho phép nhập tiếp tín hiệu sau.*

### 5.2. Máy trạng thái (FSM) hoạt động như thế nào?
Hệ thống sử dụng một máy trạng thái đơn giản để quản lý:
1. **Lọc nhiễu đa khung hình:** Kiểm tra 5 khung hình liên tiếp, chỉ khi đa số các khung hình cùng ra một tư thế thì mới tin tưởng (tránh bị giật/nhấp nháy).
2. **Trạng thái sẵn sàng (`IDLE`):** Bàn tay mở tự nhiên.
3. **Khi chạm ngón cái + ngón trỏ:**
   - Nếu thả ra nhanh ($< 0.4$ giây) $\rightarrow$ Thêm dấu `.` (DOT).
   - Nếu giữ lâu ($\ge 0.4$ giây) $\rightarrow$ Thêm dấu `-` (DASH), sau đó **khóa lại** không cho nhập tiếp.
4. **Khi chạm ngón áp út + ngón trỏ:**
   - Nếu chạm 1 lần rồi thôi $\rightarrow$ Thêm dấu cách `SPACE`.
   - Nếu chạm lần 2 trong vòng $0.45$ giây $\rightarrow$ Phát lệnh **`CONFIRM`** (giải mã chuỗi Morse hiện tại thành chữ cái tương ứng).
5. **Khi chạm ngón út + ngón trỏ:** $\rightarrow$ Lệnh **`DELETE`** (xóa tín hiệu hoặc xóa chữ vừa gõ).
6. **Mở khóa:** Chỉ khi người dùng mở tay ra (về trạng thái `IDLE`), máy trạng thái mới quay lại sẵn sàng cho lần gõ tiếp theo.

### 5.3. Ví dụ nhập chữ "S" (Mã Morse: `...`)
1. Để tay mở tự nhiên (`IDLE`).
2. Chạm nhanh ngón trỏ và cái lần 1 $\rightarrow$ Màn hình hiện `.`. Mở tay ra.
3. Chạm nhanh ngón trỏ và cái lần 2 $\rightarrow$ Màn hình hiện `..`. Mở tay ra.
4. Chạm nhanh ngón trỏ và cái lần 3 $\rightarrow$ Màn hình hiện `...`. Mở tay ra.
5. Chạm 2 lần ngón trỏ và áp út (`CONFIRM`) $\rightarrow$ Hệ thống tra bảng mã thấy `...` là chữ **`S`**, lưu chữ **`S`** vào văn bản và làm trống chuỗi Morse để sẵn sàng cho chữ tiếp theo.

---

## Chương 6: Đánh Giá Mô Hình & Hệ Thống (Evaluation)

### 6.1. Các chỉ số đánh giá mô hình học máy (trên tập Test 15%)
Để báo cáo kết quả huấn luyện của 3 mô hình (KNN, SVM, Random Forest), hệ thống dùng các chỉ số:
- **Accuracy (Độ chính xác chung):** Tỷ lệ đoán đúng trên toàn bộ tập kiểm thử.
- **Precision (Độ chuẩn xác):** Khi mô hình đoán là nhãn X, thì có bao nhiêu % thực sự là nhãn X (đo lường việc không đoán bừa).
- **Recall (Độ nhạy / Độ bao quát):** Trong toàn bộ các cử chỉ nhãn X người dùng làm, mô hình nhận diện đúng được bao nhiêu % (đo lường việc không bỏ sót).
- **F1-score:** Điểm số trung bình điều hòa kết hợp cả Precision và Recall.
- **Inference Time:** Thời gian mô hình tính toán xong 1 khung hình (tính bằng mili-giây, yêu cầu $< 10\text{ms}$ để chạy mượt thời gian thực).
- **Confusion Matrix (Ma trận nhầm lẫn):** Bảng hiển thị trực quan các cặp nhãn hay bị đoán nhầm sang nhau.

*(Các con số cụ thể sẽ được ghi nhận tự động khi chạy lệnh `python3 src/evaluate.py`).*

### 6.2. Đánh giá thực tế khi chạy với Webcam
Độ chính xác trên tập dữ liệu cao chưa chắc lúc dùng thật đã tốt. Cần kiểm tra thêm:
- **Tốc độ khung hình (FPS):** Đạt 25 – 30 FPS trên webcam laptop thông thường.
- **Khả năng chống lặp:** Giữ nguyên tay trong 3 giây xem có bị nhảy hàng chục chữ không (mục tiêu: chỉ ra đúng 1 chữ).
- **Độ chính xác ghép chữ:** Thử gõ các từ mẫu như `SOS`, `HELLO` xem tỷ lệ gõ đúng chữ cái và đúng từ đạt bao nhiêu %.
- **Thử với người mới:** Đưa cho một người bạn chưa từng tham gia thu thập dữ liệu dùng thử xem hệ thống có nhận diện tốt không.

---

## Chương 7: Cấu Trúc Dự Án & Hướng Dẫn Cài Đặt, Chạy Thử

### 7.1. Cấu trúc thư mục mã nguồn
```text
morse-code-decoder-webcam/
├── data/
│   ├── raw/                        # Chứa các file CSV dữ liệu thô thu thập từ webcam
│   ├── processed/                  # Chứa dữ liệu đã tiền xử lý 63 đặc trưng (train, val, test)
│   └── dataset_description.md      # Tài liệu mô tả chi tiết nguồn dữ liệu, nhãn và số lượng mẫu
├── models/
│   ├── scaler_config.json          # File lưu các ngưỡng thời gian, ngưỡng chạm và mô hình tốt nhất
│   └── *.pkl                       # File lưu mô hình đã huấn luyện (KNN, SVM, Random Forest)
├── src/
│   ├── config.py                   # Cấu hình danh sách nhãn, thông số mặc định và đường dẫn
│   ├── data_collection.py          # Script bật webcam để thu thập dữ liệu và gán nhãn
│   ├── hand_tracker.py             # Script trích xuất 21 điểm mốc xương tay bằng MediaPipe
│   ├── preprocessing.py            # Tiền xử lý, chuẩn hóa 63 tọa độ và chia tập 70/15/15
│   ├── train_models.py             # Huấn luyện và so sánh KNN, SVM, Random Forest
│   ├── evaluate.py                 # Đánh giá trên tập Test, tính Accuracy, F1 và vẽ Confusion Matrix
│   ├── fsm_controller.py           # Quản lý trạng thái, phân biệt chạm nhanh/giữ và chống lặp
│   └── morse_decoder.py            # Từ điển tra cứu mã Morse sang chữ cái A-Z, 0-9
├── tests/
│   └── test_morse_fsm.py           # Bộ 13 bài test tự động kiểm tra logic giải mã và FSM
├── reports/
│   ├── figures/                    # Nơi lưu ảnh biểu đồ phân bố nhãn và ma trận nhầm lẫn
│   └── report_outline.md           # Đề cương gợi ý chi tiết các mục cần viết trong báo cáo
├── requirements.txt                # Danh sách các thư viện Python cần cài
└── README.md                       # Tài liệu tổng quan toàn bộ dự án
```

### 7.2. Hướng dẫn cài đặt và các lệnh chạy

#### Bước 1: Mở Terminal, tạo môi trường ảo và cài thư viện
```bash
# 1. Tạo môi trường ảo Python
python3 -m venv venv
source venv/bin/activate        # Trên macOS / Linux
# hoặc: venv\Scripts\activate   # Trên Windows

# 2. Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

#### Bước 2: Chạy các bước thực hiện dự án
```bash
# Bước A: Chạy kiểm thử tự động để kiểm tra logic giải mã Morse
python3 -m unittest discover tests -v

# Bước B: Thu thập dữ liệu cử chỉ ngón tay qua webcam (bấm phím 0, 1, 2, 3 để chọn nhãn)
python3 src/data_collection.py --task morse --subject user_01 --session s01

# Bước C: Tiền xử lý dữ liệu và chia tập (70% Train, 15% Val, 15% Test)
python3 src/preprocessing.py --task morse

# Bước D: Huấn luyện và tìm tham số tốt nhất cho 3 mô hình (KNN, SVM, Random Forest)
python3 src/train_models.py --task morse

# Bước E: Đánh giá mô hình trên tập Test và xuất biểu đồ ma trận nhầm lẫn
python3 src/evaluate.py --task morse
```

---

## Chương 8: Đề Cương Báo Cáo & Danh Mục Sản Phẩm

### 8.1. Các chương cần viết trong báo cáo môn học / đồ án
1. **Chương 1: Tổng quan đề tài:** Lý do chọn đề tài, mục tiêu, phạm vi ứng dụng mã Morse qua cử chỉ ngón tay.
2. **Chương 2: Cơ sở lý thuyết:** Giới thiệu MediaPipe Hands (21 điểm mốc), nguyên lý của 3 thuật toán học máy (KNN, SVM, Random Forest), bảng mã Morse quốc tế.
3. **Chương 3: Thu thập và gán nhãn dữ liệu:** Mô tả cách gán 5 nhãn điều khiển, quy trình thu thập từ nhiều người, cách làm sạch dữ liệu và chia tập chống rò rỉ.
4. **Chương 4: Thiết kế hệ thống:** Luồng xử lý từ webcam qua MediaPipe đến phân loại tư thế, cơ chế máy trạng thái (FSM) để phân biệt chạm nhanh/giữ và chống nhập lặp.
5. **Chương 5: Kết quả thực nghiệm:** Bảng so sánh Accuracy, Precision, Recall, F1 và thời gian chạy của KNN, SVM, Random Forest; phân tích ma trận nhầm lẫn (cặp nhãn nào hay nhầm và tại sao).
6. **Chương 6: Kết luận và hướng mở rộng:** Những điểm đã làm được, hạn chế hiện tại (ánh sáng yếu, góc nghiêng) và hướng phát triển (nhận diện thêm cử chỉ hai tay).

### 8.2. Checklist sản phẩm cần có khi nộp bài
- [ ] **1. Bộ dữ liệu:** Các file CSV có gắn nhãn đầy đủ, tài liệu mô tả số lượng mẫu.
- [ ] **2. Mã nguồn hoàn chỉnh:** Code thu thập, tiền xử lý, huấn luyện, đánh giá và kiểm thử.
- [ ] **3. Mô hình đã lưu:** File `.pkl` của mô hình tốt nhất (thường là SVM hoặc Random Forest).
- [ ] **4. Kết quả thực nghiệm:** Bảng chỉ số đánh giá và ảnh ma trận nhầm lẫn trong thư mục `reports/figures/`.
- [ ] **5. Báo cáo & Slide:** Quyển báo cáo chi tiết và slide thuyết trình.

---
---

<a name="english-version"></a>
# 🇬🇧 ENGLISH VERSION

## Table of Contents
- [Chapter 1: Project Introduction & Scope](#chapter-1-project-introduction--scope)
- [Chapter 2: Hand Landmark Model & Morse Code Table](#chapter-2-hand-landmark-model--morse-code-table)
- [Chapter 3: Data Labeling Rules & Label Analysis](#chapter-3-data-labeling-rules--label-analysis)
- [Chapter 4: Data Preprocessing & Machine Learning Models](#chapter-4-data-preprocessing--machine-learning-models)
- [Chapter 5: Timing Logic, State Machine (FSM) & Decoding](#chapter-5-timing-logic-state-machine-fsm--decoding)
- [Chapter 6: Model & System Evaluation](#chapter-6-model--system-evaluation)
- [Chapter 7: Project Structure, Installation & Usage](#chapter-7-project-structure-installation--usage)
- [Chapter 8: Report Outline & Deliverables Checklist](#chapter-8-report-outline--deliverables-checklist)

---

## Chapter 1: Project Introduction & Scope

### 1.1. Objectives
This project implements an efficient, straightforward **Computer Vision** and **Machine Learning** system that uses a standard webcam to:
1. Detect and track hand landmarks in real time via **MediaPipe Hands**.
2. Recognize defined finger pinch gestures using classic machine learning classifiers (**KNN, SVM, Random Forest**).
3. Translate finger gestures into **Morse code signals (dot `.` and dash `-`)**, assembling them into letters and words.

### 1.2. Two Operating Modes
- **Morse Mode (Primary):** User performs finger pinches $\rightarrow$ System measures hold duration and tap count $\rightarrow$ Decodes Morse into text.
- **Static Sign Mode:** User shows static hand shapes $\rightarrow$ Classified directly by ML model into discrete letters.

> [!NOTE]
> **Project Scope Clarification:**
> Recognizing discrete finger gestures for Morse input is **not full Sign Language Translation**. Natural sign language involves continuous arm movement, facial expressions, and body orientation. This project focuses on a practical, well-bounded scope: discrete gesture recognition for Morse code data entry.

---

## Chapter 2: Hand Landmark Model & Morse Code Table

### 2.1. MediaPipe Hands 21 Landmarks
MediaPipe detects 21 3D points $(x, y, z)$ on each hand:
- `0`: Wrist — baseline origin point for coordinate normalization.
- `4`: Thumb Tip.
- `8`: Index Finger Tip — primary active finger touching other fingers.
- `16`: Ring Finger Tip.
- `20`: Pinky Tip.

### 2.2. Morse Code Dictionary (ITU Standard)
- Letters A–Z (e.g., `A` = `.-`, `S` = `...`, `O` = `---`).
- Digits 0–9 (e.g., `1` = `.----`, `0` = `-----`).

---

## Chapter 3: Data Labeling Rules & Label Analysis

### 3.1. Five Control Labels
The dataset is annotated with 5 control gestures to drive input:

| ID | Label | User Gesture | Action / Timing | Function |
| :-: | :--- | :--- | :--- | :--- |
| **1** | `DOT` | Index (`8`) touches Thumb (`4`) | Quick tap ($< 0.4$ s) | Appends a dot `.` to Morse buffer |
| **2** | `DASH` | Index (`8`) touches Thumb (`4`) | Sustained hold ($\ge 0.4$ s) | Appends a dash `-` to Morse buffer |
| **3** | `CONFIRM` | Index (`8`) touches Ring finger (`16`) | Double-tap within $0.45$ s | Decodes current Morse buffer into a letter |
| **4** | `SPACE` | Index (`8`) touches Ring finger (`16`) | Single tap | Inserts a whitespace ` ` between words |
| **5** | `DELETE` | Index (`8`) touches Pinky (`20`) | Single tap | Deletes last Morse signal or character |
| *Aux* | `IDLE` | Open neutral hand, no fingers touching | Rest state | Ready state / No command |

### 3.2. Mapping Static Hand Poses to Dynamic Morse Labels
A single static video frame cannot differentiate between a quick tap (`DOT`) and a sustained hold (`DASH`), nor between a single tap (`SPACE`) and the first tap of a double tap (`CONFIRM`).

Therefore, the system cleanly separates the workflow into two levels:
1. **Machine Learning Classifier (Static Hand Pose Recognition):**
   Classifies the 21 landmarks of each individual frame into 4 physical hand poses:
   - `IDLE` (open hand)
   - `PINCH_THUMB` (index + thumb)
   - `PINCH_RING` (index + ring)
   - `PINCH_PINKY` (index + pinky)
2. **State & Timing Logic (Generating Final Morse Labels):**
   - `PINCH_THUMB` released under $0.4\text{s}$ $\rightarrow$ Emits **`DOT`**.
   - `PINCH_THUMB` held $\ge 0.4\text{s}$ $\rightarrow$ Emits **`DASH`**.
   - `PINCH_RING` tapped once $\rightarrow$ Emits **`SPACE`**; tapped twice quickly $\rightarrow$ Emits **`CONFIRM`**.
   - `PINCH_PINKY` $\rightarrow$ Emits **`DELETE`**.

### 3.3. Confusable Labels Analysis
- **Most confusable pair:** Ring-finger pinch (`PINCH_RING`) vs. Pinky pinch (`PINCH_PINKY`).
  - *Cause:* The ring and pinky fingers are anatomically adjacent. When the hand is angled towards the camera, the ring finger can occlude the pinky, causing landmark coordinates to overlap.
  - *Solution:* Collect diverse samples at various camera angles ($15^\circ - 30^\circ$) and include relative fingertip distances as features.

### 3.4. Session-Wise Data Splitting (70% Train - 15% Val - 15% Test)
- Avoid random frame-level splitting because consecutive video frames are near-duplicates.
- **Rule:** Entire recording sessions must be assigned to Train, Val, or Test as complete blocks to evaluate true model generalization on unseen recordings or users.

---

## Chapter 4: Data Preprocessing & Machine Learning Models

### 4.1. Why 21 Landmarks Instead of Full Images?
- Feeding entire webcam images into deep neural networks (CNNs) requires heavy compute, graphics cards (GPUs), and suffers from background clutter.
- **Our Approach:** MediaPipe isolates the 21 hand landmarks first. We feed only **63 numerical coordinates** into lightweight machine learning classifiers.
- **Benefits:** Small model file size ($< 5\text{ MB}$), trains in seconds, runs at $30+\text{ FPS}$ on any standard laptop CPU.

### 4.2. Feature Normalization (63 Features)
1. **Translation:** Subtract wrist coordinates $(x_0, y_0, z_0)$ from all points so hand position on screen does not affect prediction.
2. **Scale:** Divide relative coordinates by the maximum wrist-to-finger distance so distance from camera (near vs. far) is normalized to a $[0, 1]$ range.
3. **Handedness:** Mirror the horizontal axis for left hands so left and right hands share the same coordinate representation.

### 4.3. Machine Learning Algorithms
1. **$K$-Nearest Neighbors ($K$-NN):** Classifies poses based on majority vote of the $K$ closest training examples in Euclidean space. Intuitive and requires no complex training.
2. **Support Vector Machine (SVM):** Finds the optimal decision boundary separating gesture clusters with an RBF kernel. Fast inference ($\approx 1\text{ ms/frame}$) and high accuracy.
3. **Random Forest:** Ensemble of decision trees asking sequential threshold questions, voting on the final label. Highly resilient to overfitting.

---

## Chapter 5: Timing Logic, State Machine (FSM) & Decoding

### 5.1. Anti-Repeat Debouncing Rule
At 30 FPS, holding a gesture for 1 second without debouncing would trigger 30 repeated inputs.
- **Mandatory Rule:** Each physical gesture registers **exactly once**.
- The state machine immediately enters a `LOCKED` state after triggering a command and ignores further input until the user returns their hand to the open-hand `IDLE` pose.

### 5.2. Example Walkthrough (Inputting 'S' = `...`)
1. Position hand in open `IDLE` pose.
2. Quick pinch index and thumb $\rightarrow$ Buffer displays `.`. Return to `IDLE`.
3. Quick pinch index and thumb $\rightarrow$ Buffer displays `..`. Return to `IDLE`.
4. Quick pinch index and thumb $\rightarrow$ Buffer displays `...`. Return to `IDLE`.
5. Double-tap index and ring finger (`CONFIRM`) $\rightarrow$ System decodes `...` to **`S`**, appends **`S`** to text output, and clears the Morse buffer.

---

## Chapter 6: Model & System Evaluation

- **Offline Classifier Metrics:** Accuracy, Macro-Precision, Macro-Recall, Macro-F1, Inference Latency (ms), and Confusion Matrix.
- **Live System Evaluation:** Real-time FPS ($\ge 25\text{ FPS}$), zero unwanted duplicate inputs during sustained holds, and Character/Word Accuracy on benchmark words (`SOS`, `HELLO`).

---

## Chapter 7: Project Structure, Installation & Usage

```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run unit tests
python3 -m unittest discover tests -v

# 4. Record training data
python3 src/data_collection.py --task morse --subject user_01 --session s01

# 5. Preprocess and split data
python3 src/preprocessing.py --task morse

# 6. Train and tune models (KNN, SVM, Random Forest)
python3 src/train_models.py --task morse

# 7. Evaluate on test set and plot confusion matrices
python3 src/evaluate.py --task morse
```

---

## Chapter 8: Report Outline & Deliverables Checklist

### 8.1. Academic Report Outline (6 Chapters)
1. **Chapter 1 – Introduction:** Motivation, goals, and Morse input scope.
2. **Chapter 2 – Theoretical Background:** MediaPipe 21 landmarks, KNN, SVM, Random Forest principles, Morse standards.
3. **Chapter 3 – Data Collection & Labeling:** 5 control labels, collection protocol, cleaning, and session-wise splitting.
4. **Chapter 4 – System Design:** Feature extraction pipeline, FSM architecture, debouncing, and Morse decoding logic.
5. **Chapter 5 – Experimental Results:** Comparison table (KNN vs. SVM vs. RF), confusion matrix analysis, and real-world live testing.
6. **Chapter 6 – Conclusion & Future Work:** Summary of achievements, limitations, and future extensions.

### 8.2. Deliverables Checklist
- [ ] **1. Dataset:** Annotated landmark CSVs with documentation in `data/dataset_description.md`.
- [ ] **2. Clean Codebase:** Modules for collection, preprocessing, training, evaluation, and unit tests.
- [ ] **3. Trained Models:** Persisted `.pkl` models and configuration file in `models/`.
- [ ] **4. Experimental Results:** Metric tables and confusion matrix figures in `reports/figures/`.
- [ ] **5. Report & Presentation:** Complete academic report and slide deck.
