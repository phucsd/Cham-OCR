# Báo Cáo Đo Lường Sai Số (CER / WER) & Kịch Bản Bổ Sung Dữ Liệu Huấn Luyện OCR Chăm

Tài liệu này đánh giá chi tiết thực nghiệm nhận diện văn bản Chăm dạng khổ thơ bằng mô hình **v23** và thuật toán phân đoạn **DBNet**, đồng thời đề xuất giải pháp kỹ thuật để tinh chỉnh mô hình tiếp theo.

---

## 1. Tổng Quan Thực Nghiệm

- **Đối tượng**: Văn bản thơ cổ chữ Chăm gồm 18 khổ thơ (tổng cộng 34 dòng chữ thực tế).
- **Thuật toán phân đoạn**: PaddleOCR DBNet (`ch_PP-OCRv4_det_infer`).
- **Mô hình OCR**: `rec_cham_inference_v23` (PP-OCRv4 Rec Mobile CTC).
- **Môi trường chạy**: CPU Inference cục bộ.

---

## 2. Đo Lường Định Lượng (Metrics)

| Tiêu chí | Giá trị | Ghi chú |
|---|---|---|
| **Tổng số dòng thực tế** | **34** | Phát hiện đầy đủ 34/34 dòng |
| **Tổng số ký tự Ground Truth** | **1,311** | Bao gồm cả khoảng trắng và dấu ngắt câu |
| **Tổng số từ Ground Truth** | **281** | |
| **Khoảng cách chỉnh sửa ký tự (Levenshtein)** | **158** | |
| **Tỷ lệ lỗi ký tự tổng thể (Overall CER)** | **12.05%** | Chịu ảnh hưởng lớn bởi số thứ tự khổ thơ |
| **Tỷ lệ lỗi từ tổng thể (Overall WER)** | **40.57%** | |
| **Tỷ lệ lỗi ký tự phần Thân Chữ (Body Text CER)** | **10.34%** | Khi loại trừ các số thứ tự khổ thơ Chăm |
| **Độ chính xác nhận diện từ vựng cốt lõi** | **~90%** | Các phụ âm và từ ghép Chăm hầu như nhận diện chính xác |

---

## 3. Phân Tích Chi Tiết Các Nhóm Lỗi (Error Breakdown)

### 3.1. Lỗi hệ thống: Số thứ tự khổ thơ Chăm (`꩑꩞` đến `꩑꩘꩞`) — Tỷ lệ sai 100% (18/18)

Mặc dù từ điển `cham_dict_v23.txt` có chứa đầy đủ các ký tự số Chăm (`꩐` - `꩙`) và dấu ngắt đoạn `꩞`, nhưng do mô hình v23 chưa được học cấu trúc số thứ tự khổ thơ đứng đầu dòng, CTC Decoder đã ép xác suất dự đoán sang các ký tự chữ cái tương đồng về mặt thị giác:

| Khổ | Ký tự gốc | Mã Unicode gốc | Kết quả OCR v23 | Nguyên nhân hình học & CTC Decoder |
|:---:|:---:|:---:|:---:|:---|
| **1** | `꩑꩞` | U+AA51 + U+AA5E | `ꨩꩌ` | Nét vòng cung của `꩑` bị nhầm với `ꨩ`, dấu `꩞` bị nhầm với chấm `ꩌ` |
| **2** | `꩒꩞` | U+AA52 + U+AA5E | `ꨝꨮ` | Thân số `꩒` có nét uốn ngang nhầm với phụ âm `ꨝ` |
| **3** | `꩓꩞` | U+AA53 + U+AA5E | `ꨁꨩꩀ` | Hai đường cong tròn của số `꩓` nhầm với chữ `ꨁ` |
| **4** | `꩔꩞` | U+AA54 + U+AA5E | `ꨤꩃ ` | Cấu trúc nét đứng và móc nhầm với phụ âm `ꨤ` |
| **5** | `꩕꩞` | U+AA55 + U+AA5E | `ꨅꩀ` | Đường nét tròn uốn nhầm với nguyên âm độc lập `ꨅ` |
| **6** | `꩖꩞` | U+AA56 + U+AA5E | `ꨣ` | Nhầm với phụ âm `ꨣ` |
| **7** | `꩗꩞` | U+AA57 + U+AA5E | `꩜` | Nét xoắn của `꩗` nhầm với ký hiệu xoắn ốc Spiral `꩜` |
| **8** | `꩘꩞` | U+AA58 + U+AA5E | (Khoảng trắng) | Tín hiệu CTC quá phân tán, bị khử thành blank |
| **9** | `꩙꩞` | U+AA59 + U+AA5E | `ꨩ` | Nét móc đuôi nhầm với nguyên âm `ꨩ` |
| **10** | `꩑꩐꩞` | U+AA51 + U+AA50 + U+AA5E | `ꩍ꩐ꨩꩀ` | Nhận diện đúng số `꩐` (0), nhưng `꩑` và `꩞` bị gán nhầm |
| **11** | `꩑꩑꩞` | U+AA51 + U+AA51 + U+AA5E | `ꩍ ꩀ` | Hai số `꩑` bị ép về dấu phụ |
| **12** | `꩑꩒꩞` | U+AA51 + U+AA52 + U+AA5E | `ꩍꨝꨮ ꩇ` | Nhận nhầm thành cụm phụ âm `ꨝ` |
| **13** | `꩑꩓꩞` | U+AA51 + U+AA53 + U+AA5E | `ꩍꨄꨩ` | Nhận nhầm thành phụ âm `ꨄ` |
| **14** | `꩑꩔꩞` | U+AA51 + U+AA54 + U+AA5E | `ꨤ` | Nhận nhầm thành phụ âm `ꨤ` |
| **15** | `꩑꩕꩞` | U+AA51 + U+AA55 + U+AA5E | (Gộp vào `ꨞꨯꨚꨓꨪ꩞`) | Bị nuốt vào từ đầu tiên |
| **16** | `꩑꩖꩞` | U+AA51 + U+AA56 + U+AA5E | `ꨣꨩ` | Nhầm thành `ꨣ` |
| **17** | `꩑꩗꩞` | U+AA51 + U+AA57 + U+AA5E | `꩗` | Nhận diện được số `꩗` nhưng mất `꩑` và `꩞` |
| **18** | `꩑꩘꩞` | U+AA51 + U+AA58 + U+AA5E | `ꩄ` | Nhầm thành dấu `ꩄ` |

### 3.2. Lỗi nhầm lẫn dấu phụ (Diacritics Confusion)
- **Cặp `ꨲ` (U+AA32 Cham Vowel Sign Au) vs `ꨶ` (U+AA36 Cham Vowel Sign O)**:
  - `ꨀꨲꩆ` bị nhận diện thành `ꨀꨶꩆ` (sai 100% khi gặp `ꨲ`).
  - `ꨓꨆꨴꨲꨩ` bị nhận diện thành `ꨓꨆꨴꨶꨩ`.
  - *Lý do*: Hai dấu này đều nằm dưới chân phụ âm, nét vẽ chỉ khác nhau góc lượn nhọn và móc khép kín.
- **Dấu tổ hợp 3 tầng**:
  - `ꨣꨳꨪꩌ` (phụ âm `ꨣ` + dấu dưới `ꨳ` + dấu trên `ꨪ` + chấm `ꩌ`) bị mô hình rút gọn thành `ꨣꨳꨬ`.

### 3.3. Lỗi dấu ngắt câu cuối dòng (`꩝` vs `꩝꩝`)
- Trong văn bản gốc, các khổ thơ thường kết thúc bằng dấu Double Danda `꩝꩝` (hai vạch đứng).
- Kết quả OCR ở 14/18 khổ thơ chỉ nhận được 1 dấu Single Danda `꩝`.
- *Lý do*: Khoảng cách giữa 2 vạch đứng trong font chữ rất hẹp (khoảng 2-3px), CTC Decoder với bước trượt stride 4x theo chiều ngang đã gộp cả 2 nét thành 1 activation peak duy nhất.

---

## 4. Kế Hoạch & Kịch Bản Bổ Sung Dữ Liệu Huấn Luyện (Synthetic Data Recipe)

Để giải quyết triệt để các vấn đề trên cho mô hình **v24**, cần cập nhật quy trình sinh dữ liệu tại [`ocr-training/scripts/generate_data.py`](file:///e:/Phuc's%20Data/Github/Cham-OCR/ocr-training/scripts/generate_data.py):

### 4.1. Bổ sung Template Số thứ tự khổ thơ vào Generator
Thêm logic sinh dữ liệu chuyên biệt vào hàm sinh dòng chữ:
```python
CHAM_DIGITS = "꩐꩑꩒꩓꩔꩕꩖꩗꩘꩙"
CHAM_SECTION = "꩞"
CHAM_DANDAS = ["꩝", "꩝꩝"]

def generate_stanza_line(tokens):
    # Sinh ngẫu nhiên số thứ tự khổ thơ từ 1 đến 50
    num_val = random.randint(1, 50)
    if num_val < 10:
        cham_num_str = CHAM_DIGITS[num_val]
    else:
        cham_num_str = CHAM_DIGITS[num_val // 10] + CHAM_DIGITS[num_val % 10]
        
    stanza_prefix = f"{cham_num_str}{CHAM_SECTION} "
    
    # Ghép 3-6 từ Chăm
    words = [random.choice(tokens) for _ in range(random.randint(3, 6))]
    text = stanza_prefix + " ".join(words) + f" {random.choice(CHAM_DANDAS)}"
    return text
```

### 4.2. Tập Hard Examples cân bằng dấu phụ
Sinh 5,000 ảnh tập trung vào:
1. **Các mẫu chứa `ꨲ` (Vowel Sign Au)** kết hợp với các phụ âm `ꨀ`, `ꨓ`, `ꨚ`, `ꨆ`.
2. **Các mẫu chứa Double Danda `꩝꩝`** với các cự ly khoảng cách biến thiên (từ 2px đến 8px) và độ mờ gaussian để mô hình học cách phân biệt rõ 2 vạch đứng.
3. **Các số Chăm đứng độc lập và đứng ghép đôi** (`꩑꩞`, `꩒꩞`,..., `꩑꩐꩞`,..., `꩑꩘꩞`).

### 4.3. Quy trình Huấn Luyện trên Kaggle GPU T4x2
Tuân thủ nghiêm ngặt quy chuẩn của dự án trong `.agents/AGENTS.md`:
1. Sử dụng tài khoản Kaggle của dự án:
   ```python
   os.environ["KAGGLE_USERNAME"] = "gustavnguyen"
   os.environ["KAGGLE_KEY"] = "6bf56db7e5c0fa7895d157167961d92b"
   ```
2. Khởi chạy với GPU T4x2 và lệnh phân tán đa GPU:
   ```bash
   python3 -m paddle.distributed.launch --gpus '0,1' tools/train.py -c configs/rec/rec_cham_v24.yml
   ```
3. Load pre-trained weights từ checkpoint `v23` và fine-tune với learning rate nhỏ (`1e-4`) trong 30-50 epochs để giữ nguyên khả năng nhận diện thân chữ cốt lõi và cập nhật trọng số cho các cụm số thứ tự mới.
