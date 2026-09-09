# Lộ Trình Huấn Luyện Mô Hình OCR Chăm V24 (Training Roadmap)

Tài liệu này ghi nhớ toàn bộ các nguyên nhân lỗi trên mô hình **v23** đã được phân tích từ thực nghiệm ảnh khổ thơ cổ và cung cấp cấu hình chuẩn bị cho phiên huấn luyện tiếp theo (**v24**).

---

## 1. Bốn Nguyên Nhân Cốt Lõi Cần Khắc Phục

1. **Lỗi nhận diện số thứ tự khổ thơ Chăm (`꩑꩞` - `꩑꩘꩞`)**:
   - *Hiện tượng*: Sai 100% (18/18 khổ thơ). Các cụm `꩑꩞`, `꩒꩞`, `꩓꩞`... bị CTC Decoder ép nhận diện thành các chữ cái Chăm tương đồng về hình thái (`ꨩꩌ`, `ꨝꨮ`, `ꨁꨩꩀ`, `ꨤꩃ`,...).
   - *Nguyên nhân*: Thiếu cấu trúc số thứ tự khổ thơ đứng đầu dòng trong tập dữ liệu tổng hợp (`scripts/generate_data.py`).
   - *Hành động khắc phục*: Bổ sung template `{cham_num}꩞ {cham_text}` với số chạy từ `꩑` đến `꩙꩙`.

2. **Lỗi nhầm lẫn cặp dấu phụ `ꨲ` (Au, U+AA32) thành `ꨶ` (O, U+AA36)**:
   - *Hiện tượng*: Sai 100% khi gặp dấu `ꨲ` (như `ꨀꨲꩆ` -> `ꨀꨶꩆ`, `ꨓꨆꨴꨲꨩ` -> `ꨓꨆꨴꨶꨩ`).
   - *Nguyên nhân*: Cả hai đều là dấu phụ nằm dưới chân phụ âm, nét vẽ chỉ khác nhau góc lượn nhọn và móc khép kín; tập huấn luyện thiếu các từ chứa `ꨲ`.
   - *Hành động khắc phục*: Sinh 5,000 mẫu hard-examples tập trung vào các phụ âm kết hợp `ꨲ` (`ꨀ`, `ꨓ`, `ꨚ`, `ꨆ`, `ꨟ`, `ꨣ`).

3. **Lỗi gộp dấu ngắt câu Double Danda `꩝꩝` thành Single Danda `꩝`**:
   - *Hiện tượng*: Ở 14/18 dòng kết thúc câu, dấu `꩝꩝` chỉ nhận được 1 dấu `꩝`.
   - *Nguyên nhân*: Khoảng cách giữa 2 vạch đứng trong font chữ rất hẹp (2-3px), bước trượt CTC stride 4x theo chiều ngang gộp cả 2 nét thành 1 activation peak.
   - *Hành động khắc phục*: Sinh các mẫu với khoảng cách ngẫu nhiên từ 2px đến 8px giữa 2 dấu `꩝` kèm augmentations (blur nhẹ, noise).

4. **Lỗi tổ hợp dấu phụ đa tầng (`ꨣꨳꨪꩌ` -> `ꨣꨳꨬ`)**:
   - *Hiện tượng*: Khi một phụ âm mang đồng thời cả dấu dưới (`ꨳ`) và dấu trên (`ꨪ`, `ꩌ`), mô hình có xu hướng đơn giản hóa thành một dấu phụ đơn lẻ `ꨬ`.
   - *Hành động khắc phục*: Tăng cường các mẫu từ Chăm có cấu trúc dấu phụ 3 tầng trong tập corpus và synthetic generator.

5. **Lỗi số thứ tự khổ thơ 2 chữ số (Khổ `꩔꩓꩞` - `꩕꩗꩞` / 43-57) [Phát hiện từ Case 2]**:
   - *Hiện tượng*: Số hàng chục `꩔` (digit 4) bị nhầm thành phụ âm `ꨤ` trong 100% các trường hợp (`꩔꩓꩞` -> `ꨤ...`, `꩔꩔꩞` -> `ꨤ ꩔`, `꩔꩕꩞` -> `ꨤ꩕`, `꩔꩗꩞` -> `ꨤ ꩗`, `꩔꩘꩞` -> `ꨤ ꩘`); số hàng chục `꩕` (digit 5) bị nhầm thành nguyên âm `ꨅ` hoặc `ꨂ` (`꩕꩐꩞` -> `ꨅ ꩐`, `꩕꩓꩞` -> `ꨂꨄ`, `꩕꩗꩞` -> `ꨅ꩗`).
   - *Đặc điểm đáng chú ý*: Các số hàng đơn vị đứng sau (`꩐`, `꩔`, `꩕`, `꩗`, `꩘`) mô hình vẫn nhận diện đúng! Điều này chứng minh mô hình có khả năng trích xuất đặc trưng của số Chăm, nhưng vì thiếu hoàn toàn dữ liệu ngữ cảnh số 2 chữ số đứng đầu dòng nên CTC Decoder tự động ép số đầu tiên thành phụ âm/nguyên âm để khớp với ngữ cảnh từ vựng thông thường.
   - *Hành động khắc phục*: Đảm bảo dải số sinh dữ liệu mở rộng từ 1 đến 99 (đặc biệt các số `꩔꩐` - `꩙꩙`).

6. **Lỗi rụng nguyên âm trước `ꨯ` (E-vowel, U+AA2F) và nguyên âm kép `ꨯꨱ` (Au) [Phát hiện từ Case 2]**:
   - *Hiện tượng*: Các từ như `ꨗꨯꨣꨚꨮꩅ` bị rụng mất `ꨯ` thành `ꨗꨣꨚꨮꩅ`, `ꨆꨴꨯꩅ` thành `ꨆꨴꩅ`, `ꨈꨪꨗꨯꨱꩃ` thành `ꨈꨪꨗꨮ`.
   - *Nguyên nhân*: Nét vẽ của nguyên âm trước `ꨯ` thanh mảnh và đứng trước phụ âm, ở các dòng chữ có chiều cao thấp (~30px) dễ bị lẫn vào viền crop hoặc bị bộ trích xuất CNN làm mờ.
   - *Hành động khắc phục*: Bổ sung biến đổi độ phân giải thấp (downsampling resize 0.6x - 0.9x) và làm mờ motion blur vào pipeline dữ liệu tổng hợp để mô hình nhạy bén hơn với nét thanh của `ꨯ`.

7. **Lỗi sụp đổ trước ảnh mờ rung và nhiễu hạt giấy cổ [Phát hiện từ 50 Bài Test Thực Nghiệm]**:
   - *Hiện tượng*: Motion Blur đẩy CER lên 51.1% - 64.4%; Nhiễu muối tiêu / thiếu sáng đẩy CER lên 50.6% - 67.6%, độ tự tin tụt rớt xuống 0.36; Vết loang mực / ố nước CER 53.3%.
   - *Nguyên nhân*: Mô hình nhầm các chấm nhiễu thành Visarga `ꩍ` hoặc Anusvara `ꩌ`, nhầm quầng ố thành chữ cái Chăm.
   - *Hành động khắc phục*: Tích hợp mạnh mẽ Data Augmentation (Random Motion Blur kernel 3x3 đến 7x7, Random Gaussian/Salt-and-pepper noise, quầng loang nước và nền giấy cổ giả lập).

8. **Lỗi góc nghiêng và phối cảnh 3D ($\pm 3^\circ \to \pm 6^\circ$) [Phát hiện từ 50 Bài Test Thực Nghiệm]**:
   - *Hiện tượng*: Đầu dòng bị sinh ra ký tự rác (`꩜`, `ꨣꨯꨱꩀ`, `꩗ꩀ`), CER trung bình 22.14%.
   - *Hành động khắc phục*: Tích hợp Random Rotation $\pm 5^\circ$ và Perspective Skew $\pm 4^\circ$ vào pipeline huấn luyện.

---

## 2. Kế Hoạch Chuẩn Bị Dữ Liệu V24

- **Quy mô tập dữ liệu mới**:
  - `train_synthetic`: ~150,000 dòng ảnh.
  - `val_synthetic`: ~15,000 dòng ảnh.
  - `hard_examples_v24`: 35,000 dòng ảnh đặc trị gồm:
    - 15,000 số thứ tự khổ thơ Chăm 1-99 (`{cham_num}꩞ {cham_text}`).
    - 5,000 cặp đối kháng `ꨲ` (U+AA32) vs `ꨶ` (U+AA36).
    - 5,000 Double Danda `꩝꩝` cự ly hẹp 1px - 8px.
    - 5,000 tổ hợp 3 tầng dấu phụ (`ꨣꨳꨪꩌ`...).
    - 5,000 mẫu ảnh suy thoái thực tế (Motion Blur, Dust/Noise, Parchment texture, Tilt/Perspective).
- **Từ điển**: Sử dụng từ điển chuẩn hóa `data/cham_dict_v23.txt` (83 ký tự, đã có đầy đủ số và dấu Chăm).

---

## 3. Cấu Hình Huấn Luyện Trên Kaggle GPU T4x2

Tuân thủ nghiêm ngặt quy định trong `.agents/AGENTS.md`:

```python
import os
os.environ["KAGGLE_USERNAME"] = "gustavnguyen"
os.environ["KAGGLE_KEY"] = "6bf56db7e5c0fa7895d157167961d92b"
```

- **Base checkpoint**: Khởi tạo trọng số từ checkpoint tốt nhất của `v23` (`data/output/rec_cham_inference_v23/`).
- **Accelerator**: GPU Tesla T4x2 (`--accelerator NvidiaTeslaT4`).
- **Lệnh chạy huấn luyện song song phân tán (Distributed Launch)**:
  ```bash
  python3 -m paddle.distributed.launch --gpus '0,1' tools/train.py -c configs/rec/rec_cham_v24.yml
  ```
- **Learning rate**: `1e-4` với cosine decay, batch size: `64` trên mỗi card (tổng batch: `128`), số epochs: 40.
