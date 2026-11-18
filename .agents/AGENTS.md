# Rules for Cham-OCR Project

- **Quy tắc làm việc**: Khi có bất kỳ điểm nào chưa rõ ràng hoặc thiếu thông tin cần thiết để thực hiện công việc, Agent bắt buộc phải dừng lại và đặt câu hỏi làm rõ với người dùng ngay lập tức, tránh tự ý giả định.
- **Quy tắc tối ưu hóa GPU trên Kaggle**:
  - **Tài khoản Kaggle riêng cho dự án**: Luôn sử dụng tài khoản Kaggle của dự án này: `username: "gustavnguyen"`. Khi gọi Kaggle API hoặc chạy các kịch bản tương tác với Kaggle, luôn sử dụng module `kaggle_auth.init_kaggle_auth()` hoặc đọc an toàn từ biến môi trường `KAGGLE_KEY` / tệp `.env` (tuyệt đối không hardcode API key vào mã nguồn hoặc tài liệu markdown để tránh lộ bí mật khi push lên GitHub).
  - Khi khởi chạy bất kỳ tác vụ huấn luyện nào trên Kaggle, luôn luôn chỉ định bộ tăng tốc GPU tối ưu (`--accelerator NvidiaTeslaT4` để cấp phát GPU T4x2).
  - Viết mã nguồn huấn luyện thích ứng tự động (Adaptive Multi-GPU) để truy vấn số lượng GPU khả dụng bằng `paddle.device.cuda.device_count()`.
  - Nếu phát hiện số GPU > 1, bắt buộc phải sử dụng lệnh chạy phân tán song song `python3 -m paddle.distributed.launch --gpus '0,1,...' tools/train.py` nhằm tối đa hóa tốc độ huấn luyện (nhanh gấp đôi) để tiết kiệm thời gian thực thi và hạn ngạch quota của người dùng.
  - **Giới hạn thời gian 12 tiếng và Huấn luyện đa chặng (12-Hour Timeout & Multi-Stage Checkpoints)**:
    - Mỗi phiên chạy (kernel run) trên Kaggle bị giới hạn cứng tối đa là 12 tiếng. Với khối lượng huấn luyện dài hạn (như 40 epochs ~23 tiếng), **TUYỆT ĐỐI KHÔNG** được cấu hình chạy toàn bộ trong 1 phiên duy nhất để tránh bị Kaggle ngắt tiến trình giữa chừng và mất dữ liệu.
    - Bắt buộc phải chia tiến trình huấn luyện thành nhiều chặng an toàn (ví dụ: Chặng 1: Epochs 1–20 (~11.5h), Chặng 2: Epochs 21–40 (~11.5h); hoặc phương án đệm an toàn 3 chặng ~8h/chặng).
    - Cấu hình lưu checkpoint tự động sau mỗi epoch (`Global.save_epoch_step: 1`) và thiết lập cơ chế resume liền mạch qua `Global.checkpoints` trỏ vào checkpoint của chặng trước (thông qua Kaggle Kernel Output hoặc Kaggle Dataset) để tiếp tục tiến trình huấn luyện mà không làm gián đoạn learning rate schedule và optimizer state.
  - **Chống Treo Phân Tán Đa GPU (Distributed Multi-GPU Deadlock Prevention)**:
    - Khi huấn luyện phân tán đa GPU (`paddle.distributed.launch --gpus '0,1'`), **TUYỆT ĐỐI KHÔNG** kích hoạt đánh giá định kỳ giữa train loop (`eval_batch_step` định kỳ và `cal_metric_during_train: true`).
    - **Lý do kỹ thuật**: Trong PaddleOCR, chỉ Rank 0 (GPU 0) thực hiện hàm `eval()` trên tập validation trong khi Rank 1 (GPU 1) nhàn rỗi. Khi tập validation lớn (như 14,000 ảnh), thời gian đánh giá kéo dài khiến Rank 1 bị timeout đồng bộ (NCCL Collective Barrier Desync), gây **treo cứng phân tán vĩnh viễn (Deadlock)** làm cháy toàn bộ quota GPU của người dùng.
    - **Cấu hình chuẩn mực bắt buộc**:
      ```yaml
      Global:
        eval_batch_step:
        - 0
        - 9999999              # Tắt hoàn toàn eval định kỳ trong train loop
        cal_metric_during_train: false
        save_epoch_step: 1     # Bắt buộc lưu checkpoint sau mỗi epoch (iter_epoch_*, latest)
      ```
    - **Đánh giá an toàn sau huấn luyện**: Đánh giá validation metric (Acc, NED) chỉ được thực hiện sau khi vòng lặp huấn luyện hoàn tất, bằng cách chạy lệnh `tools/eval.py` độc lập trên **Single GPU** (`Global.checkpoints=.../latest`).
    - **Cơ chế Watchdog giám sát chống treo tự động (Autonomous Stall & Deadlock Detector)**: Mọi watchdog script theo dõi kernel Kaggle bắt buộc phải kiểm tra mốc thời gian cập nhật của `current_step` và luồng log:
      * Nếu kernel đang `RUNNING` nhưng quá **300 giây (5 phút)** không có step mới $\to$ Kích hoạt cảnh báo treo (`is_stalled: true`).
      * Nếu quá **600 giây (10 phút)** không có tiến triển $\to$ Tự động ngắt và báo lỗi deadlock, tuyệt đối không được im lặng chờ đợi lãng phí tài nguyên GPU.
  - **Phòng vệ Checkpoint & Tắt PIR khi Export Model (Checkpoint Guard & Disable PIR Export)**:
    - **Tắt PIR cho Paddle 2.6**: Khi xuất bản inference model (`tools/export_model.py`), PaddleOCR main branch mặc định bật `export_with_pir: True` (chỉ hỗ trợ Paddle 3.0+ và gây `AssertionError` trên Paddle 2.6). Bắt buộc phải cấu hình `export_with_pir: false` trong file config YAML hoặc truyền cờ `-o Global.export_with_pir=False`, đồng thời ưu tiên clone nhánh ổn định `git clone -b release/2.7 --depth 1 https://github.com/PaddlePaddle/PaddleOCR.git`.
    - **Nguyên tắc đóng gói bảo vệ Checkpoint trước tiên (Checkpoint First-Defense Guard)**: Ngay sau khi lệnh train loop kết thúc, tiến trình bắt buộc phải nén lưu toàn bộ thư mục checkpoint (`output/*/*.pdparams`, `.pdopt`, `.states`) thành file `.zip` đặt tại `/kaggle/working/` TRƯỚC KHI thực hiện bất kỳ lệnh `eval.py` hay `export_model.py` nào, để bảo toàn 100% thành quả huấn luyện ngay cả khi bước export gặp trục trặc.

- **Quy tắc tương thích NumPy 2.x**:
  - Khi viết các mã nguồn hoặc chạy ứng dụng cục bộ sử dụng môi trường Python mới (như Python 3.13 trở lên), luôn áp dụng đoạn mã monkeypatch tương thích ngược ở đầu tệp (trước khi import PaddleOCR hoặc `imgaug`):
    ```python
    import numpy as np
    if not hasattr(np, 'sctypes'):
        np.sctypes = {'int': [np.int8, np.int16, np.int32, np.int64], 'uint': [np.uint8, np.uint16, np.uint32, np.uint64], 'float': [np.float16, np.float32, np.float64], 'complex': [np.complex64, np.complex128], 'others': [bool, object, bytes, str]}
    if not hasattr(np, 'bool'): np.bool = bool
    if not hasattr(np, 'int'): np.int = int
    if not hasattr(np, 'float'): np.float = float
    if not hasattr(np, 'typeDict'): np.typeDict = {}
    ```

- **Quy tắc phân đoạn dòng chữ Chăm (Indic Line Segmentation)**:
  - **Tránh xóa nét chạm dính**: Khi lập mặt nạ xóa (erasing mask) vùng chữ của dòng lân cận, bắt buộc phải loại trừ các component thuộc về dòng hiện tại (`difference_update`) để tránh làm biến dạng chữ hoặc mất nét chạm dính.
  - **Bảo vệ chiều cao nguyên âm (Padding Safeguards)**: Vùng crop `safe` của dòng phải luôn được mở rộng đệm tối thiểu là `0.40 * median_line_height` lên phía trên và phía dưới so với dải chữ lõi (core text band), giới hạn an toàn cách biên dải chữ dòng lân cận tối thiểu 2px.
  - **Tối ưu hóa hiệu năng**: Luôn áp dụng bộ đệm khử trùng lặp theo tọa độ hộp cắt (`crop_box`) và dừng sớm (early stopping khi độ tự tin $\ge 0.90$) để giữ thời gian phản hồi OCR trên CPU dưới 15 giây.
  - **Cổng Giữ Nguyên Kết Quả Cũ (Legacy-First/No-Change Gate)**: Đối với các trang sạch có khoảng cách dòng an toàn (`gap >= 0.35 * median_line_height`), không có rủi ro dính nét biên (edge-ink risk) và độ tự tin OCR ban đầu chấp nhận được (`>= 0.75`), bắt buộc phải bỏ qua thuật toán multi-crop và giữ nguyên crop/kết quả cũ để tránh suy giảm độ chính xác hệ thống (regression).
  - **Bỏ Qua Rìa Trang Giấy Khi Tính Khoảng Cách**: Khi tính khoảng cách dòng (`gap_prev`, `gap_next`) để kiểm tra độ an toàn hoặc rủi ro chạm dòng, dòng đầu tiên và dòng cuối cùng của trang phải bỏ qua biên trang (không coi biên trang là vật cản để tránh tắt cổng Legacy-First nhầm lẫn).
  - **Tối Ưu Cờ Cần Xem Lại (Needs Review Optimization)**: Không được gán cờ `needs_review=True` đại trà chỉ vì các crop cho ra kết quả OCR khác nhau. Chỉ gán cờ này khi có rủi ro thực sự:
    - Phát hiện component không rõ ràng (ambiguous gap components) nằm gần nhiều dòng.
    - Kết quả OCR thay đổi các ký tự Chăm quan trọng (nguyên âm/dấu phụ trong dải `U+A800` đến `U+A82F`, `U+A840` đến `U+A87F`) so với kết quả cũ.
    - Độ tin cậy OCR giảm mạnh ($\ge 0.15$).
    - Phát hiện nguy cơ dính nét dòng lân cận trên các crop rescue/loose.

- **Quy tắc phát triển giao diện Web App**:
  - **Tách biệt mã nguồn (Source Code Separation)**: Đối với các ứng dụng Web có giao diện phức tạp, tuyệt đối không lưu trữ mã HTML, CSS và JavaScript khổng lồ trực tiếp trong chuỗi multiline Python (như `HTML_TEMPLATE = """..."""` trong `app.py`). Bắt buộc phải tách biệt mã frontend ra các tệp độc lập (ví dụ `index.html` hoặc thư mục `static/`, `templates/`) và cấu hình backend đọc động các tệp này từ đĩa để phục vụ HTTP request.
  - **Sử dụng Design Tokens (`DESIGN.md`)**: Luôn duy trì một tệp tin `DESIGN.md` ở thư mục gốc chứa các định nghĩa YAML frontmatter cho màu sắc, typography và corners. Khi có sự thay đổi về giao diện, bắt buộc phải cập nhật `DESIGN.md` và chạy lệnh linter để xác minh tính hợp lệ trước khi áp dụng vào mã nguồn:
    ```powershell
    npx -p @google/design.md designmd lint DESIGN.md
    ```
  - **Nhất quán Typography**: Sử dụng font chữ Google Sans Flex (`'Google Sans Flex', 'Google Sans', system-ui, -apple-system, sans-serif`) làm font-family mặc định cho toàn bộ giao diện của Studio để đảm bảo tính tối giản, sắc nét và hiện đại của sản phẩm.
  - **Kiểm tra cú pháp JavaScript tĩnh (JS Static Syntax Check)**: Khi thực hiện bất kỳ thay đổi nào trên mã JavaScript trong các tệp HTML (`index.html`) hoặc tệp `.js`, bắt buộc phải chạy lệnh kiểm tra cú pháp của Node.js (trích xuất mã ra tệp tạm và chạy `node --check temp.js`) trước khi bàn giao nhằm phát hiện và loại bỏ triệt để các lỗi `SyntaxError` (như khai báo trùng biến const, thiếu dấu đóng ngoặc,...) có khả năng làm tê liệt toàn bộ giao diện.
  - **Xử lý Clipboard Blob an toàn (Safe Clipboard Blob Handling)**: Khi bắt sự kiện dán ảnh Clipboard (`paste`), hãy truyền trực tiếp đối tượng file/blob trả về từ `getAsFile()` thay vì cố gắng khởi tạo `new File(...)` (tránh lỗi bảo mật hoặc thuộc tính read-only). Luôn áp dụng toán tử logic OR fallback `file.name || 'clipboard_image_...'` ở các hàm xử lý dữ liệu tiếp theo để xử lý trường hợp tệp dán không có tên.

- **Quy tắc Chuẩn hóa Văn bản Đầu ra (Logical Order Normalization)**:
  - Đối với các mô hình OCR đã hỗ trợ Logical Order mặc định (như V24, V26), **TUYỆT ĐỐI KHÔNG** sử dụng hàm `visual_to_unicode` để xử lý đầu ra, vì bộ chẻ cụm chữ cái trực quan (visual cluster parser) sẽ băm nát và làm hỏng chuỗi Logical. 
  - Thay vào đó, bắt buộc phải sử dụng hàm `normalize_unicode` (sử dụng `parse_unicode_clusters`) để tự động sắp xếp và sửa lỗi các "ảo giác" sai thứ tự cục bộ của mô hình (ví dụ: tự động nắn `ꨙꨯꨳꨮ` thành chuẩn `ꨙꨳꨯꨮ`).
  - Thứ tự chuẩn xác trong cụm Logical Order tiếng Chăm luôn là: `Base Consonant + Medials + Pre-Ra + Pre-Vowels + Others`.

- **Quy tắc Khoanh Vùng Hiển Thị Chữ (Bounding Box Rendering)**:
  - **Chiều dọc (Vertical)**: Không được dùng toạ độ `bbox` thô của Connected Components để vẽ khung đỏ/xanh vì đặc thù chữ Chăm có các dấu phụ vươn lên cao và kéo xuống thấp, khiến khung bị tràn lấn sang dòng lân cận. Bắt buộc phải ưu tiên dùng toạ độ dải chữ lõi `coords` để giới hạn chiều cao khung hiển thị vừa khít.
  - **Chiều ngang (Horizontal)**: Không được ước lượng độ rộng ký tự bằng phép chia trung bình. Bắt buộc phải sử dụng API `CanvasRenderingContext2D.measureText()` kết hợp với `Intl.Segmenter(granularity: 'grapheme')` để tính tỷ lệ bề ngang thực tế của từng cụm ký tự (grapheme clusters).

- **Quy tắc Chuẩn bị Dữ liệu Huấn luyện Phiên bản mới (Training Data Safeguards)**:
  - **Mẫu số thứ tự khổ thơ Chăm (1 đến 999) & Chống CTC Collapse**: 
    - Bắt buộc phải đưa cấu trúc `{cham_digits}{cham_section_mark} {cham_text}` từ **1 đến 999** (bao gồm số 3 chữ số như `꩑꩙꩗꩞` đến `꩒꩐꩙꩞`) vào generator sinh dữ liệu tổng hợp (`scripts/generate_data.py`).
    - **Chống CTC Collapse trên chữ số lặp**: Đối với các số có 2 chữ số giống nhau đứng liền kề (như `꩒꩐꩐꩞` số 200, `꩑꩑꩞`, `꩒꩒꩞`), bắt buộc phải sinh các mẫu có khoảng cách co giãn ngẫu nhiên từ 2px đến 8px giữa 2 chữ số cùng nhiễu đứt nét để CTC Decoder không bị nuốt/gộp 2 chữ số giống nhau thành 1 chữ số (`꩒꩐꩐꩞` -> `꩒꩐꩞`).
  - **Cân bằng các cặp ký tự và dấu phụ dễ nhầm lẫn (Hard-Mining)**:
    - Sinh tối thiểu 5,000 mẫu hard-examples cho cặp dấu dưới chân `ꨲ` (Vowel Sign Au, U+AA32) và `ꨶ` (Vowel Sign O, U+AA36) với các phụ âm `ꨀ`, `ꨓ`, `ꨚ`, `ꨆ`.
    - Sinh bổ sung các cặp phụ âm và dấu phụ hay bị nhầm lẫn phát hiện qua 4 bài test thực tế:
      * `ꨕ` (U+AA15) vs `ꨊ` (U+AA0A)
      * `ꨔ` (U+AA14) / `ꨝ` (U+AA1D) vs `ꨟ` (U+AA1F)
      * `ꨥ` (U+AA25) vs `ꨡ` (U+AA21)
      * `ꨈ` (U+AA08) vs `ꨓ` (U+AA13)
      * `ꨤ` (U+AA24) vs `ꨈ` (U+AA08)
      * `ꨪ` (U+AA2A) vs `ꨫ` (U+AA2B)
      * `ꩅ` (U+AA45) vs `ꩊ` (U+AA4A)
      * `꩝` (U+AA5D) vs `ꩃ` (U+AA43)
  - **Khoảng cách Double Danda (`꩝꩝`)**: Phải sinh các mẫu có khoảng cách biến thiên giữa 2 nét gạch đứng từ 2px đến 8px kèm nhiễu mờ để CTC không bị gộp 2 ký tự thành 1 (`꩝`).
  - **Tổ hợp dấu phụ đa tầng**: Tăng cường các mẫu kết hợp đồng thời dấu phụ dưới (`ꨳ`) và dấu phụ trên (`ꨪ`, `ꩌ`) như `ꨣꨳꨪꩌ` để tránh bị rút gọn sai thành `ꨣꨳꨬ`.
  - **Phân tầng từ điển (Vocabulary Layering) chống phân tán xác suất**: Khi mở rộng từ điển lên trên 160 token (thêm chữ Latinh cho văn bản song ngữ), phải áp dụng AutoRouting hoặc cơ chế phân tách không gian nhãn CTC để cô lập mô hình thuần Chăm truyền thống (103-108 token), tránh hiện tượng ảo giác chèn thêm các ký tự Latinh lạ (`t`, `g`, `m`, `i`, `ă`, `ua`) và thừa khoảng trắng khi nhận diện văn bản cổ mờ nhòe.

- **Thông tin Tên miền Dịch vụ (OCR Studio Domain)**:
  - Tên miền chính thức của ứng dụng web Cham OCR Review Studio là `ocr.cham.asia` (được trỏ về hệ thống phục vụ trực tuyến). Mọi liên kết, tài liệu hướng dẫn và phản hồi liên quan đến OCR Studio cần luôn sử dụng tên miền chính: `https://ocr.cham.asia`.

- **Quy tắc sử dụng Agent Memory (Persistent Memory)**:
  - **Định danh dự án (Project Identifier)**: Luôn sử dụng canonical identifier `project: "cham-ocr"` cho mọi thao tác truy vấn và lưu trữ bộ nhớ của dự án này.
  - **Cơ chế tự động ghi nhớ (Autonomous Auto-Memory Triggers)**: Agent tuyệt đối **KHÔNG ĐƯỢC CHỜ** người dùng nhắc nhở hay gõ lệnh `/remember`, mà phải **TỰ ĐỘNG CHỦ ĐỘNG GỌI `memory_save`** ngay trong lượt phản hồi khi xảy ra các sự kiện sau:
    1. **Tự động lưu sau mỗi lần code/refactor quan trọng (`type: "architecture"` hoặc `"workflow"`)**: Sau khi phát triển tính năng mới, tối ưu hóa thuật toán (như Line Segmentation, DBNet, CTC Decoder, Logical Order Normalization), sửa đổi API backend hoặc UI Studio, Agent bắt buộc phải tự động ghi nhớ tóm tắt thay đổi mã nguồn, lý do kỹ thuật và các tệp liên quan.
    2. **Tự động lưu sau mỗi lần Benchmark & Đánh giá mô hình (`type: "pattern"`)**: Sau khi chạy test hoặc đánh giá chất lượng mô hình (v23, v24, v25...), Agent bắt buộc phải tự động lưu chi tiết kết quả định lượng: chỉ số CER, WER, tỷ lệ nhận diện số khổ thơ Chăm, các cặp ký tự/dấu phụ bị nhầm lẫn (như `ꨲ` vs `ꨶ`, `꩝꩝` vs `꩝`), độ bền vững trước nhiễu/blur để làm dữ liệu định hướng cho các phiên huấn luyện tiếp theo.
    3. **Tự động lưu chỉ thị & sở thích người dùng (`type: "preference"` hoặc `"fact"`)**: Lưu ngay các quy chuẩn về tài khoản Kaggle, hạ tầng, đường dẫn, domain, cờ cấu hình môi trường hoặc yêu cầu nghiệp vụ mà người dùng đưa ra.
  - **Tự động truy hồi ngữ cảnh trước khi hành động (Context Recall First)**: Trước khi code thay đổi kiến trúc, sửa lỗi nhận diện chữ Chăm, tối ưu pipeline hoặc thiết lập huấn luyện Kaggle, Agent bắt buộc phải tự động gọi `memory_recall` hoặc `memory_smart_search` (với các từ khóa liên quan như `v24`, `benchmark`, `kaggle`, `segmentation`...) để kế thừa toàn bộ tri thức của các phiên làm việc trước mà không cần người dùng nhắc lại.

- **Quy tắc Kiểm chứng Số liệu Định lượng & Thời gian Huấn luyện (Ground-Truth Accounting)**:
  - **Tuyệt đối không ước lượng cảm tính hoặc dùng nhầm phiên thử nghiệm**: Khi báo cáo thời gian huấn luyện, đánh giá mô hình hoặc hạn ngạch GPU, Agent bắt buộc phải kiểm tra đối chiếu trực tiếp từ tệp nhật ký thực tế (`train.log`, Kaggle session log/API), không dùng số liệu phỏng đoán hoặc lấy nhầm từ các lần chạy thử nghiệm/dở dang (trial runs).
  - **Bóc tách chuẩn xác 3 thành phần thời gian**:
    1. `Train Loop Duration`: Thời gian chạy thuật toán huấn luyện thuần túy (tính từ timestamp batch đầu tiên đến batch cuối cùng trong `train.log`).
    2. `Pipeline / Container Overhead`: Thời gian chuẩn bị môi trường, cài đặt gói, sinh dữ liệu tổng hợp, giải nén tập validation đóng băng và đóng gói xuất bản checkpoint.
    3. `Total Kernel Wall-clock Time`: Tổng thời gian thực tế phiên chạy Kaggle chiếm dụng tài nguyên GPU (tính vào quota).

- **Quy tắc Phân định Chuẩn xác Mô hình V25 (Tuyệt đối không nhầm lẫn giữa các phiên làm việc)**:
  - **Bản thử nghiệm A100 cũ (Ngày 10/09/2026 - ĐÃ HỦY/LỖI THỜI)**: Phiên chạy thử 12 epochs trên Lightning AI Cloud NVIDIA A100 chỉ nhằm mục đích benchmark tốc độ phần cứng (A100 vs L40S vs L4). Checkpoint sơ bộ này có CER 51.73% và **KHÔNG ĐƯỢC COI LÀ MÔ HÌNH V25 CHÍNH THỨC**.
  - **Bản V25 SOTA Chính thức (Hoàn tất ngày 13/09/2026 trên Kaggle Dual Tesla T4x2)**: Đây là mô hình V25 **DUY NHẤT VÀ CHÍNH THỨC** của dự án:
    - Huấn luyện trọn vẹn **40 Epochs** (3 chặng, 87,480 steps) trên **Kaggle Dual Tesla T4x2** (tài khoản `gustavnguyen`, kernel `paddleocr-cham-v25-stage3`).
    - Kết quả kiểm định chuẩn trên 10,000 ảnh đóng băng (`cham_v25_val_freeze.zip`): **Sequence Accuracy 90.96%**, **Normalized Edit Distance 99.28% (CER 0.72%)**.
    - Đây là mô hình duy nhất được chuyển đổi sang `ocr-studio/data/output/rec_cham_inference_v25/` và triển khai trên dịch vụ thực tế `https://ocr.cham.asia`.
    - Khi bắt đầu bất kỳ phiên trò chuyện mới nào, Agent **TUYỆT ĐỐI KHÔNG ĐƯỢC NHẦM** với bản thử nghiệm 12 epoch A100 cũ. Mọi báo cáo về V25 bắt buộc phải căn cứ theo kết quả SOTA 40 Epochs Kaggle T4x2 này.

- **Quy tắc Độc Lập Môi Trường Đám Mây & Kiểm Định Nghiệm Thu (Cloud & Benchmark Integrity Rules)**:
  - **Đường dẫn Thư viện & Phông chữ Đa Nền tảng (Cross-Platform Path Resolution)**:
    - Trong mọi kịch bản kiểm định hoặc xuất bản mô hình (`run_v24_frozen_suite.py`, `select_best_checkpoint.py`), bắt buộc phải quét tự động các thư mục ứng viên: `/kaggle/working/PaddleOCR`, `STUDIO_DIR/PaddleOCR`, `PROJECT_ROOT/PaddleOCR`, `os.getcwd()/PaddleOCR` trước khi import `tools.infer`.
    - Khi render chữ Chăm (`get_cham_font`), bắt buộc phải quét cả `/kaggle/working/data/fonts/`, `/kaggle/working/fonts/`, `05_Dictionaries_and_Fonts/`, và quét đệ quy `/kaggle/input/**/NotoSansCham-*.ttf`. Tuyệt đối không fallback ngầm về phông hệ thống mặc định (`ImageFont.load_default()`) vì sẽ làm hỏng kết quả kiểm định chữ Chăm.
  - **Cấm Hard-code Cờ Nghiệm Thu (Dynamic Release Gates Only)**:
    - Tuyệt đối cấm gán cứng giá trị `valid_for_release = True` trong bất kỳ tệp báo cáo hoặc mã nguồn nào.
    - Cờ `valid_for_release` bắt buộc phải được tính toán động 100% bằng biểu thức logic kiểm tra toàn bộ các Release Gates:
      * Gate 1: Clean-Frozen Non-Regression ($\le \text{Baseline CER} + 0.20\%$).
      * Gate 2: 14 Mandatory Symbols Accuracy $\ge 99.0\%$ cho từng ký hiệu (số ký hiệu failing = 0).
      * Gate 3: Hyphen Multi-Context Regression $\ge 95.0\%$ (trên 7 nhóm ngữ cảnh).
      * Gate 4: Zero Blank Hallucination trên ảnh trắng và ảnh xám.
  - **Giới Hạn Kích Thước Kaggle Notebook (< 1 MB)**:
    - Khi tạo notebook Kaggle tự động bằng kịch bản, tuyệt đối không nhúng Base64 các tệp dữ liệu tĩnh lớn (fonts, test specs, checkpoints) trực tiếp vào chuỗi cell khiến dung lượng notebook vượt quá 1 MB (gây lỗi `413 Payload Too Large` của Kaggle API). Bắt buộc phải đóng gói các tài nguyên này vào Kaggle Dataset và nạp động qua `/kaggle/input`.

- **Quy tắc Kiểm Tra Tĩnh Tiền Đóng Gói & Đối Ứng Bộ Sinh (Pre-Flight Static Integrity & Generator Parity Rules)**:
  - **Kiểm Tra Phạm Vi & Cú Pháp Tĩnh Trước Khi Đóng Gói (Pre-Flight Scope & Syntax Check)**:
    - Tuyệt đối không nhúng bất kỳ mã Python nào vào notebook Kaggle khi chưa chạy kiểm tra tĩnh bằng `py_compile` và kiểm tra phạm vi biến (static scope check) để loại trừ triệt để lỗi `NameError` (như biến `score` chưa khai báo ngoài khối điều kiện).
    - Mọi kịch bản đóng gói (`prepare_v24_1_kaggle.py`) bắt buộc phải tích hợp bước dry-run/linter tự động trước khi cho phép lệnh `api.kernels_push` thực thi.
  - **Quy Chuẩn Đối Ứng Mẫu Kiểm Định Đóng Băng (Test-Spec Pattern Parity)**:
    - Mọi ký hiệu, dấu phụ và ngữ cảnh đặc thù có trong các bộ test đóng băng (`symbol_frozen_14.json`, `hyphen_frozen_multi_context.json`, `clean_frozen_200.json`) bắt buộc phải có hàm sinh chuyên biệt (targeted generator) chiếm tối thiểu 40%–60% tỷ trọng trong nhóm quota của ký hiệu đó.
    - Tuyệt đối không gộp các ký hiệu kiểm định quan trọng (`!`, `?`, `--`) vào nhóm replay tổng quát với quota thấp và thể thức sinh chung chung, nhằm tránh hiện tượng rớt Hard Gates do lệch phân phối mẫu so với bộ test.
  - **Cơ Chế Watchdog Nhận Biết Đa Pha (Phase-Aware Watchdog)**:
    - Kịch bản giám sát Kaggle (`kaggle_watchdog_v24_1.py`) bắt buộc phải phân định rõ từng pha thực thi: Pha 3 Sinh dữ liệu (`Rendering`, `Coverage`), Pha 4 Khởi tạo Dataloader (`Initialize indexs`), Pha 5 Train loop (`step:`, `epoch:`), Pha 6 Kiểm định Checkpoint (`Evaluating`, `CER`), Pha 7 Export Model.
    - Cấm đặt ngưỡng timeout cố định của train loop (300s/600s) cho pha sinh dữ liệu hoặc kiểm định, tránh gây báo động giả treo máy (false deadlock) trong khi hệ thống đang xử lý CPU/disk bình thường.

- **Quy tắc Tích hợp TypeSafe AI & JEV System One (TypeSafe & JEV Integration Rules)**:
  - **Cấu hình & Vị trí cài đặt**:
    - Project MCP server: [`.agents/mcp_config.json`](file:///E:/Phuc%27s%20Data/Github/Cham-OCR/.agents/mcp_config.json) trỏ vào `jev mcp`.
    - Project Skill: [`.agents/skills/typesafe-ai/SKILL.md`](file:///E:/Phuc%27s%20Data/Github/Cham-OCR/.agents/skills/typesafe-ai/SKILL.md).
    - Thư viện cầu nối cốt lõi: [`scripts/typesafe_cham_evaluator.py`](file:///E:/Phuc%27s%20Data/Github/Cham-OCR/scripts/typesafe_cham_evaluator.py) và bộ kiểm thử [`tests/test_typesafe_cham_evaluator.py`](file:///E:/Phuc%27s%20Data/Github/Cham-OCR/tests/test_typesafe_cham_evaluator.py).
  - **Ứng dụng thực tế trong vòng đời Cham-OCR**:
    1. **Đánh giá rủi ro & gán cờ `needs_review` thông minh (`ChamReviewJudge`)**:
       - Thay thế hoặc bổ trợ các luật ngưỡng cứng (hard-coded thresholds) dễ gây báo động giả bằng phán đoán xác suất nhị phân (`Noul`) và thang điểm rủi ro (`Score`).
       - Kích hoạt khi có sự sai biệt trên các cặp chữ cái tối thiểu (minimal pairs) hoặc dấu phụ quan trọng (như `ꨕ` vs `ꨊ`, `ꨲ` vs `ꨶ`, `ꨈ` vs `ꨓ`) và độ chồng lấn nét dòng lân cận cao.
    2. **Đánh giá tự động Release Gates cho Checkpoint (`ChamReleaseGateJudge`)**:
       - Sử dụng `Choice` (`ACCEPT`, `INVESTIGATE`, `ROLLBACK`) để đánh giá định lượng kết quả benchmark dựa trên 4 cổng nghiêm ngặt (Non-regression, 14 Mandatory symbols, Hyphen accuracy, Zero blank hallucination).
       - Ngăn chặn triệt để tình trạng đưa các checkpoint suy thoái hoặc ảo giác vào inference engine.
    3. **Định tuyến thể văn & chữ viết (Script & Dialect Routing)**:
       - Phân loại dòng văn bản để nạp đúng từ điển/mô hình tối ưu: Chăm truyền thống (Akhar Thrah 108 token), Song ngữ Chăm - Latinh, hoặc Chăm Tây (Cam Srak).

- **Quy tắc Tải Output Kaggle Chống Đứt Luồng, Fallback Trình Duyệt Trực Tiếp & Duy Trì Kaggle CLI Mới Nhất**:
  1. **Duy trì Kaggle CLI mới nhất**: Ngay khi thấy cảnh báo `outdated kaggle version`, bắt buộc nâng cấp ngay gói `kaggle` (và `requests` nếu có dependency warning) bằng `pip install --upgrade kaggle requests`.
  2. **Cấm gọi thô `api.kernels_output()` không lọc đường dẫn**: Tuyệt đối không dùng `api.kernels_output()` thô trên các kernel có thư mục `output/` lớn (~1.5 GB file trùng lặp dễ gây `IncompleteRead` và treo socket vô thời hạn). Bắt buộc dùng luồng tải phân mảnh `4 MB` (`ApiListKernelSessionOutputRequest` + `requests.get(..., stream=True, timeout=(20, 45))`), chỉ lọc đúng `checkpoints_*.zip`, `checkpoint_manifest_*.json` và lưu `resp.log` (`<slug>.log`), kèm bỏ qua thông minh nếu file nén hiện có đã khớp CRC & SHA-256 trong manifest.
  3. **Fallback Trình Duyệt Trực Tiếp Ngay Lập Tức (Chỉ thử API đúng 1 lần duy nhất)**: Không retry API lòng vòng; nếu lệnh tải qua Kaggle API gặp bất kỳ lỗi nào (`IncompleteRead`, timeout, `429`, `403`,...) ngay ở lần đầu tiên (`BROWSER_FALLBACK_REQUIRED`), bắt buộc **ngắt ngay tiến trình API và trực tiếp dùng trình duyệt tích hợp (`browser` subagent đã đăng nhập Kaggle)** truy cập thẳng trang kernel `https://www.kaggle.com/code/gustavnguyen/<slug>` để bấm tải `Download output` (`results.zip`) và `Download Logs` (`download.txt`) về máy, rồi nạp qua `--ingest-browser-download`.


