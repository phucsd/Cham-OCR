# Rules for Cham-OCR Project

- **Quy tắc làm việc**: Khi có bất kỳ điểm nào chưa rõ ràng hoặc thiếu thông tin cần thiết để thực hiện công việc, Agent bắt buộc phải dừng lại và đặt câu hỏi làm rõ với người dùng ngay lập tức, tránh tự ý giả định.
- **Quy tắc tối ưu hóa GPU trên Kaggle**:
  - **Tài khoản Kaggle riêng cho dự án**: Luôn sử dụng tài khoản Kaggle của dự án này: `username: "gustavnguyen"`, `key: "6bf56db7e5c0fa7895d157167961d92b"`. Khi gọi Kaggle API hoặc chạy các kịch bản tương tác với Kaggle, luôn đảm bảo gán `os.environ["KAGGLE_USERNAME"] = "gustavnguyen"` và `os.environ["KAGGLE_KEY"] = "6bf56db7e5c0fa7895d157167961d92b"` trước khi xác thực để không bị xung đột với tài khoản khác trên máy.
  - Khi khởi chạy bất kỳ tác vụ huấn luyện nào trên Kaggle, luôn luôn chỉ định bộ tăng tốc GPU tối ưu (`--accelerator NvidiaTeslaT4` để cấp phát GPU T4x2).
  - Viết mã nguồn huấn luyện thích ứng tự động (Adaptive Multi-GPU) để truy vấn số lượng GPU khả dụng bằng `paddle.device.cuda.device_count()`.
  - Nếu phát hiện số GPU > 1, bắt buộc phải sử dụng lệnh chạy phân tán song song `python3 -m paddle.distributed.launch --gpus '0,1,...' tools/train.py` nhằm tối đa hóa tốc độ huấn luyện (nhanh gấp đôi) để tiết kiệm thời gian thực thi và hạn ngạch quota của người dùng.

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
  - **Mẫu số thứ tự khổ thơ Chăm**: Bắt buộc phải đưa cấu trúc `{cham_digits}{cham_section_mark} {cham_text}` từ 1 đến 99 (ví dụ `꩑꩞ ...`, `꩑꩐꩞ ...`, `꩔꩓꩞ ...`, `꩕꩗꩞ ...`) vào generator sinh dữ liệu tổng hợp (`scripts/generate_data.py`), tránh việc CTC Decoder ép nhầm số thứ tự thành phụ âm hoặc chữ tương đồng (`꩔` -> `ꨤ`, `꩕` -> `ꨅ`/`ꨂ`, `꩑꩞` -> `ꨩꩌ`, `꩒꩞` -> `ꨝꨮ`).
  - **Cân bằng cặp dấu phụ dễ nhầm lẫn**: Sinh tối thiểu 5,000 mẫu hard-examples cho cặp dấu dưới chân `ꨲ` (Vowel Sign Au, U+AA32) và `ꨶ` (Vowel Sign O, U+AA36) với các phụ âm `ꨀ`, `ꨓ`, `ꨚ`, `ꨆ` để chống thiên kiến nhầm `ꨲ` thành `ꨶ`.
  - **Khoảng cách Double Danda (`꩝꩝`)**: Phải sinh các mẫu có khoảng cách biến thiên giữa 2 nét gạch đứng từ 2px đến 8px kèm nhiễu mờ để CTC không bị gộp 2 ký tự thành 1 (`꩝`).
  - **Tổ hợp dấu phụ đa tầng**: Tăng cường các mẫu kết hợp đồng thời dấu phụ dưới (`ꨳ`) và dấu phụ trên (`ꨪ`, `ꩌ`) như `ꨣꨳꨪꩌ` để tránh bị rút gọn sai thành `ꨣꨳꨬ`.

- **Thông tin Tên miền Dịch vụ (OCR Studio Domain)**:
  - Tên miền chính thức của ứng dụng web Cham OCR Review Studio là `ocr.cham.asia` (được trỏ về hệ thống phục vụ trực tuyến). Mọi liên kết, tài liệu hướng dẫn và phản hồi liên quan đến OCR Studio cần luôn sử dụng tên miền chính: `https://ocr.cham.asia`.

- **Quy tắc sử dụng Agent Memory (Persistent Memory)**:
  - **Định danh dự án (Project Identifier)**: Luôn sử dụng canonical identifier `project: "cham-ocr"` cho mọi thao tác truy vấn và lưu trữ bộ nhớ của dự án này.
  - **Cơ chế tự động ghi nhớ (Autonomous Auto-Memory Triggers)**: Agent tuyệt đối **KHÔNG ĐƯỢC CHỜ** người dùng nhắc nhở hay gõ lệnh `/remember`, mà phải **TỰ ĐỘNG CHỦ ĐỘNG GỌI `memory_save`** ngay trong lượt phản hồi khi xảy ra các sự kiện sau:
    1. **Tự động lưu sau mỗi lần code/refactor quan trọng (`type: "architecture"` hoặc `"workflow"`)**: Sau khi phát triển tính năng mới, tối ưu hóa thuật toán (như Line Segmentation, DBNet, CTC Decoder, Logical Order Normalization), sửa đổi API backend hoặc UI Studio, Agent bắt buộc phải tự động ghi nhớ tóm tắt thay đổi mã nguồn, lý do kỹ thuật và các tệp liên quan.
    2. **Tự động lưu sau mỗi lần Benchmark & Đánh giá mô hình (`type: "pattern"`)**: Sau khi chạy test hoặc đánh giá chất lượng mô hình (v23, v24, v25...), Agent bắt buộc phải tự động lưu chi tiết kết quả định lượng: chỉ số CER, WER, tỷ lệ nhận diện số khổ thơ Chăm, các cặp ký tự/dấu phụ bị nhầm lẫn (như `ꨲ` vs `ꨶ`, `꩝꩝` vs `꩝`), độ bền vững trước nhiễu/blur để làm dữ liệu định hướng cho các phiên huấn luyện tiếp theo.
    3. **Tự động lưu chỉ thị & sở thích người dùng (`type: "preference"` hoặc `"fact"`)**: Lưu ngay các quy chuẩn về tài khoản Kaggle, hạ tầng, đường dẫn, domain, cờ cấu hình môi trường hoặc yêu cầu nghiệp vụ mà người dùng đưa ra.
  - **Tự động truy hồi ngữ cảnh trước khi hành động (Context Recall First)**: Trước khi code thay đổi kiến trúc, sửa lỗi nhận diện chữ Chăm, tối ưu pipeline hoặc thiết lập huấn luyện Kaggle, Agent bắt buộc phải tự động gọi `memory_recall` hoặc `memory_smart_search` (với các từ khóa liên quan như `v24`, `benchmark`, `kaggle`, `segmentation`...) để kế thừa toàn bộ tri thức của các phiên làm việc trước mà không cần người dùng nhắc lại.

