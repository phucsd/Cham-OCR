# Hướng dẫn tải và cấu hình Font chữ tiếng Chăm (Akhar Thrah / Srak)

Để sinh dữ liệu tổng hợp tiếng Chăm chính xác và không bị lỗi hiển thị (tofu characters/ô vuông trắng), bạn cần chuẩn bị các font chữ tiếng Chăm chất lượng cao dạng `.ttf` hoặc `.otf` và lưu trữ chúng trong thư mục này (`data/fonts/`).

## 1. Các nguồn tải Font chữ tiếng Chăm uy tín

Dưới đây là một số nguồn tải font chữ tiếng Chăm Unicode phổ biến:

1. **EFEO (École française d'Extrême-Orient)**:
   - Viện Viễn Đông Bác cổ Pháp cung cấp các bộ font chữ Chăm cổ điển (Akhar Thrah) được chuẩn hóa theo các tài liệu nghiên cứu.
   - Bạn có thể tìm kiếm và tải font **CamEFEO** hoặc **EFEO Cham** từ các trang lưu trữ tài liệu nghiên cứu Chăm học.

2. **Dự án Cham Unicode (Kauthara / Inrasara / ChamToday)**:
   - Cộng đồng hỗ trợ tiếng Chăm cung cấp các bộ font Unicode như:
     - `Cham Thrah` (Dùng cho Chăm Đông / Akhar Thrah)
     - `Cham Gar` (Dành cho kiểu chữ viết tay hoặc chữ khắc)
     - `Cham Western` (Dành cho Chăm Tây / Akhar Srak)
   - Tải trực tiếp từ các website cộng đồng Chăm hoặc github repository chuyên về Cham Unicode.

3. **SeaSite (Northern Illinois University)**:
   - Trang học liệu tiếng Chăm của NIU chứa các font chữ Chăm Unicode miễn phí đi kèm với bộ gõ.
   - URL tham khảo: [NIU Cham Fonts](http://www.seasite.niu.edu/html_pages/Cham/cham_fonts.htm)

## 2. Cách thiết lập

1. Tải các file font chữ tiếng Chăm (định dạng `.ttf` hoặc `.otf`).
2. Sao chép các file này vào thư mục này:
   ```bash
   data/fonts/
   ```
3. Script sinh dữ liệu tự động (`scripts/generate_data.py`) sẽ quét thư mục này và sử dụng các font tìm thấy để vẽ chữ lên ảnh.
