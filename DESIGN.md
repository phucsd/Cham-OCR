---
version: alpha
name: Cham OCR Review Studio Warm Light Theme
colors:
  primary: "#2B2118"
  secondary: "#6F6257"
  tertiary: "#9A8D80"
  bg_app: "#FAF7F2"
  bg_canvas: "#F7F1EA"
  bg_panel: "#FFFCF7"
  bg_panel_soft: "#FBF5EE"
  bg_elevated: "#FFFFFF"
  border_subtle: "#E8DED2"
  border_medium: "#D8C9BA"
  border_strong: "#BFAE9D"
  accent: "#C96442"
  accent_hover: "#B55336"
  accent_soft: "#FFF0E8"
  accent_soft_2: "#FBE1D4"
  selection_bg: "#FFF0E8"
  selection_border: "#D97757"
  confidence_high: "#2F7D5C"
  confidence_high_bg: "#EAF6EF"
  confidence_medium: "#B7791F"
  confidence_medium_bg: "#FFF4D8"
  confidence_low: "#C4513B"
  confidence_low_bg: "#FDE8E2"
typography:
  body:
    fontFamily: "Google Sans Flex, Google Sans, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif"
    fontSize: "14px"
    fontWeight: "400"
  heading:
    fontFamily: "Google Sans Flex, Google Sans, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif"
    fontSize: "20px"
    fontWeight: "700"
spacing:
  xs: "4px"
  sm: "8px"
  md: "16px"
  lg: "24px"
  xl: "32px"
rounded:
  sm: "8px"
  md: "12px"
  lg: "18px"
  xl: "24px"
---

# Design System - Claude-inspired Warm Light Theme

## Overview
Giao diện của Cham OCR Review Studio được thiết kế theo phong cách ấm áp, tinh tế và thanh lịch lấy cảm hứng từ Claude. Tông màu ngà ấm áp kết hợp với các bề mặt mềm mịn, chữ màu nâu sẫm, viền mảnh tự nhiên mang lại cảm giác thân thiện giống như sách giấy và tài liệu thực thụ. Thiết kế tập trung vào tính rõ ràng, trải nghiệm đọc lâu không mỏi mắt, và không gian làm việc chuyên nghiệp.

## Colors
- **App background**: `--bg-app` (#FAF7F2) - Màu nền chính của ứng dụng.
- **Canvas background**: `--bg-canvas` (#F7F1EA) - Màu nền của Document Viewer.
- **Panel background**: `--bg-panel` (#FFFCF7) - Bề mặt các bảng điều khiển và editor.
- **Text Primary**: `--text-primary` (#2B2118) - Chữ màu nâu sẫm sần, độ tương phản cao, dễ đọc.
- **Accent terracotta**: `--accent` (#C96442) - Màu nhấn đất nung/coral ấm, dùng cho các nút bấm chính, các trạng thái active và highlight bbox.

## Typography
- **Google Sans Flex**: Sử dụng font chữ hiện đại, tối giản và sắc nét đặc trưng của Google cho toàn bộ giao diện (logo, tiêu đề, và nội dung văn bản). Font chữ này hỗ trợ các trục biến đổi linh hoạt mang lại trải nghiệm đọc rất mượt mà và trực quan trên các màn hình hiển thị.

## Layout & Spacing
Bố cục 3 cột rõ ràng ngăn nắp, khoảng cách lề vừa phải để người dùng tập trung tối đa vào tài liệu và các dòng hiệu chỉnh.

## Elevation
Sử dụng bóng đổ rất nhẹ và mềm (`rgba(43, 33, 24, 0.06)`) kết hợp với viền mảnh để phân tách chiều sâu thay vì dùng các hiệu ứng nổi bật quá mức.

## Shapes
Sử dụng các góc bo mềm mại ở mức vừa phải (8px đến 12px) để tạo cảm giác thân thiện nhưng vẫn giữ được sự chuyên nghiệp, ngăn nắp.

## Do's and Don'ts
- **Nên**: Sử dụng màu nền ngà ấm cho các khung đọc sách và hiệu chỉnh.
- **Nên**: Giữ viền mảnh màu cát sẫm (#E8DED2) để phân định các vùng làm việc rõ ràng.
- **Không nên**: Sử dụng màu đen thuần (#000) hoặc xám lạnh cho chữ và nền.
- **Không nên**: Sử dụng màu tím hoặc xanh neon quá chói làm xao nhãng người dùng.
