
# DỰ ÁN PHƯƠNG PHÁP SỐ – NHÓM 8

# 1. Giới thiệu

Dự án được thực hiện nhằm cài đặt, kiểm thử và đánh giá các phương pháp số sử dụng để giải gần đúng phương trình vi phân thường bằng ngôn ngữ lập trình Python.

Dự án tập trung vào phương pháp Runge-Kutta bậc 3 (RK3), Runge-Kutta bậc 4 (RK4) và các bài toán liên quan. Các thành viên phối hợp thực hiện mã nguồn, kiểm thử, thực nghiệm và xây dựng báo cáo.

# 2. Mục tiêu

- Cài đặt các phương pháp số bằng Python.
- Tổ chức mã nguồn thành các tệp riêng biệt để dễ đọc và dễ quản lý.
- Kiểm thử chương trình nhằm phát hiện lỗi và đánh giá kết quả.
- Thực nghiệm, so sánh và nhận xét về các phương pháp số.
- Tích hợp mã nguồn và tài liệu của các thành viên thành một sản phẩm thống nhất.
- Hoàn thiện báo cáo và slide thuyết trình.

# 3. Cấu trúc dự án

| Tệp | Chức năng |
| :--- | :--- |
| rk3.py | Cài đặt thuật toán Runge-Kutta bậc 3 |
| rk4.py | Cài đặt thuật toán Runge-Kutta bậc 4 |
| rk_table.py | Xuất bảng tính tay cho từng bước lặp |
| uocluong.py | Ước lượng bậc hội tụ và tìm bước lưới thỏa mãn dung sai |
| test_p1.py - test_p4.py | Các file kiểm thử tự động bằng thư viện pytest |
| thuc_nghiem.py | Chạy thực nghiệm, in bảng sai số và vẽ đồ thị |
| bai_toan_ung_dung_sv7.py | Giải bài toán vật rơi có lực cản không khí |
# 4. Phân công công việc

| Vai trò | Họ và Tên | MSSV | Nhiệm vụ | File phụ trách |
| :--- | :--- | :--- | :--- | :--- |
| SV1 | Nguyễn Thị Ngọc Anh | 243030509 | Cài đặt thuật toán RK3 | rk3.py |
| SV2 | Vũ Minh Ánh | 243030510 | Cài đặt thuật toán RK4 | rk4.py |
| SV3 | Lê Tường Vy | 243030562 | Xuất bảng tính tay | rk_table.py |
| SV4 | Phạm Khánh Huyền | 243030529 | Ước lượng bậc & Tìm bước h | uocluong.py |
| SV5 | Trương Gia Bảo | 243030511 | Bộ kiểm thử pytest toàn nhóm, slide | test_p1.py đến test_p4.py |
| SV6 | Nguyễn Công Kỳ | 243030533 | Thực nghiệm so sánh, đồ thị log-log | thuc_nghiem.py |
| SV7 | Kiều Minh Công | 243024441 | Bài toán ứng dụng (Vật rơi), thuyết trình | bai_toan_ung_dung_sv7.py |
| SV8 | Nguyễn Hữu Khoa | 243030532 |  Tạo repo, tích hợp, chuẩn hóa , readme, gộp báo cáo |  README.md |

## 5. Yêu cầu hệ thống và Cài đặt

Dự án yêu cầu Python 3.10 trở lên.

Danh sách thư viện bao gồm: numpy, matplotlib, pytest, scipy. Trong đó, thư viện scipy chỉ được sử dụng trong các file kiểm thử (test_p*.py) để đối chiếu kết quả, không sử dụng trong mã nguồn thuật toán chính.

# 6. Hướng dẫn chạy

Chạy kiểm thử tự động toàn bộ dự án:
pytest -v

Chạy phần thực nghiệm (xuất bảng số liệu và đồ thị):
python thuc_nghiem.py

Chạy bài toán ứng dụng vật rơi:
python bai_toan_ung_dung_sv7.py
