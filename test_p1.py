import numpy as np
import pytest
from scipy.integrate import solve_ivp

from rk3 import rk3


# ==============================================================================
# CÁC HÀM BÀI TOÁN MẪU
# ==============================================================================
# 1. Bài toán mẫu trong giáo trình (Bài 6.1)
def f_bai_6_1(x, y):
    return 2.0 - np.exp(-4.0 * x) - 2.0 * y

def exact_bai_6_1(x):
    return 1.0 + 0.5 * np.exp(-4.0 * x) - 0.5 * np.exp(-2.0 * x)

# 2. Bài toán KIỂM THỬ BẮT BUỘC (Theo ảnh đề bài của cô)
# y' = y - x^2 + 1, y(0) = 0.5 trên [0, 2]
def f_bat_buoc(x, y):
    return y - x**2 + 1.0

def exact_bat_buoc(x):
    return (x + 1.0) ** 2 - 0.5 * np.exp(x)

# 3. Bài toán TỰ CHỌN (Khác hoàn toàn 2 bài trên)
# y' = -2xy, y(0) = 1 trên [0, 1]. Nghiệm đúng: y = e^(-x^2)
def f_tu_chon(x, y):
    return -2.0 * x * y

def exact_tu_chon(x):
    return np.exp(-x**2)


# ==============================================================================
# KIỂM THỬ PHẦN 1 (RK3)
# ==============================================================================
def test_rk3_r7_1_vi_du_bai_giang():
    """[R7 - Ca 1]: Ví dụ trong bài giảng (Bài 6.1)."""
    print("--- [R7 - Ca 1: Vi du giao trinh cho RK3 (Bai 6.1)] ---")
    x, y = rk3(f_bai_6_1, 0.0, 0.5, 1.0, 5)
    y_true = exact_bai_6_1(0.5)
    print(f"  + Bai 6.1 (x=0.5): y_RK3 = {y[-1]:.6f} | Nghiem dung = {y_true:.6f}")
    assert np.allclose(y, exact_bai_6_1(x), atol=5e-4)


def test_rk3_r7_2_kiem_thu_bat_buoc():
    """[R7 - Ca 2]: Kiểm thử bắt buộc của Nhóm 8 (h=0.2)."""
    print("\n--- [R7 - Ca 2: Kiem thu bat buoc cua Co] ---")
    x, y = rk3(f_bat_buoc, 0.0, 2.0, 0.5, 10)
    print(f"  + Tai x = 2.0 (n=10, h=0.2): y_RK3 = {y[-1]:.6f} (De yeu cau: 5.303725)")
    assert np.isclose(y[-1], 5.303725, atol=1e-5)


def test_rk3_r7_3_vi_du_nhom_tu_chon():
    """[R7 - Ca 3]: Ví dụ nhóm tự chọn (y' = -2xy) + đối chiếu scipy.integrate."""
    print("\n--- [R7 - Ca 3: Vi du nhom tu chon + Doi chieu scipy (n=10)] ---")
    a, b, y0, n = 0.0, 1.0, 1.0, 10
    x, y = rk3(f_tu_chon, a, b, y0, n)
    
    # 1. Kiểm tra với nghiệm giải tích
    y_true = exact_tu_chon(x)
    print(f"  + Tai x = 1.0: y_RK3 = {y[-1]:.6f} | Nghiem dung = {y_true[-1]:.6f}")
    assert np.allclose(y, y_true, atol=1e-3)
    
    # 2. Kiểm tra với thư viện scipy
    sol = solve_ivp(f_tu_chon, (x[0], x[-1]), [y0], t_eval=x, rtol=1e-10, atol=1e-12)
    assert np.allclose(y, sol.y[0], atol=1e-3)
    print(f"  + Doi chieu voi scipy: PASSED!")


def test_rk3_r7_4_ca_bien():
    """[R7 - Ca 4]: Kiểm tra các ca biên (n=1, đa thức, a >= b, n <= 0)."""
    print("\n--- [R7 - Ca 4: Kiem tra cac ca bien cho RK3] ---")
    # 1. Biên n = 1 bước duy nhất
    a, b, y0, n = 0.0, 0.1, 1.0, 1
    h = b - a
    K1 = h * f_bai_6_1(a, y0)
    K2 = h * f_bai_6_1(a + h / 2, y0 + K1 / 2)
    K3 = h * f_bai_6_1(a + h, y0 - K1 + 2 * K2)
    expected_y1 = y0 + (K1 + 4 * K2 + K3) / 6.0
    _, y_step1 = rk3(f_bai_6_1, a, b, y0, n)
    assert np.isclose(y_step1[-1], expected_y1, rtol=1e-12)

    # 2. Biên đa thức bậc 3 (sai số tuyệt đối bằng 0)
    _, y_poly = rk3(lambda x, y: 3.0 * x**2, 0.0, 2.0, 0.0, 4)
    assert np.isclose(y_poly[-1], 8.0, atol=1e-12)
    print("  + Bien n=1 va da thuc bac 3: PASSED!")

    # 3. Biên khoảng tính không hợp lệ a >= b
    for bad_a, bad_b in [(1.0, 1.0), (2.0, 1.0)]:
        with pytest.raises(ValueError, match="Khoang tinh khong hop le"):
            rk3(f_bai_6_1, bad_a, bad_b, 1.0, 10)
    print("  + Bien a >= b: Da bat dung ValueError!")

    # 4. Biên số bước n <= 0 hoặc sai kiểu
    for bad_n in [0, -5, 2.5, True, False]:
        with pytest.raises(ValueError, match="So buoc n phai la so nguyen duong"):
            rk3(f_bai_6_1, 0.0, 1.0, 1.0, bad_n)
    print("  + Bien n <= 0 hoac sai kieu: Da bat dung ValueError!")


if __name__ == "__main__":
    test_rk3_r7_1_vi_du_bai_giang()
    test_rk3_r7_2_kiem_thu_bat_buoc()
    test_rk3_r7_3_vi_du_nhom_tu_chon()
    test_rk3_r7_4_ca_bien()
    print("\n=> PASSED: Toàn bộ kiểm thử Phần 1 (rk3) đã chạy thành công!")
