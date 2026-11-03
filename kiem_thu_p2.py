import numpy as np
import pytest
from scipy.integrate import solve_ivp

from rk4 import rk4


# ==============================================================================
# CAC HAM BAI TOAN MAU CHO KIEM THU PHAN 2
# ==============================================================================
# 1. Vi du trong bai giang / giao trinh:
# Bai 6.11 (Trang 142): 3y' = x^2 + y^2 => y' = (x^2 + y^2)/3, y(1) = 1.2, h = 0.05
def f_bai_6_11(x, y):
    return (x**2 + y**2) / 3.0

# Bai 6.16 (Trang 142): y' = 3y - 4*e^(-x), y(0) = 1, h = 0.1 => Nghiem dung: y(x) = e^(-x)
def f_bai_6_16(x, y):
    return 3.0 * y - 4.0 * np.exp(-x)

def exact_bai_6_16(x):
    return np.exp(-x)

# Bai 6.17 (Trang 142): y' = (1/2)*x*y, y(0) = 1, tren [0, 1], h = 0.1 => Nghiem dung: y(x) = e^(x^2 / 4)
def f_bai_6_17(x, y):
    return 0.5 * x * y

def exact_bai_6_17(x):
    return np.exp(x**2 / 4.0)

# 2. Vi du co nghiem dung do nhom tu chon:
# y' = y - x^2 + 1, y(0) = 0.5 tren [0, 2] => Nghiem dung: y(x) = (x + 1)^2 - 0.5 * e^x
def f_nhom_tu_chon_1(x, y):
    return y - x**2 + 1.0

def exact_nhom_tu_chon_1(x):
    return (x + 1.0) ** 2 - 0.5 * np.exp(x)


# ==============================================================================
# KIEM THU PHAN 2 
# ==============================================================================
def test_rk4_r7_1_vi_du_bai_giang():
    """[R7 - Ca 1]: Vi du trong bai giang/giao trinh (Bai 6.11, 6.16, 6.17)."""
    print("--- [R7 - Ca 1: Vi du giao trinh cho RK4] ---")
    # Bai 6.11: 3y' = x^2 + y^2, y(1) = 1.2, tinh y(1.05) voi h = 0.05 (n = 1)
    h = 0.05
    K1 = h * f_bai_6_11(1.0, 1.2)
    K2 = h * f_bai_6_11(1.0 + h / 2, 1.2 + K1 / 2)
    K3 = h * f_bai_6_11(1.0 + h / 2, 1.2 + K2 / 2)
    K4 = h * f_bai_6_11(1.0 + h, 1.2 + K3)
    expected_y_611 = 1.2 + (K1 + 2 * K2 + 2 * K3 + K4) / 6.0

    _, y_611 = rk4(f_bai_6_11, 1.0, 1.05, 1.2, 1)
    print(f"  + Bai 6.11 (x=1.05): y_RK4 = {y_611[-1]:.6f} | Cong thuc tay = {expected_y_611:.6f} | Sai so = {abs(y_611[-1] - expected_y_611):.6f}")
    assert np.isclose(y_611[-1], expected_y_611, rtol=1e-12)

    # Bai 6.16: [0, 1.0], y(0) = 1, h = 0.1 (n = 10)
    _, y_616 = rk4(f_bai_6_16, 0.0, 1.0, 1.0, 10)
    y_true_616 = exact_bai_6_16(1.0)
    print(f"  + Bai 6.16 (x=1.0) : y_RK4 = {y_616[-1]:.6f} | Nghiem dung = {y_true_616:.6f} | Sai so = {abs(y_616[-1] - y_true_616):.6f}")
    assert abs(y_616[-1] - y_true_616) < 2e-4

    # Bai 6.17: [0, 1.0], y(0) = 1, h = 0.1 (n = 10)
    _, y_617 = rk4(f_bai_6_17, 0.0, 1.0, 1.0, 10)
    y_true_617 = exact_bai_6_17(1.0)
    print(f"  + Bai 6.17 (x=1.0) : y_RK4 = {y_617[-1]:.6f} | Nghiem dung = {y_true_617:.6f} | Sai so = {abs(y_617[-1] - y_true_617):.6f}")
    assert abs(y_617[-1] - y_true_617) < 1e-6


def test_rk4_r7_2_vi_du_nhom_tu_chon():
    """[R7 - Ca 2]: Vi du co nghiem dung do nhom tu chon + doi chieu scipy.integrate."""
    print("\n--- [R7 - Ca 2: Vi du nhom tu chon cho RK4 (y' = y - x^2 + 1)] ---")
    a, b, y0, n = 0.0, 2.0, 0.5, 10
    x, y = rk4(f_nhom_tu_chon_1, a, b, y0, n)
    y_true = exact_nhom_tu_chon_1(x)
    sol = solve_ivp(f_nhom_tu_chon_1, (x[0], x[-1]), [y0], t_eval=x, rtol=1e-11, atol=1e-13)

    print(f"  + Tai x = 2.0: y_RK4 = {y[-1]:.6f} | Nghiem giai tich = {y_true[-1]:.6f} | Scipy = {sol.y[0][-1]:.6f}")
    print(f"  + Sai so lon nhat tren toan luoi: {np.max(np.abs(y - y_true)):.6f}")
    assert np.allclose(y, y_true, atol=2e-4)
    assert np.allclose(y, sol.y[0], atol=2e-4)


def test_rk4_r7_3_ca_bien():
    """[R7 - Ca 3]: Ca bien (da thuc bac 4, ham hang y0 am, khoang tinh a >= b, va n <= 0 / sai kieu)."""
    print("\n--- [R7 - Ca 3: Kiem tra cac ca bien cho RK4 (theo code update cua SV2)] ---")
    # 1. Bien da thuc bac 4 (sai so chat cut bang 0): y' = 4x^3, y(0) = 0 => y(2) = 16.0
    _, y_poly = rk4(lambda x, y: 4.0 * x**3, 0.0, 2.0, 0.0, 4)
    print(f"  + Bien da thuc bac 4 (y' = 4x^3 tai x=2): y_RK4 = {y_poly[-1]:.6f} (Nghiem dung = 16.0)")
    assert np.isclose(y_poly[-1], 16.0, atol=1e-12)

    # 2. Bien ham hang y0 am
    _, y_const = rk4(lambda x, y: 0.0, 0.0, 5.0, -10.0, 5)
    print(f"  + Bien ham hang (y' = 0, y0 = -10.0): y_RK4(5.0) = {y_const[-1]:.6f}")
    assert np.allclose(y_const, -10.0, atol=1e-15)

    # 3. Bien khoang tinh khong hop le a >= b (a == b va a > b) -> bat ValueError
    for bad_a, bad_b in [(1.0, 1.0), (2.0, 1.0)]:
        with pytest.raises(ValueError):
            rk4(f_bai_6_17, bad_a, bad_b, 1.0, 10)
    print("  + Bien a >= b (a == b va a > b): Da bat dung ValueError ('a phai nho hon b')!")

    # 4. Bien so buoc n <= 0 hoac sai kieu (n = 0, n = -5, float, bool) -> bat ValueError
    for bad_n in [0, -5, 2.5, True, False]:
        with pytest.raises(ValueError):
            rk4(f_bai_6_17, 0.0, 1.0, 1.0, bad_n)
    print("  + Bien n <= 0 hoac sai kieu (0, -5, 2.5, True): Da bat dung ValueError ('n phai la so nguyen duong')!")


if __name__ == "__main__":
    test_rk4_r7_1_vi_du_bai_giang()
    test_rk4_r7_2_vi_du_nhom_tu_chon()
    test_rk4_r7_3_ca_bien()
    print("\n=> PASSED: Toan bo kiem thu Phan 2 (rk4) theo chuan R7 da chay thanh cong!")