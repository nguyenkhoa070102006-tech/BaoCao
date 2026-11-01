import numpy as np
import pytest
from scipy.integrate import solve_ivp

# IMPORT 
from rk3 import rk3


# ==============================================================================
# CAC HAM BAI TOAN MAU CHO KIEM THU PHAN 1
# ==============================================================================
def f_bai_6_1(x, y):
    return 2.0 - np.exp(-4.0 * x) - 2.0 * y

def exact_bai_6_1(x):
    return 1.0 + 0.5 * np.exp(-4.0 * x) - 0.5 * np.exp(-2.0 * x)

def f_bai_6_2(x, y):
    return x**2 - 4.0 * y

def exact_bai_6_2(x):
    return (31.0 / 32.0) * np.exp(-4.0 * x) + 0.25 * x**2 - 0.125 * x + (1.0 / 32.0)

def f_bai_6_17(x, y):
    return 0.5 * x * y

def exact_bai_6_17(x):
    return np.exp(x**2 / 4.0)

def f_nhom_tu_chon_1(x, y):
    return y - x**2 + 1.0

def exact_nhom_tu_chon_1(x):
    return (x + 1.0) ** 2 - 0.5 * np.exp(x)


# ==============================================================================
# KIEM THU PHAN 1 
# ==============================================================================
def test_rk3_r7_1_vi_du_bai_giang():
    """[R7 - Ca 1]: Vi du trong bai giang/giao trinh (Bai 6.1, 6.2, 6.17)."""
    print("--- [R7 - Ca 1: Vi du giao trinh cho RK3] ---")
    x_61, y_61 = rk3(f_bai_6_1, 0.0, 0.5, 1.0, 5)
    y_true_61 = exact_bai_6_1(0.5)
    print(f"  + Bai 6.1 (x=0.5): y_RK3 = {y_61[-1]:.8f} | Nghiem dung = {y_true_61:.8f} | Sai so = {abs(y_61[-1] - y_true_61):.2e}")
    assert np.allclose(y_61, exact_bai_6_1(x_61), atol=5e-4)

    x_62, y_62 = rk3(f_bai_6_2, 0.0, 0.03, 1.0, 3)
    y_true_62 = exact_bai_6_2(0.03)
    print(f"  + Bai 6.2 (x=0.03): y_RK3 = {y_62[-1]:.8f} | Nghiem dung = {y_true_62:.8f} | Sai so = {abs(y_62[-1] - y_true_62):.2e}")
    assert np.allclose(y_62, exact_bai_6_2(x_62), atol=1e-6)

    _, y_617 = rk3(f_bai_6_17, 0.0, 1.0, 1.0, 10)
    y_true_617 = exact_bai_6_17(1.0)
    print(f"  + Bai 6.17 (x=1.0): y_RK3 = {y_617[-1]:.8f} | Nghiem dung = {y_true_617:.8f} | Sai so = {abs(y_617[-1] - y_true_617):.2e}")
    assert abs(y_617[-1] - y_true_617) < 5e-5


def test_rk3_r7_2_vi_du_nhom_tu_chon():
    """[R7 - Ca 2]: Vi du co nghiem dung do nhom tu chon + doi chieu scipy.integrate."""
    print("\n--- [R7 - Ca 2: Vi du nhom tu chon cho RK3 (y' = y - x^2 + 1)] ---")
    a, b, y0, n = 0.0, 2.0, 0.5, 20
    x, y = rk3(f_nhom_tu_chon_1, a, b, y0, n)
    y_true = exact_nhom_tu_chon_1(x)
    sol = solve_ivp(f_nhom_tu_chon_1, (x[0], x[-1]), [y0], t_eval=x, rtol=1e-10, atol=1e-12)

    print(f"  + Tai x = 2.0: y_RK3 = {y[-1]:.8f} | Nghiem giai tich = {y_true[-1]:.8f} | Scipy = {sol.y[0][-1]:.8f}")
    print(f"  + Sai so lon nhat tren toan luoi: {np.max(np.abs(y - y_true)):.2e}")
    assert np.allclose(y, y_true, atol=2e-3)
    assert np.allclose(y, sol.y[0], atol=2e-3)


def test_rk3_r7_3_ca_bien():
    """[R7 - Ca 3]: Ca bien (n = 1, da thuc bac 3, khoang tinh a >= b, va n = 0)."""
    print("\n--- [R7 - Ca 3: Kiem tra cac ca bien cho RK3] ---")
    # 1. Bien n = 1 buoc
    a, b, y0, n = 0.0, 0.1, 1.0, 1
    h = b - a
    K1 = h * f_bai_6_1(a, y0)
    K2 = h * f_bai_6_1(a + h / 2, y0 + K1 / 2)
    K3 = h * f_bai_6_1(a + h, y0 - K1 + 2 * K2)
    expected_y1 = y0 + (K1 + 4 * K2 + K3) / 6.0
    _, y_step1 = rk3(f_bai_6_1, a, b, y0, n)
    print(f"  + Bien n = 1 buoc: y_RK3(0.1) = {y_step1[-1]:.8f} == Cong thuc tay = {expected_y1:.8f}")
    assert np.isclose(y_step1[-1], expected_y1, rtol=1e-12)

    # 2. Bien da thuc bac 3
    _, y_poly = rk3(lambda x, y: 3.0 * x**2, 0.0, 2.0, 0.0, 4)
    print(f"  + Bien da thuc bac 3 (y' = 3x^2 tai x=2): y_RK3 = {y_poly[-1]:.12f} (Nghiem dung = 8.0)")
    assert np.isclose(y_poly[-1], 8.0, atol=1e-12)

    # 3. Bo sung ca bien khoang tinh a >= b 
    # Truong hop a == b (khoang suy bien h = 0): nghiem y khong doi bang y0
    try:
        x_eq, y_eq = rk3(f_bai_6_17, 1.0, 1.0, 2.5, 5)
        assert np.allclose(x_eq, 1.0) and np.allclose(y_eq, 2.5)
        print("  + Bien a == b (1.0 == 1.0): h = 0, nghiem giu nguyen y = 2.5!")
    except ValueError:
        print("  + Bien a == b: Da bat ngoai le ValueError!")

    # Truong hop a > b (khoang nguoc h < 0): luoi x giam dan tu a ve b (hoac nem ValueError)
    try:
        x_rev, y_rev = rk3(f_bai_6_17, 1.0, 0.0, exact_bai_6_17(1.0), 10)
        assert x_rev[0] > x_rev[-1] and np.isclose(y_rev[-1], exact_bai_6_17(0.0), atol=1e-4)
        print(f"  + Bien a > b (1.0 > 0.0): h = -0.1 < 0, y_RK3(0.0) = {y_rev[-1]:.8f}!")
    except ValueError:
        print("  + Bien a > b: Da bat ngoai le ValueError!")

    # 4. Giu nguyen doan bat loi n = 0 (theo yeu cau muc 2)
    with pytest.raises(ZeroDivisionError):
        rk3(f_bai_6_17, 0.0, 1.0, 1.0, 0)
    print("  + Bien n = 0: Da bat dung loi ZeroDivisionError!")


if __name__ == "__main__":
    test_rk3_r7_1_vi_du_bai_giang()
    test_rk3_r7_2_vi_du_nhom_tu_chon()
    test_rk3_r7_3_ca_bien()
    print("\n=> PASSED: Toan bo kiem thu Phan 1 (rk3) theo chuan R7 da chay thanh cong!")