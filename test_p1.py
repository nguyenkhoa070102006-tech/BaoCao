import numpy as np
import pytest
from scipy.integrate import solve_ivp

# IMPORT CHUAN CODE CUA SV1 (PHAN 1)
from rk3 import rk3


# ==============================================================================
# CAC HAM BAI TOAN MAU THEO DE NHOM 8 VA R7
# ==============================================================================
# 1. Vi du 6.3 trong bai giang (Kiem thu bat buoc Nhom 8):
# y' = 3 * x^2 * y, y(0) = 1 tren [0, 1], n = 5 (h = 0.2) => Nghiem dung: y(x) = e^(x^3)
def f_vi_du_6_3(x, y):
    return 3.0 * (x**2) * y

def exact_vi_du_6_3(x):
    return np.exp(x**3)

# 2. Bai toan bat buoc Nhom 8 & Nhom tu chon:
# y' = y - x^2 + 1, y(0) = 0.5 tren [0, 2], h = 0.2 (n = 10)
# Ket qua bat buoc: y_10 = 5.303725 (RK3); nghiem dung = 5.305472
def f_bat_buoc(x, y):
    return y - x**2 + 1.0

def exact_bat_buoc(x):
    return (x + 1.0) ** 2 - 0.5 * np.exp(x)


# ==============================================================================
# KIEM THU PHAN 1 (SV5 - THEO CHUAN R7 & KIEM THU BAT BUOC NHOM 8)
# ==============================================================================
def test_rk3_r7_1_vi_du_bai_giang_6_3():
    """[R7 - Ca 1]: Vi du 6.3 trong bai giang (y' = 3*x^2*y, y(0) = 1, [0, 1], n = 5)."""
    print("--- [R7 - Ca 1: Vi du 6.3 trong bai giang cho RK3] ---")
    x, y = rk3(f_vi_du_6_3, 0.0, 1.0, 1.0, 5)
    y_true = exact_vi_du_6_3(1.0)
    print(f"  + Vi du 6.3 (x=1.0, n=5): y_5 (RK3) = {y[-1]:.6f} | Nghiem dung = {y_true:.6f} | Sai so = {abs(y[-1] - y_true):.6f}")
    assert len(x) == 6 and len(y) == 6
    assert abs(y[-1] - y_true) < 1e-3


def test_rk3_r7_2_kiem_thu_bat_buoc_nhom_8():
    """[R7 - Ca 2]: Kiem thu bat buoc Nhom 8 (y' = y - x^2 + 1, y(0) = 0.5, [0, 2], h = 0.2)."""
    print("\n--- [R7 - Ca 2: Kiem thu bat buoc Nhom 8 cho RK3 (y' = y - x^2 + 1)] ---")
    a, b, y0, n = 0.0, 2.0, 0.5, 10  # h = (2.0 - 0.0) / 10 = 0.2
    x, y = rk3(f_bat_buoc, a, b, y0, n)
    y_true = exact_bat_buoc(2.0)
    sol = solve_ivp(f_bat_buoc, (x[0], x[-1]), [y0], t_eval=x, rtol=1e-10, atol=1e-12)

    print(f"  + Tai x = 2.0 (n=10, h=0.2): y_10 (RK3) = {y[-1]:.6f} (De yeu cau: 5.303725)")
    print(f"  + Nghiem dung = {y_true:.6f} (De yeu cau: 5.305472) | Sai so = {abs(y[-1] - y_true):.6f}")
    # Kiem tra khop chinh xac con so bat buoc trong de: y_10 = 5.303725 va nghiem dung = 5.305472
    assert np.isclose(y[-1], 5.303725, atol=1e-6)
    assert np.isclose(y_true, 5.305472, atol=1e-6)
    assert np.allclose(y, sol.y[0], atol=3e-3)


def test_rk3_r7_3_ca_bien():
    """[R7 - Ca 3]: Ca bien bat buoc Nhom 8 (n <= 0 hoac a >= b phai bao loi)."""
    print("\n--- [R7 - Ca 3: Kiem tra cac ca bien bat buoc cho RK3] ---")
    # 1. Bien da thuc bac 3 (y' = 3x^2, y(0) = 0 tren [0, 2] => y(2) = 8.0)
    _, y_poly = rk3(lambda x, y: 3.0 * x**2, 0.0, 2.0, 0.0, 4)
    print(f"  + Bien da thuc bac 3 (y' = 3x^2 tai x=2): y_RK3 = {y_poly[-1]:.6f} (Nghiem dung = 8.000000)")
    assert np.isclose(y_poly[-1], 8.0, atol=1e-12)

    # 2. Ca bien bat buoc: a >= b (a == b va a > b) phai bao loi ValueError
    for bad_a, bad_b in [(1.0, 1.0), (2.0, 1.0)]:
        with pytest.raises(ValueError):
            rk3(f_vi_du_6_3, bad_a, bad_b, 1.0, 5)
    print("  + Bien a >= b (1.0 == 1.0 va 2.0 > 1.0): Da bao loi ValueError dung yeu cau!")

    # 3. Ca bien bat buoc: n <= 0 hoac sai kieu (0, -5, 2.5, True) phai bao loi
    for bad_n in [0, -5, 2.5, True, False]:
        with pytest.raises((ValueError, TypeError, ZeroDivisionError)):
            rk3(f_vi_du_6_3, 0.0, 1.0, 1.0, bad_n)
    print("  + Bien n <= 0 hoac sai kieu (0, -5, 2.5, True): Da bao loi dung yeu cau!")


if __name__ == "__main__":
    test_rk3_r7_1_vi_du_bai_giang_6_3()
    test_rk3_r7_2_kiem_thu_bat_buoc_nhom_8()
    test_rk3_r7_3_ca_bien()
    print("\n=> PASSED: Toan bo kiem thu Phan 1 (rk3) theo chuan Nhom 8 & R7 da chay thanh cong!")