import numpy as np
import pytest

# IMPORT CHUAN CODE TU PHAN 1, PHAN 2 VA PHAN 4 (UOCLUONG)
from rk3 import rk3
from rk4 import rk4
from uocluong import order_estimate, step_for_tol


# ==============================================================================
# CAC HAM BAI TOAN MAU
# ==============================================================================
# 1. Vi du 6.3 trong bai giang (Kiem thu bat buoc cho P4)
# y' = 3 * x^2 * y, y(0) = 1 tren [0, 1] => Nghiem dung: y(x) = e^(x^3)
def f_vi_du_6_3(x, y):
    return 3.0 * (x**2) * y

def exact_vi_du_6_3(x):
    return np.exp(x**3)

# 2. Bai toan KIEM THU BAT BUOC cua de
# y' = y - x^2 + 1, y(0) = 0.5 tren [0, 2] => Nghiem dung: y(x) = (x + 1)^2 - 0.5 * e^x
def f_bat_buoc(x, y):
    return y - x**2 + 1.0

def exact_bat_buoc(x):
    return (x + 1.0) ** 2 - 0.5 * np.exp(x)

# 3. Bai toan TU CHON (Khac hoan toan cac bai tren)
# y' = -2xy, y(0) = 1 tren [0, 1] => Nghiem dung: y(x) = e^(-x^2)
def f_tu_chon(x, y):
    return -2.0 * x * y

def exact_tu_chon(x):
    return np.exp(-x**2)


# ==============================================================================
# KIEM THU PHAN 4 (order_estimate va step_for_tol)
# ==============================================================================
def test_order_estimate_r7_1_vi_du_6_3():
    """[R7 - Ca 1]: Uoc luong bac hoi tu cho Vi du 6.3 (truyen du 7 tham so co y_exact)."""
    print("--- [R7 - Ca 1 (order_estimate): Vi du 6.3 voi n = 5, 10, 20, 40, 80] ---")
    n_list = [5, 10, 20, 40, 80]
    
    # Truyen du 7 tham so: method, f, a, b, y0, y_exact, n_list
    p3 = order_estimate(rk3, f_vi_du_6_3, 0.0, 1.0, 1.0, exact_vi_du_6_3, n_list)
    p4 = order_estimate(rk4, f_vi_du_6_3, 0.0, 1.0, 1.0, exact_vi_du_6_3, n_list)

    print(f"  + Bac hoi tu RK3 (Vi du 6.3): {[f'{val:.6f}' for val in p3]} | Sai so cuoi so voi 3.0 = {abs(p3[-1] - 3.0):.6f}")
    print(f"  + Bac hoi tu RK4 (Vi du 6.3): {[f'{val:.6f}' for val in p4]} | Sai so cuoi so voi 4.0 = {abs(p4[-1] - 4.0):.6f}")
    assert np.isclose(p3[-1], 3.0, atol=0.2)
    assert np.isclose(p4[-1], 4.0, atol=0.2)


def test_order_estimate_r7_2_kiem_thu_bat_buoc():
    """[R7 - Ca 2]: Uoc luong bac hoi tu cho bai toan bat buoc (truyen du 7 tham so co y_exact)."""
    print("\n--- [R7 - Ca 2 (order_estimate): Bai toan bat buoc y' = y - x^2 + 1] ---")
    n_list = [5, 10, 20, 40, 80]
    
    # Truyen du 7 tham so: method, f, a, b, y0, y_exact, n_list
    p3 = order_estimate(rk3, f_bat_buoc, 0.0, 2.0, 0.5, exact_bat_buoc, n_list)
    p4 = order_estimate(rk4, f_bat_buoc, 0.0, 2.0, 0.5, exact_bat_buoc, n_list)

    print(f"  + Bac hoi tu RK3: {[f'{val:.6f}' for val in p3]} | Sai so cuoi so voi 3.0 = {abs(p3[-1] - 3.0):.6f}")
    print(f"  + Bac hoi tu RK4: {[f'{val:.6f}' for val in p4]} | Sai so cuoi so voi 4.0 = {abs(p4[-1] - 4.0):.6f}")
    assert np.isclose(p3[-1], 3.0, atol=0.15)
    assert np.isclose(p4[-1], 4.0, atol=0.15)


def test_order_estimate_r7_3_vi_du_nhom_tu_chon():
    """[R7 - Ca 3]: Uoc luong bac hoi tu cho bai toan tu chon (truyen du 7 tham so co y_exact)."""
    print("\n--- [R7 - Ca 3 (order_estimate): Bai toan nhom tu chon y' = -2xy] ---")
    n_list = [5, 10, 20, 40, 80]
    
    # Truyen du 7 tham so: method, f, a, b, y0, y_exact, n_list
    p3 = order_estimate(rk3, f_tu_chon, 0.0, 1.0, 1.0, exact_tu_chon, n_list)
    p4 = order_estimate(rk4, f_tu_chon, 0.0, 1.0, 1.0, exact_tu_chon, n_list)

    print(f"  + Bac hoi tu RK3 (Tu chon): {[f'{val:.6f}' for val in p3]}")
    print(f"  + Bac hoi tu RK4 (Tu chon): {[f'{val:.6f}' for val in p4]}")
    assert np.isclose(p3[-1], 3.0, atol=0.15)
    assert np.isclose(p4[-1], 4.0, atol=0.15)


def test_order_estimate_r7_4_ca_bien():
    """[R7 - Ca 4]: Kiem tra cac ca bien cho order_estimate (tat ca cac lenh goi deu truyen du 7 tham so)."""
    print("\n--- [R7 - Ca 4 (order_estimate): Kiem tra cac ca bien] ---")
    # 1. Bien a >= b
    for bad_a, bad_b in [(1.0, 1.0), (2.0, 1.0)]:
        with pytest.raises(ValueError):
            order_estimate(rk4, f_vi_du_6_3, bad_a, bad_b, 1.0, exact_vi_du_6_3, [5, 10, 20])
    print("  + Bien a >= b: Da bat dung ValueError!")

    # 2. Bien n_list rong
    with pytest.raises(ValueError):
        order_estimate(rk4, f_vi_du_6_3, 0.0, 1.0, 1.0, exact_vi_du_6_3, [])
    print("  + Bien n_list rong []: Da bat dung ValueError!")

    # 3. Bien n_list chua so <= 0, float, bool
    for bad_list in [[0, 2, 4], [-4, -8], [2.5, 5.0], [True, 2]]:
        with pytest.raises((ValueError, TypeError)):
            order_estimate(rk3, f_vi_du_6_3, 0.0, 1.0, 1.0, exact_vi_du_6_3, bad_list)
    print("  + Bien n_list chua so <= 0 / float / bool: Da bat dung ngoai le!")

    # 4. Bien n_list khong nhan doi
    with pytest.raises(ValueError):
        order_estimate(rk4, f_vi_du_6_3, 0.0, 1.0, 1.0, exact_vi_du_6_3, [5, 10, 25])
    print("  + Bien n_list khong nhan doi [5, 10, 25]: Da bat dung ValueError!")

    # 5. Sai so e_n = 0 (phuong trinh da thuc bac 4 dung voi RK4 thi sai so = 0, gay loi chia 0 / log2)
    with pytest.raises((ArithmeticError, ZeroDivisionError, ValueError)):
        order_estimate(rk4, lambda x, y: 4.0 * x**3, 0.0, 1.0, 0.0, lambda x: x**4, [2, 4, 8])
    print("  + Bien sai so bang 0 (e_n = 0): Da bat dung ArithmeticError!")


def test_step_for_tol_r7_1_vi_du_6_3():
    """[R7 - Ca 5]: Tim buoc luoi thoa man tol cho Vi du 6.3 voi tol = 1e-6."""
    print("\n--- [R7 - Ca 5 (step_for_tol): Vi du 6.3 voi tol = 1e-6] ---")
    tol = 1e-6
    h3, yb3 = step_for_tol(rk3, f_vi_du_6_3, 0.0, 1.0, 1.0, tol)
    h4, yb4 = step_for_tol(rk4, f_vi_du_6_3, 0.0, 1.0, 1.0, tol)
    y_true = exact_vi_du_6_3(1.0)

    print(f"  + RK3: h = {h3:.6f} | y(1.0) = {yb3:.6f} | Nghiem dung = {y_true:.6f} | Sai so = {abs(yb3 - y_true):.6f}")
    print(f"  + RK4: h = {h4:.6f} | y(1.0) = {yb4:.6f} | Nghiem dung = {y_true:.6f} | Sai so = {abs(yb4 - y_true):.6f}")
    assert abs(yb3 - y_true) < 15 * tol
    assert abs(yb4 - y_true) < 15 * tol
    assert h4 >= h3


def test_step_for_tol_r7_2_kiem_thu_bat_buoc():
    """[R7 - Ca 6]: Tim buoc luoi cho bai toan bat buoc (y' = y - x^2 + 1) voi tol = 1e-5."""
    print("\n--- [R7 - Ca 6 (step_for_tol): Bai toan bat buoc voi tol = 1e-5] ---")
    tol = 1e-5
    h3, yb3 = step_for_tol(rk3, f_bat_buoc, 0.0, 2.0, 0.5, tol)
    h4, yb4 = step_for_tol(rk4, f_bat_buoc, 0.0, 2.0, 0.5, tol)
    y_true = exact_bat_buoc(2.0)

    print(f"  + RK3: h = {h3:.6f} | y(2.0) = {yb3:.6f} | Nghiem dung = {y_true:.6f} | Sai so = {abs(yb3 - y_true):.6f}")
    print(f"  + RK4: h = {h4:.6f} | y(2.0) = {yb4:.6f} | Nghiem dung = {y_true:.6f} | Sai so = {abs(yb4 - y_true):.6f}")
    assert abs(yb3 - y_true) < 10 * tol
    assert abs(yb4 - y_true) < 10 * tol
    assert h4 >= h3


def test_step_for_tol_r7_3_ca_bien():
    """[R7 - Ca 7]: Kiem tra cac ca bien cho step_for_tol (a >= b, tol <= 0, vuot max_n)."""
    print("\n--- [R7 - Ca 7 (step_for_tol): Kiem tra cac ca bien] ---")
    # 1. Bien a >= b
    for bad_a, bad_b in [(1.0, 1.0), (2.0, 1.0)]:
        with pytest.raises(ValueError):
            step_for_tol(rk4, f_vi_du_6_3, bad_a, bad_b, 1.0, 1e-4)
    print("  + Bien a >= b: Da bat dung ValueError!")

    # 2. Bien tol <= 0
    for bad_tol in [0.0, -1e-5]:
        with pytest.raises(ValueError):
            step_for_tol(rk3, f_vi_du_6_3, 0.0, 1.0, 1.0, bad_tol)
    print("  + Bien tol <= 0: Da bat dung ValueError!")

    # 3. Bien vuot qua max_n
    def non_converging_method(f, a, b, y0, n):
        return np.array([a, b]), np.array([y0, float(n)])

    with pytest.raises((RuntimeError, ValueError)):
        step_for_tol(non_converging_method, f_vi_du_6_3, 0.0, 1.0, 1.0, 1e-3)
    print("  + Bien vuot gioi han max_n: Da bat dung ngoai le!")


if __name__ == "__main__":
    test_order_estimate_r7_1_vi_du_6_3()
    test_order_estimate_r7_2_kiem_thu_bat_buoc()
    test_order_estimate_r7_3_vi_du_nhom_tu_chon()
    test_order_estimate_r7_4_ca_bien()
    test_step_for_tol_r7_1_vi_du_6_3()
    test_step_for_tol_r7_2_kiem_thu_bat_buoc()
    test_step_for_tol_r7_3_ca_bien()
    print("\n=> PASSED: Toan bo kiem thu Phan 4 (order_estimate & step_for_tol) theo chuan Nhom 8 da chay thanh cong!")