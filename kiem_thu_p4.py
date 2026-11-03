import numpy as np
import pytest

from rk3 import rk3
from rk4 import rk4
from evaluation import order_estimate, step_for_tol


# ==============================================================================
# CAC HAM BAI TOAN MAU CHO KIEM THU PHAN 4
# ==============================================================================
# 1. Vi du trong bai giang / giao trinh:
# Bai 6.1 (Trang 141): y' + 2y = 2 - e^(-4x) => y' = 2 - e^(-4x) - 2y, y(0) = 1
def f_bai_6_1(x, y):
    return 2.0 - np.exp(-4.0 * x) - 2.0 * y

def exact_bai_6_1(x):
    return 1.0 + 0.5 * np.exp(-4.0 * x) - 0.5 * np.exp(-2.0 * x)

# Bai 6.17 (Trang 142): y' = (1/2)*x*y, y(0) = 1, tren [0, 1]
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
# THUAT TOAN 1 
# ==============================================================================
def test_order_estimate_r7_1_vi_du_bai_giang():
    """[R7 - Ca 1 cho order_estimate]: Vi du trong bai giang/giao trinh (Bai 6.1 va Bai 6.17)."""
    print("--- [R7 - Ca 1 (order_estimate): Uoc luong bac hoi tu tren Bai 6.1 & Bai 6.17] ---")
    n_list = [10, 20, 40, 80]
    p3 = order_estimate(rk3, f_bai_6_1, 0.0, 0.5, 1.0, n_list)
    p4 = order_estimate(rk4, f_bai_6_17, 0.0, 1.0, 1.0, n_list)

    print(f"  + Bac hoi tu RK3 (Bai 6.1) : {[f'{val:.6f}' for val in p3]} | Sai so cuoi so voi 3.0 = {abs(p3[-1] - 3.0):.6f}")
    print(f"  + Bac hoi tu RK4 (Bai 6.17): {[f'{val:.6f}' for val in p4]} | Sai so cuoi so voi 4.0 = {abs(p4[-1] - 4.0):.6f}")
    assert len(p3) == 3 and len(p4) == 3
    assert np.isclose(p3[-1], 3.0, atol=0.1)
    assert np.isclose(p4[-1], 4.0, atol=0.1)


def test_order_estimate_r7_2_vi_du_nhom_tu_chon():
    """[R7 - Ca 2 cho order_estimate]: Vi du nhom tu chon (y' = y - x^2 + 1, y(0) = 0.5)."""
    print("\n--- [R7 - Ca 2 (order_estimate): Vi du nhom tu chon (y' = y - x^2 + 1)] ---")
    n_list = [10, 20, 40, 80]
    p3 = order_estimate(rk3, f_nhom_tu_chon_1, 0.0, 2.0, 0.5, n_list)
    p4 = order_estimate(rk4, f_nhom_tu_chon_1, 0.0, 2.0, 0.5, n_list)

    print(f"  + Bac hoi tu RK3: {[f'{val:.6f}' for val in p3]} | Sai so cuoi so voi 3.0 = {abs(p3[-1] - 3.0):.6f}")
    print(f"  + Bac hoi tu RK4: {[f'{val:.6f}' for val in p4]} | Sai so cuoi so voi 4.0 = {abs(p4[-1] - 4.0):.6f}")
    assert np.isclose(p3[-1], 3.0, atol=0.1)
    assert np.isclose(p4[-1], 4.0, atol=0.1)


def test_order_estimate_r7_3_ca_bien():
    """[R7 - Ca 3 cho order_estimate]: Kiem tra cac tinh huong bien theo R6."""
    print("\n--- [R7 - Ca 3 (order_estimate): Kiem tra cac ca bien] ---")
    # 1. Khoang tinh a >= b (a == b va a > b)
    for bad_a, bad_b in [(1.0, 1.0), (2.0, 1.0)]:
        with pytest.raises(ValueError, match="Can thoa man a < b"):
            order_estimate(rk4, f_bai_6_17, bad_a, bad_b, 1.0, [4, 8, 16])
    print("  + Bien a >= b: Da bat dung ValueError ('Can thoa man a < b.')!")

    # 2. Danh sach n_list rong
    with pytest.raises(ValueError, match="n_list khong duoc rong"):
        order_estimate(rk4, f_bai_6_17, 0.0, 1.0, 1.0, [])
    print("  + Bien n_list rong []: Da bat dung ValueError ('n_list khong duoc rong.')!")

    # 3. Phan tu trong n_list <= 0, float hoac bool
    for bad_list in [[0, 2, 4], [-4, -8], [2.5, 5.0], [True, 2]]:
        with pytest.raises(ValueError, match="so nguyen duong"):
            order_estimate(rk3, f_bai_6_17, 0.0, 1.0, 1.0, bad_list)
    print("  + Bien n_list chua so <= 0 / float / bool: Da bat dung ValueError ('so nguyen duong')!")

    # 4. Cac phan tu trong n_list khong tang gap doi
    with pytest.raises(ValueError, match="tang gap doi"):
        order_estimate(rk4, f_bai_6_17, 0.0, 1.0, 1.0, [5, 10, 25])
    print("  + Bien n_list khong nhan doi [5, 10, 25]: Da bat dung ValueError ('tang gap doi')!")

    # 5. Sai khac D_n = 0 (phuong trinh y' = 0 co nghiem hang so tren moi luoi)
    with pytest.raises(ArithmeticError, match="sai khac bang 0"):
        order_estimate(rk4, lambda x, y: 0.0, 0.0, 1.0, 5.0, [2, 4, 8])
    print("  + Bien sai khac hai luoi D_n = 0: Da bat dung ArithmeticError ('sai khac bang 0')!")


# ==============================================================================
# THUAT TOAN 2 
# ==============================================================================
def test_step_for_tol_r7_1_vi_du_bai_giang():
    """[R7 - Ca 1 cho step_for_tol]: Bai 6.17 tren [0, 1] voi tol = 1e-6."""
    print("\n--- [R7 - Ca 1 (step_for_tol): Tim buoc luoi tren Bai 6.17 voi tol = 1e-6] ---")
    tol = 1e-6
    h3, yb3 = step_for_tol(rk3, f_bai_6_17, 0.0, 1.0, 1.0, tol)
    h4, yb4 = step_for_tol(rk4, f_bai_6_17, 0.0, 1.0, 1.0, tol)
    y_true = exact_bai_6_17(1.0)

    print(f"  + RK3: h = {h3:.6f} | y(1.0) = {yb3:.6f} | Nghiem dung = {y_true:.6f} | Sai so = {abs(yb3 - y_true):.6f}")
    print(f"  + RK4: h = {h4:.6f} | y(1.0) = {yb4:.6f} | Nghiem dung = {y_true:.6f} | Sai so = {abs(yb4 - y_true):.6f}")
    assert abs(yb3 - y_true) < 10 * tol
    assert abs(yb4 - y_true) < 10 * tol
    assert h4 >= h3


def test_step_for_tol_r7_2_vi_du_nhom_tu_chon():
    """[R7 - Ca 2 cho step_for_tol]: Vi du nhom tu chon (y' = y - x^2 + 1) voi tol = 1e-5."""
    print("\n--- [R7 - Ca 2 (step_for_tol): Vi du nhom tu chon voi tol = 1e-5] ---")
    tol = 1e-5
    h3, yb3 = step_for_tol(rk3, f_nhom_tu_chon_1, 0.0, 2.0, 0.5, tol)
    h4, yb4 = step_for_tol(rk4, f_nhom_tu_chon_1, 0.0, 2.0, 0.5, tol)
    y_true = exact_nhom_tu_chon_1(2.0)

    print(f"  + RK3: h = {h3:.6f} | y(2.0) = {yb3:.6f} | Nghiem dung = {y_true:.6f} | Sai so = {abs(yb3 - y_true):.6f}")
    print(f"  + RK4: h = {h4:.6f} | y(2.0) = {yb4:.6f} | Nghiem dung = {y_true:.6f} | Sai so = {abs(yb4 - y_true):.6f}")
    assert abs(yb3 - y_true) < 10 * tol
    assert abs(yb4 - y_true) < 10 * tol
    assert h4 >= h3


def test_step_for_tol_r7_3_ca_bien():
    """[R7 - Ca 3 cho step_for_tol]: Cac ca bien (a >= b, tol <= 0, vuot max_n)."""
    print("\n--- [R7 - Ca 3 (step_for_tol): Kiem tra cac ca bien] ---")
    # 1. Khoang tinh a >= b
    for bad_a, bad_b in [(1.0, 1.0), (2.0, 1.0)]:
        with pytest.raises(ValueError, match="Can thoa man a < b"):
            step_for_tol(rk4, f_bai_6_17, bad_a, bad_b, 1.0, 1e-4)
    print("  + Bien a >= b: Da bat dung ValueError ('Can thoa man a < b.')!")

    # 2. Nguong sai so tol <= 0
    for bad_tol in [0.0, -1e-5]:
        with pytest.raises(ValueError, match="tol phai lon hon 0"):
            step_for_tol(rk3, f_bai_6_17, 0.0, 1.0, 1.0, bad_tol)
    print("  + Bien tol <= 0: Da bat dung ValueError ('tol phai lon hon 0.')!")

    # 3. Vuot qua so buoc toi da max_n = 1_000_000 (RuntimeError)
    def non_converging_method(f, a, b, y0, n):
        return np.array([a, b]), np.array([y0, float(n)])

    with pytest.raises(RuntimeError, match="Khong tim duoc buoc luoi"):
        step_for_tol(non_converging_method, f_bai_6_17, 0.0, 1.0, 1.0, 1e-3)
    print("  + Bien vuot gioi han max_n: Da bat dung RuntimeError ('Khong tim duoc buoc luoi')!")


if __name__ == "__main__":
    test_order_estimate_r7_1_vi_du_bai_giang()
    test_order_estimate_r7_2_vi_du_nhom_tu_chon()
    test_order_estimate_r7_3_ca_bien()
    test_step_for_tol_r7_1_vi_du_bai_giang()
    test_step_for_tol_r7_2_vi_du_nhom_tu_chon()
    test_step_for_tol_r7_3_ca_bien()
    print("\n=> PASSED: Toan bo kiem thu Phan 4 (order_estimate & step_for_tol) theo chuan R7 da chay thanh cong!")