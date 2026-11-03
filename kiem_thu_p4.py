import numpy as np
import pytest

from rk3 import rk3
from rk4 import rk4
from uocluong import order_estimate, step_for_tol


# ==============================================================================
# CAC HAM BAI TOAN MAU CHO KIEM THU PHAN 4
# ==============================================================================
def f_bai_6_1(x, y):
    return 2.0 - np.exp(-4.0 * x) - 2.0 * y

def f_bai_6_17(x, y):
    return 0.5 * x * y

def exact_bai_6_17(x):
    return np.exp(x**2 / 4.0)

def f_nhom_tu_chon_1(x, y):
    return y - x**2 + 1.0

def exact_nhom_tu_chon_1(x):
    return (x + 1.0) ** 2 - 0.5 * np.exp(x)


# ==============================================================================
# KIEM THU PHAN 4
# ==============================================================================
def test_order_estimate_r7_1_vi_du_bai_giang():
    """[R7 - Ca 1 cho order_estimate]: Bai 6.1 va Bai 6.17."""
    print("--- [R7 - Ca 1 (order_estimate): Uoc luong bac hoi tu tren Bai 6.1 & Bai 6.17] ---")
    n_list = [10, 20, 40, 80]
    p3 = order_estimate(rk3, f_bai_6_1, 0.0, 0.5, 1.0, n_list)
    p4 = order_estimate(rk4, f_bai_6_17, 0.0, 1.0, 1.0, n_list)
    print(f"  + Bac hoi tu thuc nghiem RK3 (Bai 6.1) : {[f'{val:.6f}' for val in p3]} (Ly thuyet: 3.000000)")
    print(f"  + Bac hoi tu thuc nghiem RK4 (Bai 6.17): {[f'{val:.6f}' for val in p4]} (Ly thuyet: 4.000000)")
    assert np.isclose(p3[-1], 3.0, atol=0.1)
    assert np.isclose(p4[-1], 4.0, atol=0.1)


def test_order_estimate_r7_2_vi_du_nhom_tu_chon():
    """[R7 - Ca 2 cho order_estimate]: Vi du nhom tu chon (y' = y - x^2 + 1)."""
    print("\n--- [R7 - Ca 2 (order_estimate): Vi du nhom tu chon (y' = y - x^2 + 1)] ---")
    n_list = [10, 20, 40, 80]
    p3 = order_estimate(rk3, f_nhom_tu_chon_1, 0.0, 2.0, 0.5, n_list)
    p4 = order_estimate(rk4, f_nhom_tu_chon_1, 0.0, 2.0, 0.5, n_list)
    print(f"  + Bac hoi tu thuc nghiem RK3: {[f'{val:.6f}' for val in p3]}")
    print(f"  + Bac hoi tu thuc nghiem RK4: {[f'{val:.6f}' for val in p4]}")
    assert np.isclose(p3[-1], 3.0, atol=0.1)
    assert np.isclose(p4[-1], 4.0, atol=0.1)


def test_order_estimate_r7_3_ca_bien():
    """[R7 - Ca 3 cho order_estimate]: Cac ca bien."""
    print("\n--- [R7 - Ca 3 (order_estimate): Kiem tra cac ca bien] ---")
    with pytest.raises(ValueError, match="Can thoa man a < b"):
        order_estimate(rk4, f_bai_6_17, 1.0, 0.0, 1.0, [4, 8, 16])
    with pytest.raises(ValueError, match="n_list khong duoc rong"):
        order_estimate(rk4, f_bai_6_17, 0.0, 1.0, 1.0, [])
    with pytest.raises(ValueError, match="so nguyen duong"):
        order_estimate(rk3, f_bai_6_17, 0.0, 1.0, 1.0, [0, 2, 4])
    with pytest.raises(ValueError, match="tang gap doi"):
        order_estimate(rk4, f_bai_6_17, 0.0, 1.0, 1.0, [5, 10, 25])
    with pytest.raises(ArithmeticError, match="sai khac bang 0"):
        order_estimate(rk4, lambda x, y: 0.0, 0.0, 1.0, 5.0, [2, 4, 8])
    print("  + Da bat thanh cong toan bo 5 ca bien (a >= b, n_list rong, <= 0, khong nhan doi, sai khac = 0)!")


def test_step_for_tol_r7_1_vi_du_bai_giang():
    """[R7 - Ca 1 cho step_for_tol]: Bai 6.17 tren [0, 1] voi tol = 1e-6."""
    print("\n--- [R7 - Ca 1 (step_for_tol): Tim buoc luoi tren Bai 6.17 voi tol = 1e-6] ---")
    tol = 1e-6
    h3, yb3 = step_for_tol(rk3, f_bai_6_17, 0.0, 1.0, 1.0, tol)
    h4, yb4 = step_for_tol(rk4, f_bai_6_17, 0.0, 1.0, 1.0, tol)
    print(f"  + RK3: Buoc luoi chap nhan h = {h3:.6f} | y(1.0) = {yb3:.6f} | Nghiem dung = {exact_bai_6_17(1.0):.6f}")
    print(f"  + RK4: Buoc luoi chap nhan h = {h4:.6f} | y(1.0) = {yb4:.6f} | Nghiem dung = {exact_bai_6_17(1.0):.6f}")
    assert abs(yb3 - exact_bai_6_17(1.0)) < 10 * tol
    assert abs(yb4 - exact_bai_6_17(1.0)) < 10 * tol
    assert h4 >= h3


def test_step_for_tol_r7_2_vi_du_nhom_tu_chon():
    """[R7 - Ca 2 cho step_for_tol]: Vi du nhom tu chon (y' = y - x^2 + 1) voi tol = 1e-5."""
    print("\n--- [R7 - Ca 2 (step_for_tol): Vi du nhom tu chon voi tol = 1e-5] ---")
    tol = 1e-5
    h3, yb3 = step_for_tol(rk3, f_nhom_tu_chon_1, 0.0, 2.0, 0.5, tol)
    h4, yb4 = step_for_tol(rk4, f_nhom_tu_chon_1, 0.0, 2.0, 0.5, tol)
    print(f"  + RK3: Buoc luoi chap nhan h = {h3:.6f} | y(2.0) = {yb3:.6f} | Nghiem dung = {exact_nhom_tu_chon_1(2.0):.6f}")
    print(f"  + RK4: Buoc luoi chap nhan h = {h4:.6f} | y(2.0) = {yb4:.6f} | Nghiem dung = {exact_nhom_tu_chon_1(2.0):.6f}")
    assert abs(yb3 - exact_nhom_tu_chon_1(2.0)) < 10 * tol
    assert abs(yb4 - exact_nhom_tu_chon_1(2.0)) < 10 * tol
    assert h4 >= h3


def test_step_for_tol_r7_3_ca_bien():
    """[R7 - Ca 3 cho step_for_tol]: Cac ca bien (a >= b, tol <= 0, vuot max_n)."""
    print("\n--- [R7 - Ca 3 (step_for_tol): Kiem tra cac ca bien] ---")
    with pytest.raises(ValueError, match="Can thoa man a < b"):
        step_for_tol(rk4, f_bai_6_17, 2.0, 1.0, 1.0, 1e-4)
    with pytest.raises(ValueError, match="tol phai lon hon 0"):
        step_for_tol(rk3, f_bai_6_17, 0.0, 1.0, 1.0, 0.0)

    def non_converging_method(f, a, b, y0, n):
        return np.array([a, b]), np.array([y0, float(n)])

    with pytest.raises(RuntimeError, match="Khong tim duoc buoc luoi"):
        step_for_tol(non_converging_method, f_bai_6_17, 0.0, 1.0, 1.0, 1e-3)
    print("  + Da bat thanh cong toan bo 3 ca bien (a >= b, tol <= 0, vuot gioi han max_n)!")


if __name__ == "__main__":
    test_order_estimate_r7_1_vi_du_bai_giang()
    test_order_estimate_r7_2_vi_du_nhom_tu_chon()
    test_order_estimate_r7_3_ca_bien()
    test_step_for_tol_r7_1_vi_du_bai_giang()
    test_step_for_tol_r7_2_vi_du_nhom_tu_chon()
    test_step_for_tol_r7_3_ca_bien()
    print("\n=> PASSED: Toan bo kiem thu Phan 4 (order_estimate & step_for_tol) theo chuan R7 da chay thanh cong!")
