import numpy as np
import pytest
from scipy.integrate import solve_ivp

from rk4 import rk4


# ==============================================================================
# CAC HAM BAI TOAN MAU
# ==============================================================================
# 1. Bai toan mau trong giao trinh (Bai 6.16)
# y' = 3y - 4*e^(-x), y(0) = 1 tren [0, 1] => Nghiem dung: y(x) = e^(-x)
def f_bai_6_16(x, y):
    return 3.0 * y - 4.0 * np.exp(-x)

def exact_bai_6_16(x):
    return np.exp(-x)

# 2. Bai toan KIEM THU BAT BUOC 1 (Theo anh de bai cua co)
# y' = y - x^2 + 1, y(0) = 0.5 tren [0, 2], h = 0.2 (n = 10) -> y10 = 5.305363
def f_bat_buoc_1(x, y):
    return y - x**2 + 1.0

def exact_bat_buoc_1(x):
    return (x + 1.0) ** 2 - 0.5 * np.exp(x)

# 3. Bai toan KIEM THU BAT BUOC 2 (Vi du 6.3 trong bai giang)
# y' = 3 * x^2 * y, y(0) = 1 tren [0, 1], n = 5 -> y5 = 2.717587
def f_bat_buoc_2(x, y):
    return 3.0 * (x**2) * y

def exact_bat_buoc_2(x):
    return np.exp(x**3)

# 4. Bai toan TU CHON (Khac hoan toan cac bai tren)
# y' = -2xy, y(0) = 1 tren [0, 1]. Nghiem dung: y = e^(-x^2)
def f_tu_chon(x, y):
    return -2.0 * x * y

def exact_tu_chon(x):
    return np.exp(-x**2)


# ==============================================================================
# KIEM THU PHAN 2 (RK4)
# ==============================================================================
def test_rk4_r7_1_vi_du_bai_giang():
    """[R7 - Ca 1]: Vi du trong bai giang (Bai 6.16)."""
    print("--- [R7 - Ca 1: Vi du giao trinh cho RK4 (Bai 6.16)] ---")
    _, y = rk4(f_bai_6_16, 0.0, 1.0, 1.0, 10)
    y_true = exact_bai_6_16(1.0)
    print(f"  + Bai 6.16 (x=1.0): y_RK4 = {y[-1]:.6f} | Nghiem dung = {y_true:.6f} | Sai so = {abs(y[-1] - y_true):.6f}")
    assert abs(y[-1] - y_true) < 2e-4


def test_rk4_r7_2_kiem_thu_bat_buoc_1():
    """[R7 - Ca 2.1]: Kiem thu bat buoc 1 (y' = y - x^2 + 1, h=0.2, n=10)."""
    print("\n--- [R7 - Ca 2.1: Kiem thu bat buoc 1 cua Co (RK4)] ---")
    _, y = rk4(f_bat_buoc_1, 0.0, 2.0, 0.5, 10)
    print(f"  + Tai x = 2.0 (n=10, h=0.2): y_RK4 = {y[-1]:.6f} (De yeu cau: 5.305363)")
    assert np.isclose(y[-1], 5.305363, atol=1e-5)


def test_rk4_r7_2_kiem_thu_bat_buoc_2():
    """[R7 - Ca 2.2]: Kiem thu bat buoc 2 (Vi du 6.3, y' = 3*x^2*y, n=5)."""
    print("\n--- [R7 - Ca 2.2: Kiem thu bat buoc 2 (Vi du 6.3 - RK4)] ---")
    _, y = rk4(f_bat_buoc_2, 0.0, 1.0, 1.0, 5)
    print(f"  + Vi du 6.3 (x=1.0, n=5): y_5 = {y[-1]:.6f} (De yeu cau: 2.717587)")
    assert np.isclose(y[-1], 2.717587, atol=1e-5)


def test_rk4_r7_3_vi_du_nhom_tu_chon():
    """[R7 - Ca 3]: Vi du nhom tu chon (y' = -2xy) + doi chieu scipy.integrate."""
    print("\n--- [R7 - Ca 3: Vi du nhom tu chon + Doi chieu scipy (n=10)] ---")
    a, b, y0, n = 0.0, 1.0, 1.0, 10
    x, y = rk4(f_tu_chon, a, b, y0, n)
    
    # 1. Kiem tra voi nghiem giai tich
    y_true = exact_tu_chon(x)
    print(f"  + Tai x = 1.0: y_RK4 = {y[-1]:.6f} | Nghiem dung = {y_true[-1]:.6f} | Sai so = {abs(y[-1] - y_true[-1]):.6f}")
    assert np.allclose(y, y_true, atol=1e-4)
    
    # 2. Kiem tra voi thu vien scipy
    sol = solve_ivp(f_tu_chon, (x[0], x[-1]), [y0], t_eval=x, rtol=1e-11, atol=1e-13)
    assert np.allclose(y, sol.y[0], atol=1e-4)
    print("  + Doi chieu voi scipy: PASSED!")


def test_rk4_r7_4_ca_bien():
    """[R7 - Ca 4]: Kiem tra cac ca bien (da thuc bac 4, a >= b, n <= 0)."""
    print("\n--- [R7 - Ca 4: Kiem tra cac ca bien cho RK4] ---")
    # 1. Bien da thuc bac 4 (sai so tuyet đối bang 0)
    _, y_poly = rk4(lambda x, y: 4.0 * x**3, 0.0, 2.0, 0.0, 4)
    assert np.isclose(y_poly[-1], 16.0, atol=1e-12)
    print("  + Bien da thuc bac 4: PASSED!")

    # 2. Bien khoang tinh khong hop le a >= b
    for bad_a, bad_b in [(1.0, 1.0), (2.0, 1.0)]:
        with pytest.raises(ValueError):
            rk4(f_bai_6_16, bad_a, bad_b, 1.0, 10)
    print("  + Bien a >= b: Da bat dung ValueError!")

    # 3. Bien so buoc n <= 0 hoac sai kieu
    for bad_n in [0, -5, 2.5, True, False]:
        with pytest.raises(ValueError):
            rk4(f_bai_6_16, 0.0, 1.0, 1.0, bad_n)
    print("  + Bien n <= 0 hoac sai kieu: Da bat dung ValueError!")


if __name__ == "__main__":
    test_rk4_r7_1_vi_du_bai_giang()
    test_rk4_r7_2_kiem_thu_bat_buoc_1()
    test_rk4_r7_2_kiem_thu_bat_buoc_2()
    test_rk4_r7_3_vi_du_nhom_tu_chon()
    test_rk4_r7_4_ca_bien()
    print("\n=> PASSED: Toan bo kiem thu Phan 2 (rk4) da chay thanh cong!")