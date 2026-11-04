import numpy as np
import pytest

# IMPORT CHUAN CODE TU PHAN 1, PHAN 2 VA PHAN 3
from rk3 import rk3
from rk4 import rk4
from rk_table import rk_table


# ==============================================================================
# CAC HAM BAI TOAN MAU
# ==============================================================================
# 1. Bai toan Vi du 6.3 trong bai giang (Kiem thu bat buoc cho rk_table)
# y' = 3 * x^2 * y, y(0) = 1 tren [0, 1], n = 5 (h = 0.2)
def f_vi_du_6_3(x, y):
    return 3.0 * (x**2) * y

def exact_vi_du_6_3(x):
    return np.exp(x**3)

# 2. Bai toan KIEM THU BAT BUOC cua de (y' = y - x^2 + 1, y(0) = 0.5 tren [0, 2], h = 0.2, n = 10)
def f_bat_buoc(x, y):
    return y - x**2 + 1.0

def exact_bat_buoc(x):
    return (x + 1.0) ** 2 - 0.5 * np.exp(x)

# 3. Bai toan TU CHON (Khac hoan toan cac bai tren)
# y' = -2xy, y(0) = 1 tren [0, 1]. Nghiem dung: y = e^(-x^2)
def f_tu_chon(x, y):
    return -2.0 * x * y

def exact_tu_chon(x):
    return np.exp(-x**2)


# ==============================================================================
# KIEM THU PHAN 3 (rk_table)
# ==============================================================================
def test_rk_table_r7_1_vi_du_6_3():
    """[R7 - Ca 1]: Bang tinh tay cho Vi du 6.3 (Su dung RK4, n = 5) khop ban va A9-A10."""
    print("--- [R7 - Ca 1: Bang tinh tay Vi du 6.3 cho rk_table (RK4)] ---")
    tbl = rk_table(rk4, f_vi_du_6_3, 0.0, 1.0, 1.0, 5)
    assert len(tbl) == 5

    print(f"{'i':>2} | {'x_i':>5} | {'y_i':>10} | {'K1':>10} | {'K2':>10} | {'K3':>10} | {'K4':>10} | {'y_next':>10}")
    print("-" * 86)
    for row in tbl:
        print(f"{row['i']:2d} | {row['x_i']:5.2f} | {row['y_i']:10.6f} | {row['K1']:10.6f} | "
              f"{row['K2']:10.6f} | {row['K3']:10.6f} | {row['K4']:10.6f} | {row['y_next']:10.6f}")

    y_5 = tbl[-1]["y_next"]
    print(f"  + Bang tinh tay VD 6.3: y5 = {y_5:.6f} | De cho = 2.717587")
    assert np.isclose(y_5, 2.717587, atol=1e-5)


def test_rk_table_r7_2_kiem_thu_bat_buoc():
    """[R7 - Ca 2]: Bang tinh tay cho bai toan bat buoc (y' = y - x^2 + 1, n = 10)."""
    print("\n--- [R7 - Ca 2: Bang tinh tay bai toan bat buoc cho rk_table] ---")
    tbl_rk3 = rk_table(rk3, f_bat_buoc, 0.0, 2.0, 0.5, 10)
    tbl_rk4 = rk_table(rk4, f_bat_buoc, 0.0, 2.0, 0.5, 10)
    y_true = exact_bat_buoc(2.0)

    print(f"  + RK3 y_10 = {tbl_rk3[-1]['y_next']:.6f} (De yeu cau: 5.303725) | Sai so = {abs(tbl_rk3[-1]['y_next'] - y_true):.6f}")
    print(f"  + RK4 y_10 = {tbl_rk4[-1]['y_next']:.6f} (De yeu cau: 5.305363) | Sai so = {abs(tbl_rk4[-1]['y_next'] - y_true):.6f}")
    assert np.isclose(tbl_rk3[-1]["y_next"], 5.303725, atol=1e-5)
    assert np.isclose(tbl_rk4[-1]["y_next"], 5.305363, atol=1e-5)


def test_rk_table_r7_3_vi_du_nhom_tu_chon():
    """[R7 - Ca 3]: Bang tinh tay cho bai toan nhom tu chon (y' = -2xy)."""
    print("\n--- [R7 - Ca 3: Bang tinh tay nhom tu chon cho rk_table] ---")
    tbl = rk_table(rk3, f_tu_chon, 0.0, 1.0, 1.0, 10)
    y_true_end = exact_tu_chon(1.0)
    y_next_val = tbl[-1]["y_next"]

    print(f"  + Tai x = 1.0: y_next (RK3 table) = {y_next_val:.6f} | Nghiem dung = {y_true_61 if 'y_true_61' in locals() else y_true_end:.6f} | Sai so = {abs(y_next_val - y_true_end):.6f}")
    assert abs(y_next_val - y_true_end) < 1e-3
    print("  + Kiem tra bang tu chon rk_table: PASSED!")


def test_rk_table_r7_4_ca_bien():
    """[R7 - Ca 4]: Kiem tra cac ca biên (n <= 0, n sai kieu, a >= b, f hoac method khong hop le)."""
    print("\n--- [R7 - Ca 4: Kiem tra cac ca bien cho rk_table] ---")
    # 1. Bien n <= 0
    for bad_n in [0, -5]:
        with pytest.raises(ValueError):
            rk_table(rk4, f_vi_du_6_3, 0.0, 1.0, 1.0, bad_n)
    print("  + Bien n <= 0: Da bat dung ValueError!")

    # 2. Bien n sai kieu (float, str, bool, None)
    for bad_type in [2.5, "10", True, False, None]:
        with pytest.raises((TypeError, ValueError)):
            rk_table(rk3, f_vi_du_6_3, 0.0, 1.0, 1.0, bad_type)
    print("  + Bien n sai kieu: Da bat dung ngoai le!")

    # 3. Bien a >= b
    for bad_a, bad_b in [(1.0, 1.0), (2.0, 1.0)]:
        with pytest.raises(ValueError):
            rk_table(rk4, f_vi_du_6_3, bad_a, bad_b, 1.0, 5)
    print("  + Bien a >= b: Da bat dung ValueError!")

    # 4. Bien f va method khong hop le
    with pytest.raises(TypeError):
        rk_table(rk4, 123, 0.0, 1.0, 1.0, 5)
    with pytest.raises(ValueError):
        rk_table(lambda f, a, b, y0, n: (None, None), f_vi_du_6_3, 0.0, 1.0, 1.0, 5)
    print("  + Bien f va method khong hop le: Da bat dung TypeError & ValueError!")


if __name__ == "__main__":
    test_rk_table_r7_1_vi_du_6_3()
    test_rk_table_r7_2_kiem_thu_bat_buoc()
    test_rk_table_r7_3_vi_du_nhom_tu_chon()
    test_rk_table_r7_4_ca_bien()
    print("\n=> PASSED: Toan bo kiem thu Phan 3 (rk_table) theo chuan Nhom 8 da chay thanh cong!")