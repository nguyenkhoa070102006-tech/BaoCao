import numpy as np
import pytest

# IMPORT CHUAN CODE TU PHAN 1, PHAN 2 VA PHAN 3 (KHONG DINH NGHIA LAI HAM)
from rk3 import rk3
from rk4 import rk4
from rk_table import rk_table


# ==============================================================================
# CAC HAM BAI TOAN MAU CHO KIEM THU PHAN 3
# ==============================================================================
def f_bai_6_17(x, y):
    return 0.5 * x * y

def exact_bai_6_17(x):
    return np.exp(x**2 / 4.0)

def f_nhom_tu_chon_1(x, y):
    return y - x**2 + 1.0

def exact_nhom_tu_chon_1(x):
    return (x + 1.0) ** 2 - 0.5 * np.exp(x)


# ==============================================================================
# KIEM THU PHAN 3 
# ==============================================================================
def test_rk_table_r7_1_vi_du_bai_giang_6_17_cau_e():
    """[R7 - Ca 1]: Bai 6.17 cau e (Trang 142) - Tinh 3 buoc dau tien bang RK4 tren [0, 1], h = 0.1."""
    print("--- [R7 - Ca 1: Bang tinh tay 3 buoc dau Bai 6.17 cau e (RK4)] ---")
    tbl = rk_table(rk4, f_bai_6_17, 0.0, 1.0, 1.0, 10)
    print(f"{'i':>2} | {'x_i':>10} | {'y_i':>10} | {'K1':>10} | {'K2':>10} | {'K3':>10} | {'K4':>10} | {'y_next':>10}")
    print("-" * 91)
    for row in tbl[:3]:
        print(f"{row['i']:2d} | {row['x_i']:10.6f} | {row['y_i']:10.6f} | {row['K1']:10.6f} | "
              f"{row['K2']:10.6f} | {row['K3']:10.6f} | {row['K4']:10.6f} | {row['y_next']:10.6f}")

    print(f"  + Nghiem sau 3 buoc y_RK4(0.3) = {tbl[2]['y_next']:.6f} | Nghiem dung = {exact_bai_6_17(0.3):.6f}")
    assert len(tbl) == 10
    assert np.isclose(tbl[2]["y_next"], exact_bai_6_17(0.3), atol=1e-9)


def test_rk_table_r7_2_vi_du_nhom_tu_chon():
    """[R7 - Ca 2]: Vi du co nghiem dung do nhom tu chon (y' = y - x^2 + 1, y(0) = 0.5)."""
    print("\n--- [R7 - Ca 2: Bang tinh tay vi du nhom tu chon (table_rk3 & table_rk4)] ---")
    table_rk3 = rk_table(rk3, f_nhom_tu_chon_1, 0.0, 2.0, 0.5, 10)
    table_rk4 = rk_table(rk4, f_nhom_tu_chon_1, 0.0, 2.0, 0.5, 10)
    y_exact_2 = exact_nhom_tu_chon_1(2.0)

    print(f"  + Dong cuoi table_rk3 (x_next=2.0): y_next = {table_rk3[-1]['y_next']:.6f} | Nghiem dung = {y_exact_2:.6f} | Sai so = {abs(table_rk3[-1]['y_next'] - y_exact_2):.6f}")
    print(f"  + Dong cuoi table_rk4 (x_next=2.0): y_next = {table_rk4[-1]['y_next']:.6f} | Nghiem dung = {y_exact_2:.6f} | Sai so = {abs(table_rk4[-1]['y_next'] - y_exact_2):.6f}")
    assert abs(table_rk3[-1]["y_next"] - y_exact_2) < 1e-2
    assert abs(table_rk4[-1]["y_next"] - y_exact_2) < 2e-4


def test_rk_table_r7_3_ca_bien():
    """[R7 - Ca 3]: Ca bien (n <= 0, n sai kieu, a >= b, f hoac method khong hop le)."""
    print("\n--- [R7 - Ca 3: Kiem tra cac ca bien cho rk_table] ---")
    for bad_n in [0, -5]:
        with pytest.raises(ValueError, match="n phai lon hon 0"):
            rk_table(rk4, f_bai_6_17, 0.0, 1.0, 1.0, bad_n)
    print("  + Bien n <= 0: Da bat dung ValueError!")

    for bad_type in [2.5, "10", True, False, None]:
        with pytest.raises(TypeError, match="n phai la so nguyen"):
            rk_table(rk3, f_bai_6_17, 0.0, 1.0, 1.0, bad_type)
    print("  + Bien n sai kieu (float, str, bool, None): Da bat dung TypeError!")

    with pytest.raises(ValueError, match="Can thoa man a < b"):
        rk_table(rk4, f_bai_6_17, 2.0, 1.0, 1.0, 10)
    print("  + Bien a >= b: Da bat dung ValueError!")

    with pytest.raises(TypeError, match="f phai la mot ham"):
        rk_table(rk4, 123, 0.0, 1.0, 1.0, 10)
    with pytest.raises(ValueError, match="method phai la rk3 hoac rk4"):
        rk_table(lambda f, a, b, y0, n: (None, None), f_bai_6_17, 0.0, 1.0, 1.0, 10)
    print("  + Bien f va method khong hop le: Da bat dung TypeError & ValueError!")


if __name__ == "__main__":
    test_rk_table_r7_1_vi_du_bai_giang_6_17_cau_e()
    test_rk_table_r7_2_vi_du_nhom_tu_chon()
    test_rk_table_r7_3_ca_bien()
    print("\n=> PASSED: Toan bo kiem thu Phan 3 (rk_table) theo chuan R7 da chay thanh cong!")