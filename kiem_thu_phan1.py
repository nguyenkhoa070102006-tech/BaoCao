import numpy as np
import pytest
from scipy.integrate import solve_ivp

# ==============================================================================
# Phan 1: ham thuat toan rk3
# ==============================================================================
def rk3(f, a, b, y0, n) -> tuple[np.ndarray, np.ndarray]:  
    h = (b - a) / n
    x = [a]
    y = [y0]
    # Cong thuc RK3: ( trong giao trinh bai (6.20))
    # K1 = h * f(x, y)
    # K2 = h * f(x + h/2, y + K1/2)
    # K3 = h * f(x + h, y - K1 + 2*K2)
    # y_{n+1} = y_n + 1/6*(K1 + 4*K2 + K3)
    
    K1 = [h * f(a, y0)]
    K2 = [h * f(a + h/2, y0 + K1[-1]/2)]
    K3 = [h * f(a + h, y0 - K1[-1] + 2 * K2[-1])]
    
    for i in range(n):  
        x.append(x[-1] + h)
        y.append(y[-1] + 1/6 * (K1[-1] + 4 * K2[-1] + K3[-1]))
        
        K1.append(h * f(x[-1], y[-1]))
        K2.append(h * f(x[-1] + h/2, y[-1] + K1[-1]/2))
        K3.append(h * f(x[-1] + h, y[-1] - K1[-1] + 2 * K2[-1]))
        
    return np.array(x), np.array(y)


# ==============================================================================
# Kiem thu phan 1
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


def test_rk3_r7_1_vi_du_bai_giang():
    """[R7 - Ca 1]: Ví dụ trong bài giảng/giáo trình (Bài 6.1, 6.2, 6.17)."""
    print("--- [R7 - Ca 1: Ví dụ giáo trình cho RK3] ---")
    x_61, y_61 = rk3(f_bai_6_1, 0.0, 0.5, 1.0, 5)
    y_true_61 = exact_bai_6_1(0.5)
    print(f"  + Bài 6.1 (x=0.5): y_RK3 = {y_61[-1]:.8f} | Nghiệm đúng = {y_true_61:.8f} | Sai số = {abs(y_61[-1] - y_true_61):.2e}")
    assert np.allclose(y_61, exact_bai_6_1(x_61), atol=5e-4)

    x_62, y_62 = rk3(f_bai_6_2, 0.0, 0.03, 1.0, 3)
    y_true_62 = exact_bai_6_2(0.03)
    print(f"  + Bài 6.2 (x=0.03): y_RK3 = {y_62[-1]:.8f} | Nghiệm đúng = {y_true_62:.8f} | Sai số = {abs(y_62[-1] - y_true_62):.2e}")
    assert np.allclose(y_62, exact_bai_6_2(x_62), atol=1e-6)

    _, y_617 = rk3(f_bai_6_17, 0.0, 1.0, 1.0, 10)
    y_true_617 = exact_bai_6_17(1.0)
    print(f"  + Bài 6.17 (x=1.0): y_RK3 = {y_617[-1]:.8f} | Nghiệm đúng = {y_true_617:.8f} | Sai số = {abs(y_617[-1] - y_true_617):.2e}")
    assert abs(y_617[-1] - y_true_617) < 5e-5


def test_rk3_r7_2_vi_du_nhom_tu_chon():
    """[R7 - Ca 2]: Ví dụ có nghiệm đúng do nhóm tự chọn + đối chiếu scipy.integrate."""
    print("\n--- [R7 - Ca 2: Ví dụ nhóm tự chọn cho RK3 (y' = y - x^2 + 1)] ---")
    a, b, y0, n = 0.0, 2.0, 0.5, 20
    x, y = rk3(f_nhom_tu_chon_1, a, b, y0, n)
    y_true = exact_nhom_tu_chon_1(x)
    sol = solve_ivp(f_nhom_tu_chon_1, (x[0], x[-1]), [y0], t_eval=x, rtol=1e-10, atol=1e-12)

    print(f"  + Tại x = 2.0: y_RK3 = {y[-1]:.8f} | Nghiệm giải tích = {y_true[-1]:.8f} | Scipy = {sol.y[0][-1]:.8f}")
    print(f"  + Sai số lớn nhất trên toàn lưới: {np.max(np.abs(y - y_true)):.2e}")
    assert np.allclose(y, y_true, atol=2e-3)
    assert np.allclose(y, sol.y[0], atol=2e-3)


def test_rk3_r7_3_ca_bien():
    """[R7 - Ca 3]: Ca biên (n = 1, đa thức bậc 3 sai số = 0, n = 0)."""
    print("\n--- [R7 - Ca 3: Kiểm tra các ca biên cho RK3] ---")
    a, b, y0, n = 0.0, 0.1, 1.0, 1
    h = b - a
    K1 = h * f_bai_6_1(a, y0)
    K2 = h * f_bai_6_1(a + h / 2, y0 + K1 / 2)
    K3 = h * f_bai_6_1(a + h, y0 - K1 + 2 * K2)
    expected_y1 = y0 + (K1 + 4 * K2 + K3) / 6.0
    _, y_step1 = rk3(f_bai_6_1, a, b, y0, n)
    print(f"  + Biên n = 1 bước: y_RK3(0.1) = {y_step1[-1]:.8f} == Công thức tay = {expected_y1:.8f}")
    assert np.isclose(y_step1[-1], expected_y1, rtol=1e-12)

    _, y_poly = rk3(lambda x, y: 3.0 * x**2, 0.0, 2.0, 0.0, 4)
    print(f"  + Biên đa thức bậc 3 (y' = 3x^2 tại x=2): y_RK3 = {y_poly[-1]:.12f} (Nghiệm đúng = 8.0)")
    assert np.isclose(y_poly[-1], 8.0, atol=1e-12)

    with pytest.raises(ZeroDivisionError):
        rk3(f_bai_6_17, 0.0, 1.0, 1.0, 0)
    print("  + Biên n = 0: Đã bắt đúng lỗi ZeroDivisionError!")


# Chay kiem thu phan 1
test_rk3_r7_1_vi_du_bai_giang()
test_rk3_r7_2_vi_du_nhom_tu_chon()
test_rk3_r7_3_ca_bien()
print("\n=> PASSED: Toàn bộ kiểm thử Phần 1 (rk3) theo chuẩn R7 đã chạy thành công!")