"""
Học phần: Phương pháp số
Bài tập lớn - Nhóm 8: Runge-Kutta bậc 3 và Runge-Kutta bậc 4 (Chương 6)
Phần: Bài toán ứng dụng (Phụ trách: SV7)
Đề tài: Mô hình vật rơi tự do có lực cản không khí
"""

import numpy as np
import matplotlib.pyplot as plt

from rk3 import rk3
from rk4 import rk4
from uocluong import step_for_tol


# 1. Hằng số mô hình và các hàm cơ bản
G_CONST: float = 9.81         # g (m/s^2)
K_OVER_M: float = 0.004       # k/m (1/m)
T_START: float = 0.0
T_END: float = 20.0
V0: float = 0.0

# Vận tốc giới hạn v_inf = sqrt(g / (k/m))
V_INF: float = float(np.sqrt(G_CONST / K_OVER_M))


def f_falling_body(t: float, v: float) -> float:
    """PTVP: v'(t) = g - (k/m) * v^2"""
    return G_CONST - K_OVER_M * (v ** 2)


def v_exact(t: float | np.ndarray) -> float | np.ndarray:
    """Nghiệm giải tích chính xác."""
    return V_INF * np.tanh(G_CONST * t / V_INF)


# 2. Xác định thời điểm đạt tỷ lệ vận tốc giới hạn
def find_time_for_ratio(
    t_grid: np.ndarray,
    v_grid: np.ndarray,
    v_limit: float,
    ratio: float = 0.95
) -> float:
    """Tìm thời điểm t để v(t) = ratio * v_limit bằng nội suy tuyến tính."""
    if not (0.0 < ratio < 1.0):
        raise ValueError("Tỷ lệ ratio phải nằm trong khoảng (0, 1).")
    if len(t_grid) != len(v_grid) or len(t_grid) < 2:
        raise ValueError("Lưới t_grid và v_grid phải cùng kích thước và >= 2 điểm.")

    v_target = ratio * v_limit

    indices = np.where(v_grid >= v_target)[0]
    if len(indices) == 0:
        raise RuntimeError(f"Vận tốc trên lưới chưa đạt tới ngưỡng {ratio*100}%.")

    idx = int(indices[0])
    if idx == 0:
        return float(t_grid[0])

    # Nội suy tuyến tính giữa idx-1 và idx
    t1, t2 = t_grid[idx - 1], t_grid[idx]
    v1, v2 = v_grid[idx - 1], v_grid[idx]

    t_interp = t1 + (v_target - v1) / (v2 - v1) * (t2 - t1)
    return float(t_interp)


# 3. Chương trình thực nghiệm và xuất kết quả
def run_application() -> None:
    print("=" * 80)
    print("BÁO CÁO BÀI TOÁN ỨNG DỤNG (NHÓM 8 - SV7): VẬT RƠI CÓ LỰC CẢN KHÔNG KHÍ")
    print("=" * 80)
    print(f"Hằng số mô hình: g = {G_CONST} m/s^2, k/m = {K_OVER_M} m^-1, Khoảng xét: [0, 20] s")
    print(f"Vận tốc giới hạn v_inf = sqrt(g / (k/m)) = {V_INF:.6f} m/s")
    print(f"Ngưỡng 95% v_inf = {0.95 * V_INF:.6f} m/s")
    print(f"Nghiệm giải tích tại t = 20s: v_exact(20) = {v_exact(T_END):.10f} m/s\n")

    # Nhiệm vụ 1: Tìm bước h để sai số tại t=20 < 10^-6
    tol = 1e-6
    print(f"--- 1. TÌM BƯỚC h ĐỂ SAI KHÁC TẠI t=20 NHỎ HƠN TOL = {tol:.0e} (DÙNG P4) ---")

    h_rk3, v20_rk3 = step_for_tol(rk3, f_falling_body, T_START, T_END, V0, tol)
    n_rk3 = int(round((T_END - T_START) / h_rk3))
    err_true_rk3 = abs(v_exact(T_END) - v20_rk3)
    evals_rk3 = 3 * n_rk3

    h_rk4, v20_rk4 = step_for_tol(rk4, f_falling_body, T_START, T_END, V0, tol)
    n_rk4 = int(round((T_END - T_START) / h_rk4))
    err_true_rk4 = abs(v_exact(T_END) - v20_rk4)
    evals_rk4 = 4 * n_rk4

    fmt_table = "{:<12} | {:<12} | {:<10} | {:<16} | {:<14} | {:<12}"
    print(fmt_table.format("Phương pháp", "Bước h (s)", "Số bước n", "Số lần gọi f", "Sai số thực tế", "v(20) xấp xỉ"))
    print("-" * 88)
    print(fmt_table.format("RK3", f"{h_rk3:.6f}", str(n_rk3), str(evals_rk3), f"{err_true_rk3:.4e}", f"{v20_rk3:.8f}"))
    print(fmt_table.format("RK4", f"{h_rk4:.6f}", str(n_rk4), str(evals_rk4), f"{err_true_rk4:.4e}", f"{v20_rk4:.8f}"))
    print("-" * 88)
    print(f"Nhận xét: RK4 cần bước h lớn hơn ({h_rk4:.4f}s vs {h_rk3:.4f}s), số bước n ít hơn "
          f"và tổng số lần tính đạo hàm f chỉ bằng ~{(evals_rk4/evals_rk3)*100:.1f}% so với RK3.\n")

    # Nhiệm vụ 2: Tìm thời điểm đạt 95% v_inf
    print("--- 2. XÁC ĐỊNH THỜI ĐIỂM VẬN TỐC ĐẠT 95% VẬN TỐC GIỚI HẠN (v = 0.95 * v_inf) ---")
    t_95_exact = (V_INF / G_CONST) * float(np.arctanh(0.95))

    t_grid_rk3, v_grid_rk3 = rk3(f_falling_body, T_START, T_END, V0, n_rk3)
    t_grid_rk4, v_grid_rk4 = rk4(f_falling_body, T_START, T_END, V0, n_rk4)

    t_95_rk3 = find_time_for_ratio(t_grid_rk3, v_grid_rk3, V_INF, 0.95)
    t_95_rk4 = find_time_for_ratio(t_grid_rk4, v_grid_rk4, V_INF, 0.95)

    print(f"- Nghiệm giải tích chính xác: t_95 = {t_95_exact:.6f} giây")
    print(f"- Xác định từ RK3:            t_95 = {t_95_rk3:.6f} giây (lệch: {abs(t_95_rk3 - t_95_exact):.2e} s)")
    print(f"- Xác định từ RK4:            t_95 = {t_95_rk4:.6f} giây (lệch: {abs(t_95_rk4 - t_95_exact):.2e} s)\n")

    # Nhiệm vụ 3: Vẽ đồ thị nghiệm và sai số (minh họa với n = 20)
    t_cont = np.linspace(T_START, T_END, 400)
    v_cont = v_exact(t_cont)

    _, v_rk3_demo = rk3(f_falling_body, T_START, T_END, V0, 20)
    t_demo, v_rk4_demo = rk4(f_falling_body, T_START, T_END, V0, 20)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))

    # Đồ thị 1: Vận tốc
    ax1.plot(t_cont, v_cont, "k-", linewidth=2.0, label="Nghiệm đúng $v(t)$")
    ax1.plot(t_demo, v_rk3_demo, "bs--", markersize=4, label="RK3 ($n=20, h=1.0$s)")
    ax1.plot(t_demo, v_rk4_demo, "ro:", markersize=4, label="RK4 ($n=20, h=1.0$s)")
    ax1.axhline(V_INF, color="purple", linestyle="--", alpha=0.8, label=f"$v_\\infty = {V_INF:.2f}$ m/s")
    ax1.axhline(0.95 * V_INF, color="forestgreen", linestyle="-.", alpha=0.8, label=f"$0.95 v_\\infty = {0.95*V_INF:.2f}$ m/s")
    ax1.axvline(t_95_exact, color="forestgreen", linestyle=":", alpha=0.8, label=f"$t_{{95\\%}} = {t_95_exact:.2f}$ s")

    ax1.set_title("Vận tốc vật rơi theo thời gian", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Thời gian $t$ (s)")
    ax1.set_ylabel("Vận tốc $v(t)$ (m/s)")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right", fontsize=8.5)

    # Đồ thị 2: Sai số tuyệt đối (log scale)
    err_rk3_demo = np.abs(v_exact(t_demo) - v_rk3_demo)
    err_rk4_demo = np.abs(v_exact(t_demo) - v_rk4_demo)

    ax2.semilogy(t_demo[1:], err_rk3_demo[1:], "b-s", markersize=5, label="Sai số RK3 ($h = 1.0$ s)")
    ax2.semilogy(t_demo[1:], err_rk4_demo[1:], "r-o", markersize=5, label="Sai số RK4 ($h = 1.0$ s)")
    ax2.set_title("Sai số tuyệt đối $|v(t_i) - v_i|$ (thang đo logarit)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Thời gian $t$ (s)")
    ax2.set_ylabel("Sai số tuyệt đối (log scale)")
    ax2.grid(True, which="both", linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig("ung_dung_vat_roi_nhom8.png", dpi=300)
    print("-> Đã xuất đồ thị chất lượng cao vào file: 'ung_dung_vat_roi_nhom8.png'")


if __name__ == "__main__":
    run_application()
