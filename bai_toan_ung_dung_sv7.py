"""
Học phần: Phương pháp số
Bài tập lớn - Nhóm 8: Runge-Kutta bậc 3 và Runge-Kutta bậc 4 (Chương 6)
Phần: Bài toán ứng dụng (Phụ trách: SV7)
Đề tài: Mô hình vật rơi tự do có lực cản không khí

Mô hình: v'(t) = g - (k/m) v^2, v(0) = 0.
Nghiệm giải tích: v(t) = v_inf * tanh(g t / v_inf), v_inf = sqrt(g / (k/m)).

Lưu ý R3: file này KHÔNG dùng numpy.linalg / scipy cho phần tính toán.
np.tanh / np.arctanh chỉ dùng để tính nghiệm giải tích đối chiếu.
"""

import warnings
from typing import Callable, Tuple

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
N_DEMO: int = 20              # số bước dùng để minh họa đồ thị

# Vận tốc giới hạn v_inf = sqrt(g / (k/m))
V_INF: float = float(np.sqrt(G_CONST / K_OVER_M))


def f_falling_body(t: float, v: float) -> float:
    """Vế phải của PTVP vật rơi có lực cản bình phương vận tốc.

    Mô tả: v'(t) = g - (k/m) * v^2.
    Require: t, v là số thực (t không xuất hiện trong công thức - hệ tự trị).
    Ensure: trả về số thực v'(t).
    Thuật toán: mô hình hóa (không phải thuật toán bài giảng).
    """
    return G_CONST - K_OVER_M * (v ** 2)


def v_exact(t):
    """Nghiệm giải tích chính xác của bài toán giá trị đầu v(0) = 0.

    Mô tả: v(t) = v_inf * tanh(g t / v_inf).
    Require: t là số thực hoặc mảng numpy các số thực.
    Ensure: trả về v(t) cùng kích thước với t, dùng để đối chiếu sai số.
    Thuật toán: nghiệm giải tích (đối chiếu, không phải phần lõi).
    """
    return V_INF * np.tanh(G_CONST * t / V_INF)


def count_calls(f: Callable) -> Tuple[Callable, dict]:
    """Bọc hàm f để đếm số lần được gọi.

    Require: f là hàm hai biến f(t, v).
    Ensure: trả về (f_bọc, bộ_đếm) với bộ_đếm["n"] là số lần f_bọc đã gọi.
    Thuật toán: tiện ích đo chi phí (không phải thuật toán bài giảng).
    """
    counter = {"n": 0}

    def wrapped(t, v):
        counter["n"] += 1
        return f(t, v)

    return wrapped, counter


# 2. Xác định thời điểm đạt tỷ lệ vận tốc giới hạn
def find_time_for_ratio(
    method: Callable,
    f: Callable,
    t_grid: np.ndarray,
    v_grid: np.ndarray,
    v_limit: float,
    ratio: float = 0.95,
    tol_t: float = 1e-12,
    max_iter: int = 100,
) -> float:
    """Tìm t sao cho v(t) = ratio * v_limit.

    Mô tả: tìm khoảng lưới [t_{i-1}, t_i] đầu tiên có v_i >= ngưỡng, rồi chia đôi
    theo bước con s in [0, h]; mỗi lần thử dùng ĐÚNG MỘT bước của chính phương pháp
    RK (method) nên sai số chỉ do phương pháp RK, không bị nhiễm sai số nội suy
    tuyến tính (cỡ h^2, lớn hơn sai số RK).
    Require: method(f, a, b, y0, n) -> (t, y) (rk3/rk4); t_grid tăng ngặt, cùng
        kích thước với v_grid, >= 2 điểm; v_limit > 0; 0 < ratio < 1; v tăng
        trong khoảng chứa ngưỡng.
    Ensure: trả về t (float) với |v(t) - ratio*v_limit| xấp xỉ <= tol_t theo t.
    Thuật toán: Chia đôi (Chương 3) kết hợp một bước RK (Chương 6).
    Raises: ValueError nếu đầu vào sai; RuntimeError nếu v chưa đạt ngưỡng.
    """
    if not (0.0 < ratio < 1.0):
        raise ValueError("Tỷ lệ ratio phải nằm trong khoảng (0, 1).")
    if v_limit <= 0.0:
        raise ValueError("v_limit phải dương.")
    t_grid = np.asarray(t_grid, dtype=float)
    v_grid = np.asarray(v_grid, dtype=float)
    if t_grid.ndim != 1 or t_grid.shape != v_grid.shape or len(t_grid) < 2:
        raise ValueError("t_grid và v_grid phải là mảng 1 chiều cùng kích thước, >= 2 điểm.")
    if not np.all(np.diff(t_grid) > 0.0):
        raise ValueError("t_grid phải tăng ngặt.")
    if not (np.all(np.isfinite(t_grid)) and np.all(np.isfinite(v_grid))):
        raise ValueError("Lưới chứa giá trị NaN hoặc vô hạn.")

    v_target = ratio * v_limit
    indices = np.where(v_grid >= v_target)[0]
    if len(indices) == 0:
        raise RuntimeError(f"Vận tốc trên lưới chưa đạt tới ngưỡng {ratio * 100:.1f}%.")

    idx = int(indices[0])
    if idx == 0:
        return float(t_grid[0])  # điều kiện đầu đã đạt/vượt ngưỡng

    t_lo = float(t_grid[idx - 1])
    v_lo = float(v_grid[idx - 1])
    lo, hi = 0.0, float(t_grid[idx] - t_grid[idx - 1])

    for _ in range(max_iter):               # chia đôi theo bước con s
        if hi - lo < tol_t:
            break
        mid = 0.5 * (lo + hi)
        _, v_mid = method(f, t_lo, t_lo + mid, v_lo, 1)   # một bước RK
        if v_mid[-1] < v_target:
            lo = mid
        else:
            hi = mid
    else:
        warnings.warn("find_time_for_ratio: hết max_iter mà chưa đạt tol_t.")

    return float(t_lo + 0.5 * (lo + hi))


# 3. Đồ thị
def plot_application_results(t_95_exact: float, filename: str = "ung_dung_vat_roi_nhom8.png") -> None:
    """Vẽ đồ thị vận tốc và sai số tuyệt đối (minh họa với N_DEMO bước).

    Require: t_95_exact là thời điểm đạt 95% v_inf (giải tích); filename kết thúc .png.
    Ensure: ghi file ảnh vào filename; không trả về giá trị.
    Thuật toán: không (trình bày kết quả).
    """
    h_demo = (T_END - T_START) / N_DEMO
    t_cont = np.linspace(T_START, T_END, 400)
    v_cont = v_exact(t_cont)

    _, v_rk3_demo = rk3(f_falling_body, T_START, T_END, V0, N_DEMO)
    t_demo, v_rk4_demo = rk4(f_falling_body, T_START, T_END, V0, N_DEMO)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))

    ax1.plot(t_cont, v_cont, "k-", linewidth=2.0, label="Nghiệm đúng $v(t)$")
    ax1.plot(t_demo, v_rk3_demo, "bs--", markersize=4, label=f"RK3 ($n={N_DEMO}, h={h_demo:g}$ s)")
    ax1.plot(t_demo, v_rk4_demo, "ro:", markersize=4, label=f"RK4 ($n={N_DEMO}, h={h_demo:g}$ s)")
    ax1.axhline(V_INF, color="purple", linestyle="--", alpha=0.8, label=f"$v_\\infty = {V_INF:.2f}$ m/s")
    ax1.axhline(0.95 * V_INF, color="forestgreen", linestyle="-.", alpha=0.8,
                label=f"$0.95 v_\\infty = {0.95 * V_INF:.2f}$ m/s")
    ax1.axvline(t_95_exact, color="forestgreen", linestyle=":", alpha=0.8,
                label=f"$t_{{95\\%}} = {t_95_exact:.2f}$ s")
    ax1.set_title("Vận tốc vật rơi theo thời gian", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Thời gian $t$ (s)")
    ax1.set_ylabel("Vận tốc $v(t)$ (m/s)")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right", fontsize=8.5)

    floor = 1e-18  # tránh log(0)
    err_rk3 = np.maximum(np.abs(v_exact(t_demo) - v_rk3_demo), floor)
    err_rk4 = np.maximum(np.abs(v_exact(t_demo) - v_rk4_demo), floor)
    ax2.semilogy(t_demo[1:], err_rk3[1:], "b-s", markersize=5, label=f"Sai số RK3 ($h = {h_demo:g}$ s)")
    ax2.semilogy(t_demo[1:], err_rk4[1:], "r-o", markersize=5, label=f"Sai số RK4 ($h = {h_demo:g}$ s)")
    ax2.set_title("Sai số tuyệt đối $|v(t_i) - v_i|$ (thang logarit)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Thời gian $t$ (s)")
    ax2.set_ylabel("Sai số tuyệt đối (log scale)")
    ax2.grid(True, which="both", linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)


# 4. Chương trình thực nghiệm và xuất kết quả
def run_application() -> None:
    """Chạy toàn bộ bài toán ứng dụng và in kết quả.

    Require: các module rk3, rk4, uocluong (P4) import được.
    Ensure: in bảng so sánh RK3/RK4 (bước h, số bước, số lần gọi f đếm thực tế,
        sai số thực tế), thời điểm đạt 95% v_inf, và ghi file đồ thị .png.
    Thuật toán: RK3, RK4 (Chương 6); step_for_tol (phần P4 của nhóm).
    """
    print("=" * 80)
    print("BÁO CÁO BÀI TOÁN ỨNG DỤNG (NHÓM 8 - SV7): VẬT RƠI CÓ LỰC CẢN KHÔNG KHÍ")
    print("=" * 80)
    print(f"Hằng số mô hình: g = {G_CONST} m/s^2, k/m = {K_OVER_M} m^-1, Khoảng xét: [{T_START:g}, {T_END:g}] s")
    print(f"Vận tốc giới hạn v_inf = sqrt(g / (k/m)) = {V_INF:.6f} m/s")
    print(f"Ngưỡng 95% v_inf = {0.95 * V_INF:.6f} m/s")
    print(f"Nghiệm giải tích tại t = {T_END:g}s: v_exact = {v_exact(T_END):.10f} m/s\n")

    # Nhiệm vụ 1: Tìm bước h để sai số tại t = T_END nhỏ hơn tol
    tol = 1e-6
    print(f"--- 1. TÌM BƯỚC h ĐỂ SAI SỐ TẠI t={T_END:g} NHỎ HƠN TOL = {tol:.0e} (DÙNG P4) ---")

    rows = {}
    for name, method in (("RK3", rk3), ("RK4", rk4)):
        try:
            h, _ = step_for_tol(method, f_falling_body, T_START, T_END, V0, tol)
        except Exception as exc:  # R6: báo lỗi rõ ràng thay vì dừng đột ngột
            raise RuntimeError(f"step_for_tol thất bại với {name}: {exc}") from exc
        n = int(round((T_END - T_START) / h))
        f_counted, counter = count_calls(f_falling_body)
        t_g, v_g = method(f_counted, T_START, T_END, V0, n)   # chạy lại với n bước, đếm f
        rows[name] = dict(h=h, n=n, calls=counter["n"], v_end=float(v_g[-1]),
                          err=abs(float(v_exact(T_END)) - float(v_g[-1])), t=t_g, v=v_g)
        if rows[name]["err"] >= tol:
            print(f"  [Cảnh báo] {name}: sai số thực tế {rows[name]['err']:.2e} >= tol; "
                  f"P4 ước lượng sai số bằng h và h/2 nên có thể chưa đạt.")

    fmt = "{:<12} | {:<12} | {:<10} | {:<14} | {:<14} | {:<12}"
    print(fmt.format("Phương pháp", "Bước h (s)", "Số bước n", "Số lần gọi f", "Sai số thực tế", "v(T) xấp xỉ"))
    print("-" * 88)
    for name, r in rows.items():
        print(fmt.format(name, f"{r['h']:.6f}", str(r["n"]), str(r["calls"]), f"{r['err']:.4e}", f"{r['v_end']:.8f}"))
    print("-" * 88)
    r3, r4 = rows["RK3"], rows["RK4"]
    ratio_calls = r4["calls"] / r3["calls"] * 100.0
    if r4["calls"] < r3["calls"]:
        verdict = f"RK4 rẻ hơn: số lần gọi f chỉ bằng ~{ratio_calls:.1f}% so với RK3"
    elif r4["calls"] > r3["calls"]:
        verdict = f"RK3 rẻ hơn: RK4 tốn ~{ratio_calls:.1f}% số lần gọi f của RK3"
    else:
        verdict = "hai phương pháp tốn số lần gọi f bằng nhau"
    print(f"Nhận xét: h(RK3) = {r3['h']:.4f}s, h(RK4) = {r4['h']:.4f}s; {verdict} "
          f"(với cùng ngưỡng sai số {tol:.0e}).\n")

    # Nhiệm vụ 2: Tìm thời điểm đạt 95% v_inf
    print("--- 2. XÁC ĐỊNH THỜI ĐIỂM VẬN TỐC ĐẠT 95% VẬN TỐC GIỚI HẠN (v = 0.95 * v_inf) ---")
    t_95_exact = (V_INF / G_CONST) * float(np.arctanh(0.95))
    print(f"- Nghiệm giải tích chính xác: t_95 = {t_95_exact:.8f} giây")
    for name, method in (("RK3", rk3), ("RK4", rk4)):
        r = rows[name]
        t95 = find_time_for_ratio(method, f_falling_body, r["t"], r["v"], V_INF, 0.95)
        print(f"- Xác định từ {name}:            t_95 = {t95:.8f} giây (lệch: {abs(t95 - t_95_exact):.2e} s)")
    print()

    # Nhiệm vụ 3: Đồ thị
    plot_application_results(t_95_exact)
    print("-> Đã xuất đồ thị vào file: 'ung_dung_vat_roi_nhom8.png'")


if __name__ == "__main__":
    run_application()
