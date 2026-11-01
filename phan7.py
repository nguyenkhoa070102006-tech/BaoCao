"""
PHẦN BÀI TOÁN ỨNG DỤNG
Đề tài: Giải bài toán truyền nhiệt của chi tiết máy trong buồng làm mát
(Định luật làm nguội Newton với nhiệt độ môi trường biến thiên tuần hoàn)
bằng phương pháp Runge-Kutta bậc 3 và bậc 4.

Mô hình vật lý và giả thiết tham số:
- Xét một chi tiết máy bằng kim loại (hệ số dẫn nhiệt lambda lớn, kích thước
  đặc trưng L_c nhỏ) thỏa mãn điều kiện thông số tập trung (Lumped Capacitance
  Model) với số Biot Bi = (h_conv * L_c) / lambda < 0.1. Khi đó, gradien nhiệt độ
  bên trong vật thể không đáng kể và nhiệt độ chi tiết máy T(t) chỉ phụ thuộc
  vào thời gian t, với nhiệt độ ban đầu T(0) = T0 = 100 °C.
- Theo định luật làm nguội Newton, tốc độ giảm nhiệt độ của chi tiết máy tỉ lệ
  thuận với độ chênh lệch nhiệt độ giữa chi tiết máy T(t) và môi trường buồng
  làm mát T_env(t):
      dT/dt = -k * (T(t) - T_env(t)),   t in [a, b] (phút).
- Do chu kỳ đóng/ngắt của máy nén điều hòa, nhiệt độ trong buồng làm mát dao động
  tuần hoàn quanh mức nền T_base = 25 °C với biên độ amp = 5 °C và tần số góc
  omega = 1.0 rad/phút (chu kỳ T_p = 2*pi ≈ 6.28 phút):
      T_env(t) = T_base - amp * cos(omega * t)  (°C).
- Hệ số truyền nhiệt bề mặt chuẩn k = 0.2 (1/phút) đặc trưng cho quá trình làm
  mát đối lưu cưỡng bức nhẹ trong không khí. Trong phần khảo sát giới hạn ổn
  định tuyệt đối, k được mở rộng lên kịch bản tôi kim loại trong chất lỏng/dầu
  cưỡng bức mạnh (k in [5.0, 10.0] 1/phút, hằng số thời gian vài giây) nhằm
  kiểm chứng hành vi của RK3 và RK4 khi tích số k*h vượt ngưỡng ổn định lý thuyết.
"""

import numpy as np
import matplotlib.pyplot as plt

# Bọc import scipy an toàn theo R1, R3: nếu môi trường chấm chỉ cài numpy,
# matplotlib, pytest mà không có scipy thì chương trình và kiểm thử vẫn chạy tốt.
try:
    from scipy.integrate import solve_ivp
    HAS_SCIPY = True
except ImportError:
    solve_ivp = None
    HAS_SCIPY = False

# Import các hàm từ Phần 1 - Phần 4 của nhóm (SV1 - SV4)
from rk3 import rk3
from rk4 import rk4
from rk_table import rk_table
from phan4 import order_estimate, step_for_tol


def _validate_real_params(**kwargs) -> None:
    """Kiểm tra các tham số vô hướng không phải kiểu bool và là số thực hợp lệ."""
    for name, val in kwargs.items():
        if isinstance(val, bool) or not isinstance(val, (int, float, np.number)):
            raise TypeError(f"Tham số '{name}' phải là một số thực.")



# 1. CÁC HÀM MÔ HÌNH HÓA BÀI TOÁN ỨNG DỤNG


def cooling_ode(
    t: float | np.ndarray,
    temp: float | np.ndarray,
    k: float = 0.2,
    t_base: float = 25.0,
    amp: float = 5.0,
    omega: float = 1.0
) -> float | np.ndarray:
    """
    Hàm vế phải f(t, T) của phương trình vi phân truyền nhiệt Newton cho chi tiết
    máy trong buồng làm mát:
        dT/dt = -k * (T - T_env(t)), với T_env(t) = t_base - amp * cos(omega * t).

    Require:
        t: Thời điểm hiện tại (phút), t >= 0.
        temp: Nhiệt độ của chi tiết máy tại thời điểm t (°C).
        k: Hệ số truyền nhiệt đối lưu (1/phút), cần thỏa mãn k > 0.
        t_base: Nhiệt độ trung bình của buồng làm mát (°C).
        amp: Biên độ dao động nhiệt độ của buồng làm mát (°C), amp >= 0.
        omega: Tần số góc dao động nhiệt độ (rad/phút), omega >= 0.

    Ensure:
        Trả về tốc độ biến thiên nhiệt độ dT/dt (°C/phút) tại trạng thái (t, temp).

    Thuật toán:
        Mô hình hóa định luật làm nguội Newton (Chương 6 - Phương trình vi phân).
    """
    _validate_real_params(k=k, t_base=t_base, amp=amp, omega=omega)
    if isinstance(t, bool) or isinstance(temp, bool):
        raise TypeError("Thời gian t và nhiệt độ temp không được là kiểu bool.")

    # R6: Kiểm tra tình huống biên cho tham số vật lý và thời gian
    if np.any(np.asarray(t) < 0):
        raise ValueError("Thời điểm t không được âm (t >= 0).")
    if k <= 0:
        raise ValueError("Hệ số truyền nhiệt k phải lớn hơn 0.")
    if amp < 0:
        raise ValueError("Biên độ dao động nhiệt độ amp không được âm.")
    if omega < 0:
        raise ValueError("Tần số góc omega không được âm.")

    t_env = t_base - amp * np.cos(omega * t)  # dòng 1 - Tính nhiệt độ buồng làm mát T_env(t)
    return -k * (temp - t_env)                # dòng 2 - Định luật làm nguội Newton


def cooling_exact(
    t: float | np.ndarray,
    t0: float = 100.0,
    k: float = 0.2,
    t_base: float = 25.0,
    amp: float = 5.0,
    omega: float = 1.0,
    t_a: float = 0.0
) -> float | np.ndarray:
    """
    Nghiệm giải tích chính xác của bài toán làm nguội Newton với điều kiện đầu
    tổng quát T(t_a) = t0, dùng để đối chiếu sai số thực nghiệm.

    Require:
        t: Thời điểm hoặc mảng thời điểm cần tính nghiệm (phút), t >= t_a >= 0.
        t0: Nhiệt độ ban đầu của chi tiết máy tại thời điểm t = t_a (°C).
        k: Hệ số truyền nhiệt (1/phút), k > 0.
        t_base: Nhiệt độ trung bình của buồng làm mát (°C).
        amp: Biên độ dao động nhiệt độ của buồng làm mát (°C), amp >= 0.
        omega: Tần số góc dao động (rad/phút), omega >= 0.
        t_a: Thời điểm bắt đầu khảo sát (phút), t_a >= 0.

    Ensure:
        Trả về giá trị nhiệt độ chính xác T_exact(t) (°C).

    Thuật toán:
        Nghiệm tổng quát của PTVP tuyến tính cấp 1 (đối chiếu kết quả theo R3).
    """
    _validate_real_params(t0=t0, k=k, t_base=t_base, amp=amp, omega=omega, t_a=t_a)
    if isinstance(t, bool):
        raise TypeError("Thời gian t không được là kiểu bool.")

    if t_a < 0 or np.any(np.asarray(t) < t_a):
        raise ValueError("Cần thỏa mãn t >= t_a >= 0.")
    if k <= 0:
        raise ValueError("Hệ số truyền nhiệt k phải lớn hơn 0.")
    if amp < 0 or omega < 0:
        raise ValueError("Biên độ amp và tần số góc omega không được âm.")

    denom = k**2 + omega**2

    # Nghiệm riêng T_p(t) của phương trình không thuần nhất
    def particular_sol(time_val):
        return t_base - amp * (
            k**2 * np.cos(omega * time_val) + k * omega * np.sin(omega * time_val)
        ) / denom

    # Hằng số C xác định từ điều kiện đầu T(t_a) = t0
    c_const = t0 - particular_sol(t_a)
    homogeneous_part = c_const * np.exp(-k * (t - t_a))

    return particular_sol(t) + homogeneous_part



# 2. HÀM CHẠY VÀ SO SÁNH CÁC PHƯƠNG PHÁP (RK3, RK4, NGHIỆM ĐÚNG, SCIPY)


def solve_cooling_application(
    a: float = 0.0,
    b: float = 10.0,
    y0: float = 100.0,
    n: int = 20,
    k: float = 0.2,
    t_base: float = 25.0,
    amp: float = 5.0,
    omega: float = 1.0
) -> dict:
    """
    Giải bài toán làm nguội chi tiết máy bằng RK3 và RK4, đồng thời tính sai số
    đối chiếu với nghiệm giải tích và scipy.integrate.solve_ivp (nếu có).

    Require:
        a, b: Khoảng thời gian khảo sát [a, b] (phút), thỏa mãn 0 <= a < b.
        y0: Nhiệt độ ban đầu của chi tiết máy tại t = a (°C).
        n: Số bước chia lưới thời gian, n là số nguyên dương.
        k, t_base, amp, omega: Các tham số vật lý của mô hình (k > 0, amp >= 0, omega >= 0).

    Ensure:
        Trả về một dict chứa:
            - 't': Mảng các mốc thời gian.
            - 'y_rk3': Nghiệm xấp xỉ từ hàm rk3 của nhóm.
            - 'y_rk4': Nghiệm xấp xỉ từ hàm rk4 của nhóm.
            - 'y_exact': Nghiệm giải tích chính xác trên lưới.
            - 'y_scipy': Nghiệm đối chiếu từ scipy (hoặc None nếu không có scipy).
            - 'scipy_used': Cờ boolean cho biết scipy có thực sự được gọi hay không.
            - 'signed_err_rk3': Sai số có dấu (y_rk3 - y_exact) trên lưới.
            - 'signed_err_rk4': Sai số có dấu (y_rk4 - y_exact) trên lưới.
            - 'err_rk3': Mảng sai số tuyệt đối của RK3 tại từng nút lưới.
            - 'err_rk4': Mảng sai số tuyệt đối của RK4 tại từng nút lưới.
            - 'max_err_rk3': Sai số tuyệt đối lớn nhất của RK3 trên [a, b].
            - 'max_err_rk4': Sai số tuyệt đối lớn nhất của RK4 trên [a, b].
            - 'end_err_rk3': Sai số tuyệt đối của RK3 tại thời điểm cuối t = b.
            - 'end_err_rk4': Sai số tuyệt đối của RK4 tại thời điểm cuối t = b.
            - 'nfev_rk3_theory': Số lần gọi hàm f theo lý thuyết của RK3 (3*n).
            - 'nfev_rk4_theory': Số lần gọi hàm f theo lý thuyết của RK4 (4*n).
            - 'nfev_rk3_actual': Số lần gọi hàm f thực tế đo được khi chạy rk3.
            - 'nfev_rk4_actual': Số lần gọi hàm f thực tế đo được khi chạy rk4.

    Thuật toán:
        Áp dụng Thuật toán Runge-Kutta bậc 3 (CT 6.20) và bậc 4 (CT 6.21)
        trong bài giảng Chương 6 để giải bài toán giá trị ban đầu.
    """
    _validate_real_params(a=a, b=b, y0=y0, k=k, t_base=t_base, amp=amp, omega=omega)
    # R6: Xử lý tình huống biên
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n <= 0:
        raise ValueError("Số bước chia n phải là số nguyên dương.")
    if a < 0 or a >= b:
        raise ValueError("Khoảng thời gian cần thỏa mãn 0 <= a < b.")
    if k <= 0 or amp < 0 or omega < 0:
        raise ValueError("Tham số vật lý cần thỏa mãn k > 0, amp >= 0, omega >= 0.")

    call_count_rk3 = 0
    call_count_rk4 = 0

    def f_rk3(t, y):
        nonlocal call_count_rk3
        call_count_rk3 += 1
        return cooling_ode(t, y, k=k, t_base=t_base, amp=amp, omega=omega)

    def f_rk4(t, y):
        nonlocal call_count_rk4
        call_count_rk4 += 1
        return cooling_ode(t, y, k=k, t_base=t_base, amp=amp, omega=omega)

    # Gọi hàm lõi rk3 (SV1) và rk4 (SV2)
    t_grid, y_rk3 = rk3(f_rk3, a, b, y0, n)   # Thuật toán RK3 - Phần 1 (CT 6.20)
    _, y_rk4 = rk4(f_rk4, a, b, y0, n)        # Thuật toán RK4 - Phần 2 (CT 6.21)

    # Tính nghiệm giải tích chính xác tổng quát với điều kiện đầu tại t_a = a
    y_exact = cooling_exact(
        t_grid, t0=y0, k=k, t_base=t_base, amp=amp, omega=omega, t_a=a
    )

    # Đối chiếu minh bạch với scipy.integrate.solve_ivp (không gán giả y_exact)
    if HAS_SCIPY and solve_ivp is not None:
        sol_scipy = solve_ivp(
            lambda t, y: cooling_ode(t, y, k=k, t_base=t_base, amp=amp, omega=omega),
            (a, b),
            [y0],
            t_eval=t_grid,
            method="DOP853",
            rtol=1e-13,
            atol=1e-15,
        )
        y_scipy = sol_scipy.y[0]
        scipy_used = True
    else:
        y_scipy = None
        scipy_used = False

    # Tính sai số có dấu và sai số tuyệt đối
    signed_err_rk3 = y_rk3 - y_exact
    signed_err_rk4 = y_rk4 - y_exact
    err_rk3 = np.abs(signed_err_rk3)
    err_rk4 = np.abs(signed_err_rk4)

    return {
        "t": t_grid,
        "y_rk3": y_rk3,
        "y_rk4": y_rk4,
        "y_exact": y_exact,
        "y_scipy": y_scipy,
        "scipy_used": scipy_used,
        "signed_err_rk3": signed_err_rk3,
        "signed_err_rk4": signed_err_rk4,
        "err_rk3": err_rk3,
        "err_rk4": err_rk4,
        "max_err_rk3": float(np.max(err_rk3)),
        "max_err_rk4": float(np.max(err_rk4)),
        "end_err_rk3": float(err_rk3[-1]),
        "end_err_rk4": float(err_rk4[-1]),
        "nfev_rk3_theory": 3 * n,
        "nfev_rk4_theory": 4 * n,
        "nfev_rk3_actual": call_count_rk3,
        "nfev_rk4_actual": call_count_rk4,
    }


def analyze_convergence_and_tolerance(
    a: float = 0.0,
    b: float = 10.0,
    y0: float = 100.0,
    n_list: list[int] | None = None,
    tol: float = 1e-5,
    k: float = 0.2,
    t_base: float = 25.0,
    amp: float = 5.0,
    omega: float = 1.0
) -> dict:
    """
    Phân tích sai số theo bước lưới h, ước lượng bậc hội tụ thực nghiệm,
    đo thực tế chi phí tính toán (số lần gọi hàm f cho cả 1 lưới và toàn bộ
    thủ tục dò lưới step_for_tol) thỏa mãn |y_n(b) - y_2n(b)| <= tol.

    Require:
        a, b: Khoảng thời gian [a, b] với 0 <= a < b.
        y0: Nhiệt độ ban đầu tại t = a.
        n_list: Danh sách số bước chia tăng gấp đôi (mặc định [10, 20, 40, 80, 160]).
        tol: Ngưỡng sai khác hai lưới cho phép tại thời điểm cuối b, tol > 0.
        k, t_base, amp, omega: Các tham số vật lý của mô hình.

    Ensure:
        Trả về dict chứa bảng sai số theo n_list, bậc hội tụ ước lượng từ
        hàm order_estimate (Phần 4), bước lưới từ hàm step_for_tol (Phần 4)
        kèm số đo thực tế và lý thuyết của số lần gọi hàm f.

    Thuật toán:
        Kết hợp RK3, RK4 với đánh giá sai khác hai lưới Runge (Phần 4).
    """
    if n_list is None:
        n_list = [10, 20, 40, 80, 160]

    _validate_real_params(a=a, b=b, y0=y0, tol=tol, k=k, t_base=t_base, amp=amp, omega=omega)
    if a < 0 or a >= b:
        raise ValueError("Cần thỏa mãn 0 <= a < b.")
    if tol <= 0:
        raise ValueError("Ngưỡng dung sai tol phải lớn hơn 0.")
    if k <= 0 or amp < 0 or omega < 0:
        raise ValueError("Tham số vật lý cần thỏa mãn k > 0, amp >= 0, omega >= 0.")

    def f_model(t, y):
        return cooling_ode(t, y, k=k, t_base=t_base, amp=amp, omega=omega)

    y_exact_b = float(
        cooling_exact(b, t0=y0, k=k, t_base=t_base, amp=amp, omega=omega, t_a=a)
    )
    records = []

    for n in n_list:
        if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n <= 0:
            raise ValueError("Mọi số bước trong n_list phải là số nguyên dương.")
        h = (b - a) / n
        c3, c4 = 0, 0

        def f_c3(t, y):
            nonlocal c3
            c3 += 1
            return f_model(t, y)

        def f_c4(t, y):
            nonlocal c4
            c4 += 1
            return f_model(t, y)

        _, y3 = rk3(f_c3, a, b, y0, n)
        _, y4 = rk4(f_c4, a, b, y0, n)
        records.append({
            "n": n,
            "h": h,
            "y_rk3_b": float(y3[-1]),
            "y_rk4_b": float(y4[-1]),
            "err_rk3_b": abs(float(y3[-1]) - y_exact_b),
            "err_rk4_b": abs(float(y4[-1]) - y_exact_b),
            "nfev_rk3_theory": 3 * n,
            "nfev_rk4_theory": 4 * n,
            "nfev_rk3_actual": c3,
            "nfev_rk4_actual": c4,
        })

    # Gọi hàm của SV4 để tính bậc hội tụ thực nghiệm
    orders_rk3 = order_estimate(rk3, f_model, a, b, y0, n_list)
    orders_rk4 = order_estimate(rk4, f_model, a, b, y0, n_list)

    # Đo thực tế tổng số lần gọi hàm f trong toàn bộ thủ tục step_for_tol
    total_search_calls_rk3 = 0
    total_search_calls_rk4 = 0

    def f_search_rk3(t, y):
        nonlocal total_search_calls_rk3
        total_search_calls_rk3 += 1
        return f_model(t, y)

    def f_search_rk4(t, y):
        nonlocal total_search_calls_rk4
        total_search_calls_rk4 += 1
        return f_model(t, y)

    h_tol_rk3, y_tol_rk3 = step_for_tol(rk3, f_search_rk3, a, b, y0, tol)
    h_tol_rk4, y_tol_rk4 = step_for_tol(rk4, f_search_rk4, a, b, y0, tol)

    n_tol_rk3 = int(round((b - a) / h_tol_rk3))
    n_tol_rk4 = int(round((b - a) / h_tol_rk4))

    # Đo trực tiếp số lần gọi hàm f trên đúng 1 lưới được chọn (n_tol) bằng bộ đếm
    single_grid_calls_rk3 = 0
    single_grid_calls_rk4 = 0

    def f_single_rk3(t, y):
        nonlocal single_grid_calls_rk3
        single_grid_calls_rk3 += 1
        return f_model(t, y)

    def f_single_rk4(t, y):
        nonlocal single_grid_calls_rk4
        single_grid_calls_rk4 += 1
        return f_model(t, y)

    rk3(f_single_rk3, a, b, y0, n_tol_rk3)
    rk4(f_single_rk4, a, b, y0, n_tol_rk4)

    # Tổng số bước chia tích lũy qua các vòng lặp n = 1, 2, 4, ..., n_tol (mỗi vòng chạy n và 2n)
    # Tổng(n + 2n) từ n=1 đến n_tol là: 3 * (2 * n_tol - 1) bước
    total_steps_rk3 = 3 * (2 * n_tol_rk3 - 1)
    total_steps_rk4 = 3 * (2 * n_tol_rk4 - 1)

    return {
        "y_exact_b": y_exact_b,
        "records": records,
        "orders_rk3": orders_rk3,
        "orders_rk4": orders_rk4,
        "tol_rk3": {
            "h": h_tol_rk3,
            "n": n_tol_rk3,
            "y_b": y_tol_rk3,
            "true_err_b": abs(y_tol_rk3 - y_exact_b),
            "nfev_single_grid_theory": 3 * n_tol_rk3,
            "nfev_single_grid_actual": single_grid_calls_rk3,
            "nfev_total_search_theory": 3 * total_steps_rk3,
            "nfev_total_search_actual": total_search_calls_rk3,
        },
        "tol_rk4": {
            "h": h_tol_rk4,
            "n": n_tol_rk4,
            "y_b": y_tol_rk4,
            "true_err_b": abs(y_tol_rk4 - y_exact_b),
            "nfev_single_grid_theory": 4 * n_tol_rk4,
            "nfev_single_grid_actual": single_grid_calls_rk4,
            "nfev_total_search_theory": 4 * total_steps_rk4,
            "nfev_total_search_actual": total_search_calls_rk4,
        },
    }


def analyze_stability_comparison(
    a: float = 0.0,
    b: float = 10.0,
    y0: float = 100.0,
    n: int = 20,
    k_list: list[float] | None = None,
    t_base: float = 25.0,
    amp: float = 5.0,
    omega: float = 1.0
) -> list[dict]:
    """
    Khảo sát tính ổn định tuyệt đối của RK3 và RK4 khi tích số k*h tiến gần và
    vượt quá ngưỡng ổn định lý thuyết (|R(-k*h)| <= 1: k*h <= 2.5127 với RK3,
    k*h <= 2.7853 với RK4).

    Lưu ý quan trọng phân biệt giữa "Ổn định" và "Chính xác":
    - Điều kiện |R(-k*h)| < 1 chỉ là điều kiện ổn định tuyệt đối (đảm bảo nhiễu
      và thành phần quá độ tắt dần khi t -> vô cùng, tức sai số cuối end_err nhỏ),
      KHÔNG đồng nghĩa với việc nghiệm chính xác ở các bước đầu khi k*h lớn (~2.5).
    - Khi k*h ≈ 2.6 (ví dụ k = 5.2, h = 0.5 — kịch bản làm nguội cực nhanh khi
      tôi kim loại trong chất lỏng), bước đầu tiên có sai số quá độ lớn (~54 °C),
      nhưng với RK4 (|R| = 0.759 < 1) sai số này tắt dần về ~0.031 °C tại t = 10,
      trong khi với RK3 (|R| = 1.149 > 1) sai số bị khuếch đại vọt lên ~1294 °C.

    Require:
        a, b: Khoảng thời gian [a, b] với 0 <= a < b.
        y0: Nhiệt độ ban đầu tại t = a.
        n: Số bước chia cố định (mặc định n = 20 -> h = 0.5 phút).
        k_list: Danh sách hệ số truyền nhiệt k > 0 để thử nghiệm.

    Ensure:
        Trả về danh sách dict ghi nhận tích số k*h, hệ số khuếch đại |R(-k*h)|,
        sai số lớn nhất trên toàn lưới (max_err - bị chi phối bởi bước đầu) và
        sai số tại thời điểm cuối t = b (end_err - phản ánh tính ổn định tiệm cận).

    Thuật toán:
        Đối chiếu thực nghiệm với lý thuyết miền ổn định tuyệt đối của họ RK hiện.
    """
    if k_list is None:
        k_list = [0.2, 5.0, 5.2, 5.5, 10.0]
    _validate_real_params(a=a, b=b, y0=y0, t_base=t_base, amp=amp, omega=omega)
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n <= 0:
        raise ValueError("Số bước chia n phải là số nguyên dương.")
    if a < 0 or a >= b:
        raise ValueError("Cần thỏa mãn 0 <= a < b.")

    h = (b - a) / n
    results = []
    for k_val in k_list:
        _validate_real_params(k_val=k_val)
        if k_val <= 0:
            raise ValueError("Mọi hệ số k trong k_list phải lớn hơn 0.")
        kh = k_val * h
        z = -kh
        # Đa thức ổn định của RK3 và RK4 hiện
        r_rk3 = abs(1.0 + z + (z**2) / 2.0 + (z**3) / 6.0)
        r_rk4 = abs(1.0 + z + (z**2) / 2.0 + (z**3) / 6.0 + (z**4) / 24.0)

        sim = solve_cooling_application(
            a=a, b=b, y0=y0, n=n, k=k_val, t_base=t_base, amp=amp, omega=omega
        )
        results.append({
            "k": k_val,
            "h": h,
            "kh": kh,
            "amp_factor_rk3": float(r_rk3),
            "amp_factor_rk4": float(r_rk4),
            "stable_rk3": bool(r_rk3 <= 1.0),
            "stable_rk4": bool(r_rk4 <= 1.0),
            "max_err_rk3": sim["max_err_rk3"],
            "max_err_rk4": sim["max_err_rk4"],
            "end_err_rk3": sim["end_err_rk3"],
            "end_err_rk4": sim["end_err_rk4"],
        })
    return results



# 3. HÀM VẼ ĐỒ THỊ MINH HỌA KẾT QUẢ ỨNG DỤNG (DÙNG CHO BÁO CÁO & SLIDE)


def plot_application_results(
    save_path: str = "app_cooling_analysis.png",
    a: float = 0.0,
    b: float = 10.0,
    y0: float = 100.0,
    n: int = 20,
    k: float = 0.2,
    t_base: float = 25.0,
    amp: float = 5.0,
    omega: float = 1.0
) -> None:
    """
    Vẽ bộ 3 đồ thị phân tích bài toán ứng dụng (đã xử lý triệt để các góp ý trực quan):
        1. Quỹ đạo nhiệt độ chi tiết máy T(t) và buồng làm mát T_env(t). Khung
           phóng đại (inset) đặt tại vùng trống góc dưới-trái, chỉ phóng đại
           trực tiếp tại nút lưới t = 2.0 phút (khi h = 2.0) và không nối dây cung
           để tránh hiểu nhầm sai số nội suy tuyến tính thành sai số phương pháp.
        2. Sai số tuyệt đối theo thời gian |T_i - T_exact(t_i)| trên thang log,
           đánh dấu đầy đủ các khoảng đổi dấu thực sự của sai số có dấu RK4.
        3. Đồ thị hội tụ log-log của sai số tại t = b theo bước lưới h (neo đường
           chuẩn tại bước lưới mịn nhất h_min và hiển thị đúng bậc hội tụ thực tế).

    Require:
        save_path: Đường dẫn file ảnh đầu ra (.png, .pdf), chuỗi không rỗng.
        a, b, y0, n, k, t_base, amp, omega: Các tham số của mô hình làm mát.

    Ensure:
        Lưu file đồ thị chất lượng cao phục vụ báo cáo LaTeX và thuyết trình.

    Thuật toán:
        Trực quan hóa kết quả thực nghiệm bằng matplotlib (R1, A3, A4).
    """
    if not isinstance(save_path, str) or not save_path.strip():
        raise ValueError("Đường dẫn save_path phải là chuỗi hợp lệ.")

    res = solve_cooling_application(
        a=a, b=b, y0=y0, n=n, k=k, t_base=t_base, amp=amp, omega=omega
    )
    res_coarse = solve_cooling_application(
        a=a, b=b, y0=y0, n=5, k=k, t_base=t_base, amp=amp, omega=omega
    )
    conv = analyze_convergence_and_tolerance(
        a=a, b=b, y0=y0, n_list=[10, 20, 40, 80, 160], k=k, t_base=t_base, amp=amp, omega=omega
    )

    t_fine = np.linspace(a, b, 400)
    y_exact_fine = cooling_exact(
        t_fine, t0=y0, k=k, t_base=t_base, amp=amp, omega=omega, t_a=a
    )
    t_env_fine = t_base - amp * np.cos(omega * t_fine)
    h_step = (b - a) / n

    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.2))

    # --------------------------------------------------------------------------
    # Hình 1: Quỹ đạo nhiệt độ kèm khung phóng đại tại nút t_1 (không nối dây cung)
    # --------------------------------------------------------------------------
    ax1 = axes[0]
    ax1.plot(t_fine, y_exact_fine, "k-", lw=2.0, label="Nghiệm chính xác $T_{exact}(t)$")
    ax1.plot(
        res["t"], res["y_rk3"], "bo--", ms=4, lw=1.2,
        label=f"RK3 ($n={n}, h={h_step:.2g}$)"
    )
    ax1.plot(
        res["t"], res["y_rk4"], "r^:", ms=4, lw=1.2,
        label=f"RK4 ($n={n}, h={h_step:.2g}$)"
    )
    ax1.plot(
        t_fine, t_env_fine, "g-.", lw=1.2, alpha=0.75,
        label="Nhiệt độ buồng làm mát $T_{env}(t)$"
    )
    ax1.set_title("Nhiệt độ chi tiết máy $T(t)$ trong buồng làm mát")
    ax1.set_xlabel("Thời gian $t$ (phút)")
    ax1.set_ylabel(r"Nhiệt độ $T$ ($^\circ$C)")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=8.0, loc="upper right")

    # Đặt khung phóng đại ở vùng trống phía trên bên phải dưới legend [0.46, 0.48, 0.48, 0.25] (hoàn toàn không chạm đường cong)
    # Chỉ vẽ các điểm nút rời rạc tại t_1 = a + h_coarse (t = 2.0 khi h = 2.0) để thấy đúng sai số tại nút
    ax_ins = ax1.inset_axes([0.46, 0.48, 0.48, 0.25])
    t_node = float(res_coarse["t"][1])
    y_ex_node = float(res_coarse["y_exact"][1])
    y_rk3_node = float(res_coarse["y_rk3"][1])
    y_rk4_node = float(res_coarse["y_rk4"][1])
    err3_node = abs(y_rk3_node - y_ex_node)
    err4_node = abs(y_rk4_node - y_ex_node)

    t_local = np.linspace(t_node - 0.008 * (b - a), t_node + 0.008 * (b - a), 50)
    y_local = cooling_exact(
        t_local, t0=y0, k=k, t_base=t_base, amp=amp, omega=omega, t_a=a
    )
    ax_ins.plot(t_local, y_local, "k-", lw=1.3)
    # Chỉ vẽ các điểm nút rời rạc (không nối dây cung giữa hai nút) và vạch gióng sai số nút
    ax_ins.vlines(t_node, y_rk3_node, y_ex_node, colors="blue", linestyles="--", lw=1.2)
    ax_ins.plot([t_node], [y_ex_node], "ko", ms=4.5, label="$T_{exact}(2)$")
    ax_ins.plot([t_node], [y_rk3_node], "bo", ms=5.0, label=f"RK3 (sai {err3_node:.2f}°C)")
    ax_ins.plot([t_node], [y_rk4_node], "r^", ms=5.0, label=f"RK4 (sai {err4_node:.3f}°C)")

    pad_y = max(err3_node * 1.35, 0.05)
    ax_ins.set_ylim(y_rk3_node - 0.35 * pad_y, y_ex_node + 0.45 * pad_y)
    ax_ins.set_title(f"Tại nút $t={t_node:g}$ ($h={(b-a)/5:g}$, không nối dây cung)", fontsize=7.2)
    ax_ins.tick_params(labelsize=6.5)
    ax_ins.legend(fontsize=6.2, loc="lower left", framealpha=0.85)
    ax_ins.grid(True, linestyle=":", alpha=0.5)


    # Hình 2: Sai số tuyệt đối theo thời gian (thang logarit) + các vùng đổi dấu

    ax2 = axes[1]
    ax2.semilogy(res["t"][1:], res["err_rk3"][1:], "bo-", ms=4, label="Sai số tuyệt đối RK3")
    ax2.semilogy(res["t"][1:], res["err_rk4"][1:], "r^-", ms=4, label="Sai số tuyệt đối RK4")

    # Tìm tất cả các khoảng (t_i, t_{i+1}) mà sai số có dấu của RK4 đổi dấu (e_i * e_{i+1} < 0)
    s_rk4 = res["signed_err_rk4"]
    sign_change_intervals = []
    for idx in range(1, len(s_rk4) - 1):
        if s_rk4[idx] * s_rk4[idx + 1] < 0:
            sign_change_intervals.append((res["t"][idx], res["t"][idx + 1]))

    if sign_change_intervals:
        intervals_str = "; ".join(f"[{t1:g}, {t2:g}]" for t1, t2 in sign_change_intervals)
        min_idx_rk4 = int(np.argmin(res["err_rk4"][1:])) + 1
        ax2.annotate(
            f"RK4 đổi dấu sai số trên:\n{intervals_str} phút",
            xy=(res["t"][min_idx_rk4], res["err_rk4"][min_idx_rk4]),
            xytext=(res["t"][min_idx_rk4] - 5.2, res["err_rk4"][min_idx_rk4] * 3.8),
            arrowprops=dict(arrowstyle="->", color="darkred", lw=1.0),
            fontsize=7.8,
            color="darkred",
            bbox=dict(boxstyle="round,pad=0.2", fc="#fff8f8", ec="darkred", alpha=0.8),
        )

    ax2.set_title(f"Sai số tuyệt đối $|T_i - T_{{exact}}(t_i)|$ với $h = {h_step:.2g}$")
    ax2.set_xlabel("Thời gian $t$ (phút)")
    ax2.set_ylabel(r"Sai số tuyệt đối ($^\circ$C - thang log)")
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)
    ax2.legend(fontsize=8.5, loc="center left")


    # Hình 3: Tốc độ hội tụ theo bước lưới h (thang log-log)

    ax3 = axes[2]
    h_vals = np.array([r["h"] for r in conv["records"]])
    e3_vals = np.array([r["err_rk3_b"] for r in conv["records"]])
    e4_vals = np.array([r["err_rk4_b"] for r in conv["records"]])

    p3_last = conv["orders_rk3"][-1]
    p4_last = conv["orders_rk4"][-1]

    ref_h3 = e3_vals[-1] * (h_vals / h_vals[-1]) ** 3
    ref_h4 = e4_vals[-1] * (h_vals / h_vals[-1]) ** 4

    ax3.loglog(
        h_vals, e3_vals, "bo-", lw=1.5,
        label=rf"Sai số RK3 tại $t={b:g}$ (bậc $\approx {p3_last:.2f}$)"
    )
    ax3.loglog(h_vals, ref_h3, "b--", lw=1.0, alpha=0.6, label=r"Đường chuẩn $O(h^3)$ (neo tại $h_{min}$)")
    ax3.loglog(
        h_vals, e4_vals, "r^-", lw=1.5,
        label=rf"Sai số RK4 tại $t={b:g}$ (bậc $\approx {p4_last:.2f}$)"
    )
    ax3.loglog(h_vals, ref_h4, "r--", lw=1.0, alpha=0.6, label=r"Đường chuẩn $O(h^4)$ (neo tại $h_{min}$)")
    ax3.set_title("Đồ thị hội tụ sai số theo bước lưới $h$")
    ax3.set_xlabel("Bước thời gian $h$ (phút - thang log)")
    ax3.set_ylabel(f"Sai số $|T_n({b:g}) - T_{{exact}}({b:g})|$ (thang log)")
    ax3.grid(True, which="both", linestyle="--", alpha=0.5)
    ax3.legend(fontsize=8.0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)



# 4. CHƯƠNG TRÌNH CHÍNH (IN KẾT QUẢ THỰC NGHIỆM)


if __name__ == "__main__":
    print("=" * 96)
    print("BÁO CÁO KẾT QUẢ BÀI TOÁN ỨNG DỤNG (SV7 - NHÓM 8)")
    print("Làm mát chi tiết máy trong buồng điều hòa (Định luật làm nguội Newton, Bi < 0.1)")
    print("=" * 96)

    # 1. Trích xuất 5 bước tính tay đầu tiên bằng hàm rk_table (Phần 3) với n = 20 (h = 0.5)
    def f_app(t, y):
        return cooling_ode(t, y)

    table_rk4_sample = rk_table(rk4, f_app, 0.0, 10.0, 100.0, 20)
    print("\n[1] BẢNG GIÁ TRỊ 5 BƯỚC ĐẦU TIÊN CỦA RK4 (h = 0.5 phút):")
    print(f"{'i':>2} | {'t_i':>5} | {'T_i':>10} | {'K1':>10} | {'K2':>10} | {'K3':>10} | {'K4':>10} | {'T_{i+1}':>10}")
    print("-" * 82)
    for row in table_rk4_sample[:5]:
        print(
            f"{row['i']:2d} | {row['x_i']:5.2f} | {row['y_i']:10.6f} | "
            f"{row['K1']:10.6f} | {row['K2']:10.6f} | {row['K3']:10.6f} | "
            f"{row['K4']:10.6f} | {row['y_next']:10.6f}"
        )

    # 2. Phân tích hội tụ và so sánh RK3, RK4
    conv_data = analyze_convergence_and_tolerance(0.0, 10.0, 100.0, [10, 20, 40, 80, 160], tol=1e-5)
    print(f"\n[2] NGHIỆM CHÍNH XÁC TẠI t = 10 PHÚT: T_exact(10) = {conv_data['y_exact_b']:.10f} °C")
    print("\n[3] BẢNG SO SÁNH SAI SỐ & CHI PHÍ TẠI t = 10 PHÚT THEO BƯỚC LƯỚI h:")
    print(f"{'n':>5} | {'h':>7} | {'T_RK3(10)':>13} | {'Sai số RK3':>11} | {'Gọi f (LT/TT)':>13} | {'T_RK4(10)':>13} | {'Sai số RK4':>11} | {'Gọi f (LT/TT)':>13}")
    print("-" * 106)
    for rec in conv_data["records"]:
        c3_str = f"{rec['nfev_rk3_theory']}/{rec['nfev_rk3_actual']}"
        c4_str = f"{rec['nfev_rk4_theory']}/{rec['nfev_rk4_actual']}"
        print(
            f"{rec['n']:5d} | {rec['h']:7.4f} | {rec['y_rk3_b']:13.8f} | "
            f"{rec['err_rk3_b']:11.4e} | {c3_str:>13} | "
            f"{rec['y_rk4_b']:13.8f} | {rec['err_rk4_b']:11.4e} | {c4_str:>13}"
        )

    print("\n[4] ƯỚC LƯỢNG BẬC HỘI TỤ THỰC NGHIỆM (GỌI HÀM order_estimate - PHẦN 4):")
    print("  - Bậc hội tụ RK3:", [round(p, 4) for p in conv_data["orders_rk3"]])
    print("  - Bậc hội tụ RK4:", [round(p, 4) for p in conv_data["orders_rk4"]])

    print("\n[5] BƯỚC LƯỚI LỚN NHẤT TRONG DÃY CHIA ĐÔI ĐỂ SAI KHÁC HAI LƯỚI |y_n - y_2n| <= 10^-5 (step_for_tol):")
    t3 = conv_data["tol_rk3"]
    t4 = conv_data["tol_rk4"]
    print(
        f"  - Với RK3: n = {t3['n']:4d} (h = {t3['h']:.6f} phút) -> T(10) ≈ {t3['y_b']:.8f} °C | Sai số thật = {t3['true_err_b']:.2e}\n"
        f"      + Chi phí chạy 1 lưới (n={t3['n']}): {t3['nfev_single_grid_theory']} lần (lý thuyết) / {t3['nfev_single_grid_actual']} lần (đo thực tế)\n"
        f"      + Tổng chi phí cả thủ tục dò lưới step_for_tol: {t3['nfev_total_search_theory']} lần (lý thuyết) / {t3['nfev_total_search_actual']} lần (đo thực tế)"
    )
    print(
        f"  - Với RK4: n = {t4['n']:4d} (h = {t4['h']:.6f} phút) -> T(10) ≈ {t4['y_b']:.8f} °C | Sai số thật = {t4['true_err_b']:.2e}\n"
        f"      + Chi phí chạy 1 lưới (n={t4['n']}): {t4['nfev_single_grid_theory']} lần (lý thuyết) / {t4['nfev_single_grid_actual']} lần (đo thực tế)\n"
        f"      + Tổng chi phí cả thủ tục dò lưới step_for_tol: {t4['nfev_total_search_theory']} lần (lý thuyết) / {t4['nfev_total_search_actual']} lần (đo thực tế)"
    )

    # 3. Khảo sát tính ổn định theo tích số k*h (kịch bản mở rộng làm nguội nhanh trong chất lỏng)
    stab_data = analyze_stability_comparison(a=0.0, b=10.0, y0=100.0, n=20)
    print("\n[6] KHẢO SÁT MIỀN ỔN ĐỊNH VỚI h = 0.5 PHÚT (n = 20) - PHÂN BIỆT SAI SỐ QUÁ ĐỘ (MAX) VÀ TIỆM CẬN (END TẠI t=10):")
    print(f"{'k':>6} | {'k*h':>5} | {'|R_RK3|':>9} | {'MaxErr RK3':>11} | {'EndErr RK3':>11} | {'|R_RK4|':>9} | {'MaxErr RK4':>11} | {'EndErr RK4':>11}")
    print("-" * 96)
    for s in stab_data:
        print(
            f"{s['k']:6.2f} | {s['kh']:5.2f} | {s['amp_factor_rk3']:9.4f} | "
            f"{s['max_err_rk3']:11.3e} | {s['end_err_rk3']:11.3e} | {s['amp_factor_rk4']:9.4f} | "
            f"{s['max_err_rk4']:11.3e} | {s['end_err_rk4']:11.3e}"
        )

    # 4. Xuất đồ thị
    plot_application_results("app_cooling_analysis.png")
    print("\n[7] Đã xuất đồ thị phân tích ra file: app_cooling_analysis.png")
