import pytest
import numpy as np
from rk4 import rk4
from rk_table import rk_table
from application import (
    cooling_ode,
    cooling_exact,
    solve_cooling_application,
    analyze_convergence_and_tolerance,
    analyze_stability_comparison,
)


def test_cooling_ode_standard():
    """Ca 1 (Ví dụ cơ bản): Kiểm tra đạo hàm, nghiệm đúng tại t = 0 và tích hợp bảng rk_table."""
    val = cooling_ode(0.0, 100.0, k=0.2, t_base=25.0, amp=5.0, omega=1.0)
    # T_env(0) = 25 - 5*cos(0) = 20 -> dT/dt = -0.2 * (100 - 20) = -16.0
    assert np.isclose(val, -16.0, atol=1e-12)
    assert np.isclose(cooling_exact(0.0, t0=100.0), 100.0, atol=1e-12)

    # Kiểm tra tích hợp hàm rk_table (Phần 3) với bài toán làm nguội
    tbl = rk_table(rk4, cooling_ode, 0.0, 10.0, 100.0, 20)
    assert len(tbl) == 20
    assert np.isclose(tbl[0]["K1"], 0.5 * (-16.0), atol=1e-12)


def test_rk3_rk4_accuracy_against_exact():
    """Ca 2 (Nghiệm đúng đối chiếu): Kiểm tra độ chính xác và bộ đếm chi phí của RK3, RK4."""
    res = solve_cooling_application(a=0.0, b=10.0, y0=100.0, n=40)
    assert res["max_err_rk3"] < 2e-4
    assert res["max_err_rk4"] < 1e-5
    assert res["max_err_rk4"] < res["max_err_rk3"]
    assert res["nfev_rk3_theory"] == 120
    assert res["nfev_rk4_theory"] == 160
    assert res["nfev_rk3_actual"] >= res["nfev_rk3_theory"]
    assert res["nfev_rk4_actual"] >= res["nfev_rk4_theory"]

    # Kiểm tra khoảng dịch chuyển t_a = 2.0 > 0 để đảm bảo nghiệm giải tích tổng quát đúng
    res_shifted = solve_cooling_application(a=2.0, b=10.0, y0=80.0, n=32)
    assert res_shifted["max_err_rk4"] < 1e-5
    assert res_shifted["max_err_rk4"] < res_shifted["max_err_rk3"]


def test_scipy_reference_comparison():
    """Ca 2b (Đối chiếu độc lập với SciPy): Bỏ qua minh bạch nếu môi trường không cài scipy."""
    pytest.importorskip("scipy")
    res = solve_cooling_application(a=0.0, b=10.0, y0=100.0, n=40)
    assert res["scipy_used"] is True
    assert res["y_scipy"] is not None
    assert np.allclose(res["y_exact"], res["y_scipy"], atol=1e-10)

    res_shifted = solve_cooling_application(a=2.0, b=10.0, y0=80.0, n=32)
    assert np.allclose(res_shifted["y_exact"], res_shifted["y_scipy"], atol=1e-10)


def test_convergence_cost_and_stability():
    """Ca 3 (Thực nghiệm hội tụ, chi phí đo thực tế & ổn định): Kiểm tra hội tụ, tol và miền ổn định."""
    conv = analyze_convergence_and_tolerance(
        a=0.0, b=10.0, y0=100.0, n_list=[20, 40, 80, 160], tol=1e-5
    )
    assert np.isclose(conv["orders_rk3"][-1], 3.0, atol=0.05)
    assert np.isclose(conv["orders_rk4"][-1], 4.0, atol=0.05)

    # Với tol = 1e-5, RK4 cần n = 32 bước chia trong khi RK3 cần n = 128 bước chia
    assert conv["tol_rk4"]["n"] == 32
    assert conv["tol_rk3"]["n"] == 128
    assert conv["tol_rk4"]["nfev_single_grid_theory"] == 128
    assert conv["tol_rk3"]["nfev_single_grid_theory"] == 384

    # Kiểm tra chi phí tổng của cả thủ tục step_for_tol được đo bằng bộ đếm thực tế
    assert conv["tol_rk4"]["nfev_total_search_theory"] == 756
    assert conv["tol_rk3"]["nfev_total_search_theory"] == 2295
    assert conv["tol_rk4"]["nfev_total_search_actual"] == 804
    assert conv["tol_rk3"]["nfev_total_search_actual"] == 2343

    # Kiểm tra tính ổn định khi k*h = 2.6 (RK3 mất ổn định |R|>1, RK4 ổn định |R|<1 nên end_err tắt dần)
    # và khi k*h = 5.0 (cả hai vượt miền ổn định nên phát nổ)
    stab = analyze_stability_comparison(a=0.0, b=10.0, y0=100.0, n=20, k_list=[5.2, 10.0])
    assert stab[0]["amp_factor_rk3"] > 1.0 and stab[0]["amp_factor_rk4"] < 1.0
    assert stab[0]["end_err_rk4"] < 1.0 and stab[0]["end_err_rk3"] > 200.0
    assert stab[1]["end_err_rk3"] > 1e20 and stab[1]["end_err_rk4"] > 1e20


def test_application_edge_cases():
    """Ca 4 (Ca biên - R6, R7): Kiểm tra xử lý lỗi khi tham số đầu vào vi phạm điều kiện."""
    with pytest.raises(ValueError):
        cooling_ode(-1.0, 100.0)
    with pytest.raises(ValueError):
        cooling_ode(0.0, 100.0, k=-0.5)
    with pytest.raises(ValueError):
        cooling_ode(0.0, 100.0, amp=-1.0)
    with pytest.raises(TypeError):
        cooling_ode(0.0, 100.0, k=True)
    with pytest.raises(ValueError):
        cooling_exact(1.0, t0=100.0, t_a=2.0)  # t < t_a
    with pytest.raises(ValueError):
        solve_cooling_application(a=10.0, b=5.0, y0=100.0, n=10)
    with pytest.raises(ValueError):
        solve_cooling_application(a=0.0, b=10.0, y0=100.0, n=0)
    with pytest.raises(ValueError):
        solve_cooling_application(a=0.0, b=10.0, y0=100.0, n=10.5)
    with pytest.raises(ValueError):
        analyze_convergence_and_tolerance(tol=-1e-3)
