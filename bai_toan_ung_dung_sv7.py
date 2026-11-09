"""Kiểm thử pytest cho bài toán ứng dụng vật rơi có lực cản (SV7)."""

import numpy as np
import pytest

import application_vatroi as app
from rk3 import rk3
from rk4 import rk4


def test_v_exact_la_nghiem_cua_ptvp():
    """Ca có nghiệm đúng: v_exact thỏa v(0)=0 và v' = g - (k/m)v^2."""
    assert app.v_exact(0.0) == pytest.approx(0.0, abs=1e-15)
    for t in (1.0, 5.0, 12.0):
        e = 1e-5
        dv = (app.v_exact(t + e) - app.v_exact(t - e)) / (2 * e)
        assert dv == pytest.approx(app.f_falling_body(t, app.v_exact(t)), abs=1e-6)
    assert 0.99 * app.V_INF < app.v_exact(app.T_END) < app.V_INF


@pytest.mark.parametrize("method, tol", [(rk3, 1e-4), (rk4, 1e-6)])
def test_find_time_khop_giai_tich(method, tol):
    """Ví dụ chuẩn: t_95 từ RK khớp giá trị giải tích arctanh(0.95)*v_inf/g."""
    t, v = method(app.f_falling_body, 0.0, 20.0, 0.0, 200)
    t95 = app.find_time_for_ratio(method, app.f_falling_body, t, v, app.V_INF, 0.95)
    exact = app.V_INF / app.G_CONST * np.arctanh(0.95)
    assert t95 == pytest.approx(exact, abs=tol)


def test_find_time_ca_bien():
    """Ca biên: ratio sai, lưới sai, chưa đạt ngưỡng, đã đạt ngưỡng tại t0."""
    t, v = rk4(app.f_falling_body, 0.0, 20.0, 0.0, 100)
    args = (rk4, app.f_falling_body)
    for bad in (0.0, 1.0, 1.5, -0.1):
        with pytest.raises(ValueError):
            app.find_time_for_ratio(*args, t, v, app.V_INF, bad)
    with pytest.raises(ValueError):
        app.find_time_for_ratio(*args, t, v[:-1], app.V_INF)
    with pytest.raises(ValueError):
        app.find_time_for_ratio(*args, t[::-1], v, app.V_INF)
    with pytest.raises(ValueError):
        app.find_time_for_ratio(*args, t, v, -1.0)
    t_short, v_short = rk4(app.f_falling_body, 0.0, 1.0, 0.0, 10)
    with pytest.raises(RuntimeError):
        app.find_time_for_ratio(*args, t_short, v_short, app.V_INF)
    t0, v0 = rk4(app.f_falling_body, 0.0, 5.0, 0.99 * app.V_INF, 5)
    assert app.find_time_for_ratio(*args, t0, v0, app.V_INF, 0.95) == 0.0


def test_count_calls_dung_so_lan():
    """Số lần gọi f: RK3 = 3n, RK4 = 4n."""
    n = 17
    for method, per_step in ((rk3, 3), (rk4, 4)):
        f, counter = app.count_calls(app.f_falling_body)
        method(f, 0.0, 20.0, 0.0, n)
        assert counter["n"] == per_step * n
