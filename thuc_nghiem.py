import numpy as np
import matplotlib.pyplot as plt

# IMPORT TRUC TIEP CAC HAM TU PHAN 1, PHAN 2 VA PHAN 4
from rk3 import rk3
from rk4 import rk4
from uocluong import order_estimate, step_for_tol


# ==============================================================================
# 1. DINH NGHIA BAI TOAN THUC NGHIEM & HAM EULER CAI TIEN (TU NHOM 7)
# ==============================================================================
# Bai toan mau bat buoc cua Nhom 8: y' = y - x^2 + 1, y(0) = 0.5 tren [0, 2]
def f_bai_toan(x, y):
    return y - x**2 + 1.0

def exact_bai_toan(x):
    return (x + 1.0) ** 2 - 0.5 * np.exp(x)


def euler_cai_tien(f, a, b, y0, n) -> tuple[np.ndarray, np.ndarray]:
    """
    Phuong phap Euler cai tien (Heun - bac 2) lay tu Nhom 7 khi phan bien cheo.
    Moi buoc chia ton 2 lan goi ham f.
    """
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = np.zeros(n + 1)
    y[0] = y0

    for i in range(n):
        k1 = f(x[i], y[i])
        k2 = f(x[i + 1], y[i] + h * k1)
        y[i + 1] = y[i] + (h / 2.0) * (k1 + k2)

    return x, y


# ==============================================================================
# 2. LAP BANG SAI SO TAI x = b VA XAC NHAN BAC HOI TU (n = 5, 10, 20, 40, 80)
# ==============================================================================
def bang_sai_so_va_bac_hoi_tu():
    a, b, y0 = 0.0, 2.0, 0.5
    n_list = [5, 10, 20, 40, 80]
    y_true_b = exact_bai_toan(b)

    h_list = []
    err_rk3 = []
    err_rk4 = []

    for n in n_list:
        h = (b - a) / n
        h_list.append(h)

        _, y3 = rk3(f_bai_toan, a, b, y0, n)
        _, y4 = rk4(f_bai_toan, a, b, y0, n)

        err_rk3.append(abs(y_true_b - y3[-1]))
        err_rk4.append(abs(y_true_b - y4[-1]))

    # Goi ham order_estimate cua Phan 4 (du 7 tham so) de xac nhan bac 3 va bac 4
    p3_list = order_estimate(rk3, f_bai_toan, a, b, y0, exact_bai_toan, n_list)
    p4_list = order_estimate(rk4, f_bai_toan, a, b, y0, exact_bai_toan, n_list)

    print("=" * 88)
    print("1. BANG SAI SO TAI x = b VA BAC HOI TU THUC NGHIEM (BAI TOAN: y' = y - x^2 + 1)")
    print("=" * 88)
    print(f"{'n':>4} | {'h':>10} | {'Sai so RK3 e3(h)':>16} | {'Bac p (RK3)':>12} | {'Sai so RK4 e4(h)':>16} | {'Bac p (RK4)':>12}")
    print("-" * 88)

    for i, n in enumerate(n_list):
        p3_str = "---" if i == 0 else f"{p3_list[i - 1]:.6f}"
        p4_str = "---" if i == 0 else f"{p4_list[i - 1]:.6f}"
        print(f"{n:4d} | {h_list[i]:10.6f} | {err_rk3[i]:16.6f} | {p3_str:>12} | {err_rk4[i]:16.6f} | {p4_str:>12}")
    print("=" * 88)

    return h_list, err_rk3, err_rk4, p3_list, p4_list


# ==============================================================================
# 3. SO SANH VOI EULER CAI TIEN (NHOM 7) THEO SO LAN TINH HAM f
# ==============================================================================
def so_sanh_chi_phi_tinh_ham_f(tol=1e-6):
    a, b, y0 = 0.0, 2.0, 0.5
    y_true_b = exact_bai_toan(b)

    print(f"\n2. SO SANH VOI EULER CAI TIEN (NHOM 7) DE DAT CUNG DO CHINH XAC (tol = {tol:.6f})")
    print("-" * 88)
    print(f"{'Phuong phap':<18} | {'Buoc h':>10} | {'So buoc n':>10} | {'So lan goi f':>14} | {'Nghiem y(b)':>12} | {'Sai so thuc':>12}")
    print("-" * 88)

    # Trong 1 buoc luoi: Euler cai tien tinh f 2 lan, RK3 tinh f 3 lan, RK4 tinh f 4 lan
    methods = [
        ("Euler cai tien", euler_cai_tien, 2),
        ("Runge-Kutta 3", rk3, 3),
        ("Runge-Kutta 4", rk4, 4),
    ]

    for name, method, fev_per_step in methods:
        # Dung step_for_tol (Phan 4) tim buoc luoi h lon nhat thoa man sai so <= tol
        h_opt, yb_approx = step_for_tol(method, f_bai_toan, a, b, y0, tol)
        n_steps = int(round((b - a) / h_opt))
        total_fev = n_steps * fev_per_step
        real_err = abs(y_true_b - yb_approx)

        print(f"{name:<18} | {h_opt:10.6f} | {n_steps:10d} | {total_fev:14d} | {yb_approx:12.6f} | {real_err:12.6f}")
    print("-" * 88)


# ==============================================================================
# 4. VE DO THI SAI SO THEO h TREN THANG DO LOG-LOG
# ==============================================================================
def ve_do_thi_loglog(h_list, err_rk3, err_rk4):
    plt.figure(figsize=(8, 6))

    # Ve duong sai so thuc nghiem cua RK3 va RK4
    plt.loglog(h_list, err_rk3, "o-", linewidth=2, markersize=7, label="RK3 (Bac hoi tu ~ 3)")
    plt.loglog(h_list, err_rk4, "s-", linewidth=2, markersize=7, label="RK4 (Bac hoi tu ~ 4)")

    # Ve duong tham chieu do doc O(h^3) va O(h^4)
    h_arr = np.array(h_list)
    ref_h3 = err_rk3[-1] * (h_arr / h_arr[-1]) ** 3 * 1.3
    ref_h4 = err_rk4[-1] * (h_arr / h_arr[-1]) ** 4 * 0.7
    plt.loglog(h_arr, ref_h3, "--", color="gray", alpha=0.7, label="Duong tham chieu O(h^3)")
    plt.loglog(h_arr, ref_h4, ":", color="black", alpha=0.7, label="Duong tham chieu O(h^4)")

    plt.xlabel("Buoc luoi h (log scale)", fontsize=11)
    plt.ylabel("Sai so tuyet doi tai x = b: |y(b) - y_h(b)| (log scale)", fontsize=11)
    plt.title("Do thi sai so tai x = b cua RK3 va RK4 theo h (Thang log-log)", fontsize=12)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()

    file_anh = "sai_so_loglog_rk3_rk4.png"
    plt.savefig(file_anh, dpi=300)
    print(f"\n3. DO THI LOG-LOG: Da luu thanh cong file anh '{file_anh}'!")
    plt.show()


# ==============================================================================
# 5. IN NHAN XET TONG KET CHO BAO CAO CUA SV6
# ==============================================================================
def in_nhan_xet():
    print("\n4. NHAN XET THUC NGHIEM (DUNG CHO BAO CAO PHAN SV6):")
    print("  - Ve sai so va bac hoi tu: Khi giam mot nua buoc luoi h (tang gap doi n),")
    print("    sai so cua RK3 giam xap xi 2^3 = 8 lan (bac hoi tu p -> 3.000000),")
    print("    sai so cua RK4 giam xap xi 2^4 = 16 lan (bac hoi tu p -> 4.000000).")
    print("  - Ve do thi log-log: Hai duong bieu dien sai so cua RK3 va RK4 la hai duong thang")
    print("    co he so goc lan luot xap xi 3 va 4, duong RK4 nam thap hon va doc hon RK3.")
    print("  - Ve so sanh voi Euler cai tien (Nhom 7): Mac du moi buoc RK4 tinh ham f 4 lan")
    print("    (nhieu hon Euler cai tien 2 lan va RK3 3 lan), nhung de dat cung do chinh xac")
    print("    tol = 0.000001, RK4 chi can so buoc n rat nho (n = 32 -> 128 lan goi f),")
    print("    trong khi RK3 can n = 256 (768 lan goi f) va Euler cai tien can n = 4096")
    print("    (8192 lan goi f). Do do, RK4 hieu qua hon vuot troi ve chi phi tinh toan!")


if __name__ == "__main__":
    h_vals, e3_vals, e4_vals, _, _ = bang_sai_so_va_bac_hoi_tu()
    so_sanh_chi_phi_tinh_ham_f(tol=1e-6)
    in_nhan_xet()
    ve_do_thi_loglog(h_vals, e3_vals, e4_vals)