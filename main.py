import os, sys, runpy, subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

MENU = {
    "1": ("Bảng tính tay RK3/RK4 (rk_table.py)", "rk_table.py"),
    "2": ("Thực nghiệm so sánh, đồ thị log-log (thuc_nghiem.py)", "thuc_nghiem.py"),
    "3": ("Bài toán ứng dụng: vật rơi (bai_toan_ung_dung_sv7.py)", "bai_toan_ung_dung_sv7.py"),
    "4": ("Chạy bộ kiểm thử pytest -v", None),
}

def main():
    while True:
        print("\n=== NHÓM 8 - RUNGE-KUTTA RK3, RK4 ===")
        for k, (name, _) in MENU.items():
            print(f"{k}. {name}")
        print("0. Thoát")
        c = input("Chọn: ").strip()
        if c == "0":
            break
        if c == "4":
            subprocess.run([sys.executable, "-m", "pytest", "-v"], cwd=ROOT)
        elif c in MENU:
            runpy.run_path(os.path.join(ROOT, MENU[c][1]), run_name="__main__")
        else:
            print("Lựa chọn không hợp lệ.")

if __name__ == "__main__":
    main()
