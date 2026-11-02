import numpy as np
# PHẦN 1: HÀM THUẬT TOÁN RK3
def rk3(f, a, b, y0, n) -> tuple[np.ndarray, np.ndarray]: #trang 138 thuật toán 11 dòng 1 
    """
    Phuong phap Runge-Kutta bac 3 giai phuong trinh vi phan y' = f(x, y).
    Tham khảo: Thuật toán 11, Trang 138

    Require:
        f: ham so f(x, y) trong phuong trinh vi phan y' = f(x, y).
        a, b: hai dau mut trai va phai cua khoang tinh (thoa man a < b).
        y0: gia tri ban dau y(a) = y0.
        n: so buoc chia, la so nguyen duong (n > 0).

    Ensure:
        Tra ve tuple gom hai numpy array (x, y) chua cac diem luoi va nghiem gan dung.
    """
    if a >= b:
        raise ValueError("Khoang tinh khong hop le: a phai nho hon b")
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n <= 0:
        raise ValueError("So buoc n phai la so nguyen duong")
    h = (b - a) / n
    x = [a]
    y = [y0]
    # Công thức RK3: ( trong giáo trình ct (6.20))
    # K1 = h * f(x, y)
    # K2 = h * f(x + h/2, y + K1/2)
    # K3 = h * f(x + h, y - K1 + 2*K2)
    # y_{n+1} = y_n + 1/6*(K1 + 4*K2 + K3)
    
    K1 = [h * f(a, y0)]
    K2 = [h * f(a + h/2, y0 + K1[-1]/2)]
    K3 = [h * f(a + h, y0 - K1[-1] + 2 * K2[-1])]
    
    for i in range(n): #tương tự thuật toán 11 
        x.append(x[-1] + h)
        y.append(y[-1] + 1/6 * (K1[-1] + 4 * K2[-1] + K3[-1]))
        
        K1.append(h * f(x[-1], y[-1]))
        K2.append(h * f(x[-1] + h/2, y[-1] + K1[-1]/2))
        K3.append(h * f(x[-1] + h, y[-1] - K1[-1] + 2 * K2[-1]))
        
    return np.array(x), np.array(y)
