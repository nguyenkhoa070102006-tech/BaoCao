import numpy as np
# PHẦN 1: HÀM THUẬT TOÁN RK3
def rk3(f, a, b, y0, n) -> tuple[np.ndarray, np.ndarray]: #trang 138 thuật toán 11 dòng 1 
    """ 
    Phuong phap Rungge -Kutta bac 3 giai phuong trinh vi phan y' = f(x,y)
    Tham khao thuat toan 11 trang 138 

    Require:
        f   : ham so f(x,y) trong phuong trinh y'= f(x,y)
        a,b : dau mut trai va phai cua khoang tinh (a < b)
        y0  : gia tri ban dau y(a) = y0
        n   : so buoc chia (n > 0)

    Ensure : 
       Tra ve tuple gom hai numpy array(x,y) chua cac diem luoi va nghiem gan dung 
    """ 
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
