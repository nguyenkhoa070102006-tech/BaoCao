import numpy as np

def rk4(f, a, b, y0, n) -> tuple[np.ndarray, np.ndarray]:
    """
    Phương pháp Runge-Kutta bậc 4 giải phương trình vi phân y' = f(x, y).

     Thuật toán Runge-Kutta bậc 4.

    Require:
        f: Hàm số f(x, y) trong phương trình vi phân y' = f(x, y).
        a, b: Hai đầu mút của đoạn [a, b] (a < b).
        y0: Giá trị ban đầu của hàm số tại x = a.
        n: Số bước chia, là số nguyên dương (n > 0).


    Ensure:
        Trả về tuple gồm hai numpy array (x, y),
        chứa các điểm lưới và nghiệm gần đúng tương ứng.
    """

    # Kiểm tra điều kiện áp dụng
    if a >= b:
        raise ValueError("Khoảng tính không hợp lệ: a phải nhỏ hơn b.")

    if not isinstance(n, (int, np.integer)) or isinstance(n, (bool, np.bool_)) or n <= 0:
        raise ValueError("Số bước chia n phải là số nguyên dương.")

    h = (b-a)/n
    x = [a]
    y = [y0]
    # Thuật toán Runge-Kutta bậc 4 trang 138
    # K1 = [h*func(a,y0)]
    K1 = [h*f(a,y0)]
    # K2 = [h*func(a + h/2 , y0 + k1[-1]/2)]
    K2 = [h*f(a+h/2,y0+K1[-1]/2)]
    # K3 = [h*func(a + h/2 , y0 + k2[-1]/2]
    K3 = [h*f(a+h/2,y0+K2[-1]/2)]
    # K4 = [h*func(a+h),y0 + K3[-1]]
    K4 = [h*f(a+h,y0+K3[-1])]

    for i in range(n): # tương tự thuật toán 11
        x.append(x[-1]+h)
        y.append(y[-1]+1/6*(K1[-1]+2*K2[-1]+2*K3[-1]+K4[-1]))
        K1.append(h*f(x[-1],y[-1]))
        K2.append(h*f(x[-1]+h/2,y[-1]+K1[-1]/2))
        K3.append(h*f(x[-1]+h/2,y[-1]+K2[-1]/2))
        K4.append(h*f(x[-1]+h,y[-1]+K3[-1]))

    return np.array(x), np.array(y)
