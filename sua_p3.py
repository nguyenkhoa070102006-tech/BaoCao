
import numpy as np

# Phuong phap Runge-Kutta bac 3
def rk3(f, x, y, h):
    K1 = h * f(x, y)
    K2 = h * f(x + h / 2, y + K1 / 2)
    K3 = h * f(x + h, y - K1 + 2 * K2)
    return y + (K1 + 4 * K2 + K3) / 6


# Phuong phap Runge-Kutta bac 4
def rk4(f, x, y, h):
    K1 = h * f(x, y)
    K2 = h * f(x + h / 2, y + K1 / 2)
    K3 = h * f(x + h / 2, y + K2 / 2)
    K4 = h * f(x + h, y + K3)
    return y + (K1 + 2 * K2 + 2 * K3 + K4) / 6


def rk_table(method, f, a, b, y0, n) -> list[dict]:
    """
    Xuat bang tinh tay cua phuong phap Runge-Kutta bac 3 hoac bac 4.

    Require:
        method: ham rk3 hoac rk4 da cai dat o phan 1, 2.
        f: ham trong phuong trinh vi phan y' = f(x, y).
        a, b: hai dau mut khoang tinh, can thoa man a < b.
        y0: gia tri ban dau y(a) = y0.
        n: so buoc chia, can la so nguyen duong.

    Ensure:
        Tra ve danh sach cac dong trong bang tinh tay.
        Moi dong la mot dict chua:
            - i: chi so buoc tinh.
            - x_i: gia tri x tai buoc i.
            - y_i: nghiem gan dung tai buoc i.
            - K1, K2, K3: cac he so Runge-Kutta.
            - K4: he so bo sung neu su dung RK4.
            - y_next: nghiem gan dung tai buoc i+1.

    Thuat toan:
        1. Kiem tra dieu kien dau vao.
        2. Tinh buoc chia h = (b - a) / n.
        3. Khoi tao x = a, y = y0.
        4. Lap n buoc:
            - Tinh cac he so K theo cong thuc RK3 hoac RK4.
            - Tinh nghiem y_(i+1).
            - Luu cac gia tri vao bang.
            - Cap nhat x, y cho buoc tiep theo.
        5. Tra ve bang tinh tay
    """

    # BUOC 1: KIEM TRA DIEU KIEN DAU VAO

    # Kiem tra n phai la so nguyen duong.
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool):
        raise TypeError("n phai la so nguyen.")

    if n <= 0:
        raise ValueError("n phai lon hon 0.")

    # Kiem tra khoang tinh.
    if a >= b:
        raise ValueError("Can thoa man a < b.")

    # Kiem tra ham f.
    if not callable(f):
        raise TypeError("f phai la mot ham.")

    # Kiem tra method phai la ham rk3 hoac rk4.
    if not callable(method):
        raise TypeError("method phai la ham rk3 hoac rk4.")

    # Lay ten phuong phap.
    method_name = method.__name__

    if method_name not in ("rk3", "rk4"):
        raise ValueError("method phai la ham rk3 hoac rk4.")

    # BUOC 2: KHOI TAO GIA TRI

    # Tinh buoc chia h = (b - a) / n.
    h = (b - a) / n

    # Khoi tao gia tri ban dau.
    x = a
    y = y0

    # Danh sach luu bang tinh tay.
    table = []

    # BUOC 3: LAP QUA CAC BUOC TINH

    for i in range(n):

        # Khoi tao mot dong bang voi cac gia tri hien tai.
        row = {
            "i": i,
            "x_i": x,
            "y_i": y
        }

        # TRUONG HOP 1: RUNGE-KUTTA BAC 3

        if method_name == "rk3":

            # K1 = h * f(x_i, y_i)
            K1 = h * f(x, y)

            # K2 = h * f(x_i + h/2, y_i + K1/2)
            K2 = h * f(
                x + h / 2,
                y + K1 / 2
            )

            # K3 = h * f(x_i + h, y_i - K1 + 2*K2)
            K3 = h * f(
                x + h,
                y - K1 + 2 * K2
            )

            # Cong thuc cap nhat nghiem RK3:
            # y_(i+1) = y_i + (K1 + 4*K2 + K3)/6
            y_next = y + (
                K1 + 4 * K2 + K3
            ) / 6

            # Luu cac he so K vao dong bang.
            row["K1"] = K1
            row["K2"] = K2
            row["K3"] = K3

        # TRUONG HOP 2: RUNGE-KUTTA BAC 4

        else:

            # K1 = h * f(x_i, y_i)
            K1 = h * f(x, y)

            # K2 = h * f(x_i + h/2, y_i + K1/2)
            K2 = h * f(
                x + h / 2,
                y + K1 / 2
            )

            # K3 = h * f(x_i + h/2, y_i + K2/2)
            K3 = h * f(
                x + h / 2,
                y + K2 / 2
            )

            # K4 = h * f(x_i + h, y_i + K3)
            K4 = h * f(
                x + h,
                y + K3
            )

            # Cong thuc cap nhat nghiem RK4:
            # y_(i+1) = y_i + (K1 + 2*K2 + 2*K3 + K4)/6
            y_next = y + (
                K1 + 2 * K2 + 2 * K3 + K4
            ) / 6

            # Luu cac he so K vao dong bang.
            row["K1"] = K1
            row["K2"] = K2
            row["K3"] = K3
            row["K4"] = K4

        # BUOC 4: LUU NGHIEM TAI BUOC TIEP THEO

        # Gia tri x tai buoc tiep theo.
        x_next = x + h

        # Luu x_(i+1), y_(i+1) vao dong bang.
        row["x_next"] = x_next
        row["y_next"] = y_next

        # Them dong vao bang tinh tay.
        table.append(row)

        # Cap nhat x, y cho buoc tinh tiep theo.
        x = x_next
        y = y_next

    # BUOC 5: TRA VE BANG TINH TAY
    return table


# Phuong trinh vi phan
def f(x, y):
    return y - x**2 + 1


# Xuat bang tinh tay RK3
table_rk3 = rk_table(rk3, f, 0, 2, 0.5, 10)

# Xuat bang tinh tay RK4
table_rk4 = rk_table(rk4, f, 0, 2, 0.5, 10)


# In bang RK3
print("BANG TINH TAY RK3")
for row in table_rk3:
    print(row)


# In bang RK4
print("\nBANG TINH TAY RK4")
for row in table_rk4:
    print(row)
