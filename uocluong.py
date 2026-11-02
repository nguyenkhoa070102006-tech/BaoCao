import numpy as np

#ham order_estimate
def order_estimate(method, f, a, b, y0, n_list) -> list[float]:  #muc chu ky ham
    """
    Uoc luong bac hoi tu thuc nghiem cua phuong phap RK.

    Require:
        method: ham rk3 hoac rk4.
        f: ham trong phuong trinh y' = f(x, y).
        a < b, y0 la dieu kien ban dau.
        n_list: danh sach so buoc nguyen duong, tang gap doi.

    Ensure:
        Tra ve danh sach bac hoi tuong ung voi cac cap
        so buoc lien tiep trong n_list.
        Moi phan tu duoc tinh tu sai khac hai luoi.

    Thuat toan:
        Uoc luong bac hoi tu bang log2(D_n / D_2n).
    """
    # Kiem tra dieu kien dau vao.
    if a >= b:                                     #R6-xu ly tinh huong bien
        raise ValueError("Can thoa man a < b.")

    if not n_list:
        raise ValueError("n_list khong duoc rong.")

    if any(not isinstance(n, (int, np.integer))
           or isinstance(n, bool) or n <= 0 for n in n_list):
        raise ValueError("Moi so buoc trong n_list phai la so nguyen duong.")

    if any(n_list[i + 1] != 2 * n_list[i]
           for i in range(len(n_list) - 1)):
        raise ValueError("Cac phan tu n_list phai tang gap doi.")

    # Tinh sai khac D_n = |y_n(b) - y_2n(b)|.
    errors = []

    for n in n_list:
        # Luoi tho co n buoc.
        _, y_coarse = method(f, a, b, y0, n)       #ham rk3,rk4 cua sv1,sv2

        # Luoi min co 2n buoc.
        _, y_fine = method(f, a, b, y0, 2 * n)     #cong thuc buoc luoi: dung 2n buoc de co h/2

        # Sai khac giua hai nghiem tai diem cuoi b.
        D_n = abs(y_coarse[-1] - y_fine[-1])        #cong thuc uoc luong sai khac hai luoi trong p4. Tinh Dn=|y_n(b)-y_2n(b)|
        errors.append(D_n)

    # Tinh bac hoi tu tu hai sai khac lien tiep.
    orders = []

    for i in range(len(errors) - 1):
        if errors[i] > 0 and errors[i + 1] > 0:
            p = np.log2(errors[i] / errors[i + 1]) #cong thuc p4 trong de
            orders.append(float(p))
        else:
            raise ArithmeticError(
                "Khong the uoc luong bac hoi tu khi sai khac bang 0."
            )

    return orders

#ham step_for_tol
def step_for_tol(method, f, a, b, y0, tol) -> tuple[float, float]:     #muc chu ky ham
    """
    Tim buoc luoi lon nhat trong day chia doi thoa man
    sai khac giua hai nghiem tren luoi h va h/2 khong vuot tol.

    Require:
        method: ham rk3 hoac rk4.
        f: ham trong phuong trinh y' = f(x, y).
        a < b, tol > 0.

    Ensure:
        Tra ve (h, y_b), trong do h la buoc luoi duoc chap nhan
        va y_b la nghiem tai b tinh voi buoc h.

    Thuat toan:
        Bat dau n = 1; so sanh nghiem voi n va 2n buoc.
        Neu chua dat, tang n gap doi va lap lai.
    """
    # Kiem tra dieu kien dau vao.
    if a >= b:                                   #R6: kiem tra dieu kien khoang tinh toan
        raise ValueError("Can thoa man a < b.")

    if tol <= 0:
        raise ValueError("tol phai lon hon 0.")

    # Bat dau voi mot buoc tren [a, b].
    n = 1
    max_n = 1_000_000

    while n <= max_n:
        # Nghiem tren luoi tho co n buoc.
        _, y_coarse = method(f, a, b, y0, n)       #rk3,rk4 cua sv1,sv2

        # Nghiem tren luoi min co 2n buoc.
        _, y_fine = method(f, a, b, y0, 2 * n)      #yeu cau cua p4: tinh nghiem voi 2n buoc de so sanh buoc h va h/2

        # Uoc luong sai khac tai diem cuoi b.
        error_estimate = abs(y_coarse[-1] - y_fine[-1])  #cong thuc p4: tinh En=|y_n(b)-y_2n(b)|

        # Neu dat nguong, tra ve buoc luoi lon hon h.
        if error_estimate <= tol:
            h = (b - a) / n
            return float(h), float(y_coarse[-1])

        # Chua dat: tang so buoc gap doi, tuc giam h mot nua.
        n *= 2

    raise RuntimeError(
        "Khong tim duoc buoc luoi thoa man trong gioi han tim kiem."
    )
