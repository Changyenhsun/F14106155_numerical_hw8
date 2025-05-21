import numpy as np
import scipy.integrate as integrate

def f(x):
    return 0.5 * np.cos(x) + 0.25 * np.sin(2 * x)

phi_0 = lambda x: 1
phi_1 = lambda x: x
phi_2 = lambda x: x**2

# 內積函數 <f, g> = ∫ f(x) * g(x) dx from -1 to 1
def inner_product(f1, f2):
    return integrate.quad(lambda x: f1(x) * f2(x), -1, 1)[0]

# 計算矩陣與右邊常數項
A = np.array([
    [inner_product(phi_0, phi_0), inner_product(phi_0, phi_1), inner_product(phi_0, phi_2)],
    [inner_product(phi_1, phi_0), inner_product(phi_1, phi_1), inner_product(phi_1, phi_2)],
    [inner_product(phi_2, phi_0), inner_product(phi_2, phi_1), inner_product(phi_2, phi_2)]
])

b = np.array([
    inner_product(f, phi_0),
    inner_product(f, phi_1),
    inner_product(f, phi_2)
])

# 解聯立方程式得到 a0, a1, a2
coeffs = np.linalg.solve(A, b)

# 顯示結果
print()
print(f"a0 = {coeffs[0]:.3f}, a1 = {coeffs[1]:.3f}, a2 = {coeffs[2]:.3f}")
print(f"P2(x) = {coeffs[0]:.3f} + {coeffs[1]:.3f}x + ({coeffs[2]:.3f})x^2")
print()
