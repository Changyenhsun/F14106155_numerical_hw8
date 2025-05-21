import numpy as np
import scipy.integrate as integrate

m = 16
n = 4 
z_i = np.linspace(-np.pi, np.pi, 2 * m)
x_i = (z_i + np.pi) / (2 * np.pi)  
f_i = x_i**2 * np.sin(x_i)         # f(x_i) = x^2 sin x

# 計算傅立葉係數 a_k, b_k
a_k = np.zeros(n)
b_k = np.zeros(n)
for k in range(n):
    a_k[k] = (1 / m) * np.sum(f_i * np.cos(k * z_i))
    b_k[k] = (1 / m) * np.sum(f_i * np.sin(k * z_i))

# 定義 S_4(x)
def S4(x):
    z = 2 * np.pi * x - np.pi
    result = 0.5 * a_k[0]
    for k in range(1, n):
        result += a_k[k] * np.cos(k * z) + b_k[k] * np.sin(k * z)
    return result

# (b) 計算 ∫_0^1 S4(x) dx
S4_integral, _ = integrate.quad(S4, 0, 1)

# (c) 真正的 ∫_0^1 x^2 sin x dx
actual_integral, _ = integrate.quad(lambda x: x**2 * np.sin(x), 0, 1)

# (d) 誤差 E(S4) 
S4_vals = S4(x_i)
error = np.sum((f_i - S4_vals) ** 2)

# 輸出
print()
print("=== (a) S_4(x) coefficients ===")
for k in range(n):
    print(f"a_{k} = {a_k[k]:.3f}")
    if k != 0:
        print(f"b_{k} = {b_k[k]:.3f}")
        print()

print("\n=== (b) ∫_0^1 S4(x)dx ===")
print(f"∫_0^1 S4(x)dx = {S4_integral:.3f}")

print("\n=== (c) ∫_0^1 x^2 sin(x)dx ===")
print(f"∫_0^1 x^2 sin(x)dx = {actual_integral:.3f}")

print("\n=== (d) Error E(S4) ===")
print(f"Error = {error:.3f}")
print()