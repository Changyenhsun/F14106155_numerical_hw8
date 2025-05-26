import numpy as np
import scipy.integrate as integrate

def f(x):
    return 0.5 * np.cos(x) + 0.25 * np.sin(2 * x)

phi_0 = lambda x: 1
phi_1 = lambda x: x
phi_2 = lambda x: x**2

def inner_product(f1, f2):
    return integrate.quad(lambda x: f1(x) * f2(x), -1, 1)[0]

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

coeffs = np.linalg.solve(A, b)

def P2(x):
    return coeffs[0] + coeffs[1] * x + coeffs[2] * x**2

def error_function(x):
    return (f(x) - P2(x))**2

sse, _ = integrate.quad(error_function, -1, 1)


print()
print(f"P2(x) = {coeffs[0]:.5f} + {coeffs[1]:.5f}x + ({coeffs[2]:.5f})x^2")
print(f"error = {sse:.5f}")
print()