import numpy as np

x = np.array([4.0, 4.2, 4.5, 4.7, 5.1, 5.5, 5.9, 6.3])
y = np.array([102.6, 113.2, 130.1, 142.1, 167.5, 195.1, 224.9, 256.8])

# (a) Degree 2 Polynomial Least Squares 
coeffs_deg2 = np.polyfit(x, y, deg=2)
poly_deg2 = np.poly1d(coeffs_deg2)
y_fit_deg2 = poly_deg2(x)
sse_deg2 = np.sum((y - y_fit_deg2) ** 2)

# (b) y = b * e^(a x)
ln_y = np.log(y)
A = np.vstack([x, np.ones_like(x)]).T
a_exp, ln_b = np.linalg.lstsq(A, ln_y, rcond=None)[0]
b_exp = np.exp(ln_b)
y_fit_exp = b_exp * np.exp(a_exp * x)
sse_exp = np.sum((y - y_fit_exp) ** 2)

# (c) y = b * x^n 
ln_x = np.log(x)
B = np.vstack([ln_x, np.ones_like(ln_x)]).T
n_pow, ln_b_pow = np.linalg.lstsq(B, ln_y, rcond=None)[0]
b_pow = np.exp(ln_b_pow)
y_fit_pow = b_pow * x ** n_pow
sse_pow = np.sum((y - y_fit_pow) ** 2)

# 輸出結果 
print()
print("=== (a) Degree 2 Polynomial ===")
print(f"y ≈ {coeffs_deg2[0]:.3f} x^2 + ({coeffs_deg2[1]:.3f})x + {coeffs_deg2[2]:.3f}")
print(f"error: {sse_deg2:.3f}")

print("\n=== (b) Exponential Model ===")
print(f"a = {a_exp:.3f}, b = {b_exp:.3f}")
print(f"y ≈ {b_exp:.3f} * e^ {a_exp:.3f}x")
print(f"error: {sse_exp:.3f}")

print("\n=== (c) Power Model ===")
print(f"n = {n_pow:.3f}, b = {b_pow:.3f}")
print(f"y ≈ {b_pow:.3f} * x^ {n_pow:.3f}")
print(f"error: {sse_pow:.3f}")
print()
