"""
Experiment 3: Cross-Correlation and Circular Convolution
Objective: To implement cross-correlation and circular convolution of discrete signals
"""

import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('outputs', exist_ok=True)

print("=" * 50)
print("EXPERIMENT 3: CORRELATION & CIRCULAR CONVOLUTION")
print("=" * 50)
print()

x = np.array([1, 2, 3, 4])
h = np.array([1, 1, 1, 1])

print(f"Signal x: {x}")
print(f"Signal h: {h}")
print()

# Cross-correlation
print("=" * 50)
print("CROSS-CORRELATION")
print("=" * 50)
print()

cross_corr = np.correlate(x, h, mode='full')
print(f"Cross-correlation result: {cross_corr}")
print()

print("Manual Calculation:")
for lag in range(-len(h)+1, len(x)):
    val = 0
    for i in range(len(x)):
        j = i - lag
        if 0 <= j < len(h):
            val += x[i] * h[j]
    print(f"  R_xh[{lag}] = {val}")
print()

# Circular Convolution
print("=" * 50)
print("CIRCULAR CONVOLUTION")
print("=" * 50)
print()

# Using FFT
X = np.fft.fft(x)
H = np.fft.fft(h)
Y_circular = np.fft.ifft(X * H).real

print(f"Circular Convolution (using FFT): {Y_circular}")

# Manual circular convolution
N = len(x)
y_circ_manual = np.zeros(N)
for n in range(N):
    for k in range(N):
        y_circ_manual[n] += x[k] * h[(n - k) % N]

print(f"Circular Convolution (manual): {y_circ_manual}")
print()

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Cross-Correlation and Circular Convolution', fontsize=16, fontweight='bold')

# Input signals
axes[0, 0].stem(range(len(x)), x, basefmt=' ')
axes[0, 0].set_title('Signal x[n]', fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].stem(range(len(h)), h, basefmt=' ')
axes[0, 1].set_title('Signal h[n]', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Cross-correlation
axes[1, 0].stem(range(len(cross_corr)), cross_corr, basefmt=' ')
axes[1, 0].set_title('Cross-Correlation R_xh[n]', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Circular convolution
axes[1, 1].stem(range(len(Y_circular)), Y_circular, basefmt=' ')
axes[1, 1].set_title('Circular Convolution y[n]', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/correlation_circular_convolution.png', dpi=300, bbox_inches='tight')
print("[SUCCESS] Plot saved")
print()

# Auto-correlation
auto_corr = np.correlate(x, x, mode='full')
fig2, ax = plt.subplots(figsize=(10, 5))
ax.stem(range(len(auto_corr)), auto_corr, basefmt=' ')
ax.set_title('Auto-Correlation of x[n]', fontweight='bold')
ax.grid(True, alpha=0.3)
plt.savefig('outputs/auto_correlation.png', dpi=300, bbox_inches='tight')
print("[SUCCESS] Auto-correlation saved")
print()
print("=" * 50)
print("EXPERIMENT 3 COMPLETED SUCCESSFULLY!")
print("=" * 50)
