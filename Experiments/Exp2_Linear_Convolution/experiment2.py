"""
Experiment 2: Linear Convolution
Objective: To implement and understand linear convolution of discrete-time signals
"""

import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('outputs', exist_ok=True)

print("=" * 50)
print("EXPERIMENT 2: LINEAR CONVOLUTION")
print("=" * 50)
print()

# Define input signals
x = np.array([1, 2, 3, 4])
h = np.array([1, 1, 1])

print(f"Input Signal x: {x}")
print(f"Impulse Response h: {h}")
print()

# Perform linear convolution using NumPy
y = np.convolve(x, h, mode='full')

print(f"Convolution Result y: {y}")
print(f"Length of x: {len(x)}")
print(f"Length of h: {len(h)}")
print(f"Length of y: {len(y)} (should be {len(x) + len(h) - 1})")
print()

# Manual convolution for verification
print("Manual Convolution Calculation:")
M = len(x)
N = len(h)
L = M + N - 1
y_manual = np.zeros(L)

for n in range(L):
    for k in range(M):
        if 0 <= n - k < N:
            y_manual[n] += x[k] * h[n - k]
            print(f"  y[{n}] += x[{k}] * h[{n-k}] = {x[k]} * {h[n-k]}")
    print(f"  y[{n}] = {y_manual[n]:.0f}")

print()

# Visualization
fig, axes = plt.subplots(3, 1, figsize=(10, 8))
fig.suptitle('Linear Convolution', fontsize=16, fontweight='bold')

# Plot input signal x
axes[0].stem(range(len(x)), x, basefmt=' ')
axes[0].set_title('Input Signal x[n]', fontweight='bold')
axes[0].set_xlabel('n')
axes[0].set_ylabel('Amplitude')
axes[0].grid(True, alpha=0.3)

# Plot impulse response h
axes[1].stem(range(len(h)), h, basefmt=' ')
axes[1].set_title('Impulse Response h[n]', fontweight='bold')
axes[1].set_xlabel('n')
axes[1].set_ylabel('Amplitude')
axes[1].grid(True, alpha=0.3)

# Plot convolution result y
axes[2].stem(range(len(y)), y, basefmt=' ')
axes[2].set_title('Convolution Result y[n] = x[n] * h[n]', fontweight='bold')
axes[2].set_xlabel('n')
axes[2].set_ylabel('Amplitude')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/linear_convolution.png', dpi=300, bbox_inches='tight')
print("[SUCCESS] Plot saved to outputs/linear_convolution.png")
print()

# Additional example
print("=" * 50)
print("EXAMPLE 2: Different Signals")
print("=" * 50)

x2 = np.array([1, 2, 3, 2, 1])
h2 = np.array([1, -1, 1])
y2 = np.convolve(x2, h2, mode='full')

print(f"x2: {x2}")
print(f"h2: {h2}")
print(f"y2: {y2}")
print()

fig2, axes2 = plt.subplots(3, 1, figsize=(10, 8))
fig2.suptitle('Linear Convolution - Example 2', fontsize=16, fontweight='bold')

axes2[0].stem(range(len(x2)), x2, basefmt=' ')
axes2[0].set_title('Signal x2[n]', fontweight='bold')
axes2[0].grid(True, alpha=0.3)

axes2[1].stem(range(len(h2)), h2, basefmt=' ')
axes2[1].set_title('Signal h2[n]', fontweight='bold')
axes2[1].grid(True, alpha=0.3)

axes2[2].stem(range(len(y2)), y2, basefmt=' ')
axes2[2].set_title('Convolution y2[n]', fontweight='bold')
axes2[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/linear_convolution_example2.png', dpi=300, bbox_inches='tight')
print("[SUCCESS] Example 2 saved")
print()
print("=" * 50)
print("EXPERIMENT 2 COMPLETED SUCCESSFULLY!")
print("=" * 50)
