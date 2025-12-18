"""
Experiment 4: Butterworth and Chebyshev IIR Filters
Objective: To design and analyze Butterworth and Chebyshev IIR filters
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

os.makedirs('outputs', exist_ok=True)

print("=" * 50)
print("EXPERIMENT 4: IIR FILTER DESIGN")
print("=" * 50)

# Filter specifications
fs = 1000  # Sampling frequency
fc = 100   # Cutoff frequency
order = 4  # Filter order

print(f"Sampling Frequency: {fs} Hz")
print(f"Cutoff Frequency: {fc} Hz")
print(f"Filter Order: {order}")
print()

# Butterworth Filter
print("=" * 50)
print("BUTTERWORTH FILTER")
print("=" * 50)
b_butter, a_butter = signal.butter(order, fc, fs=fs, btype='low')
print(f"Numerator coefficients (b): {b_butter}")
print(f"Denominator coefficients (a): {a_butter}")
print()

# Chebyshev Type I Filter
print("=" * 50)
print("CHEBYSHEV TYPE I FILTER")
print("=" * 50)
b_cheby1, a_cheby1 = signal.cheby1(order, 0.5, fc, fs=fs, btype='low')
print(f"Numerator coefficients (b): {b_cheby1}")
print(f"Denominator coefficients (a): {a_cheby1}")
print()

# Chebyshev Type II Filter
print("=" * 50)
print("CHEBYSHEV TYPE II FILTER")
print("=" * 50)
b_cheby2, a_cheby2 = signal.cheby2(order, 40, fc, fs=fs, btype='low')
print(f"Numerator coefficients (b): {b_cheby2}")
print(f"Denominator coefficients (a): {a_cheby2}")
print()

# Frequency response
w, h_butter = signal.freqz(b_butter, a_butter, worN=2000, fs=fs)
_, h_cheby1 = signal.freqz(b_cheby1, a_cheby1, worN=2000, fs=fs)
_, h_cheby2 = signal.freqz(b_cheby2, a_cheby2, worN=2000, fs=fs)

# Plot magnitude and phase responses
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
fig.suptitle('IIR Filter Frequency Responses', fontsize=16, fontweight='bold')

# Magnitude response
axes[0].plot(w, 20 * np.log10(abs(h_butter)), label='Butterworth', linewidth=2)
axes[0].plot(w, 20 * np.log10(abs(h_cheby1)), label='Chebyshev I', linewidth=2)
axes[0].plot(w, 20 * np.log10(abs(h_cheby2)), label='Chebyshev II', linewidth=2)
axes[0].axvline(fc, color='r', linestyle='--', label=f'Cutoff = {fc} Hz')
axes[0].set_title('Magnitude Response', fontweight='bold')
axes[0].set_xlabel('Frequency (Hz)')
axes[0].set_ylabel('Magnitude (dB)')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Phase response
axes[1].plot(w, np.angle(h_butter), label='Butterworth', linewidth=2)
axes[1].plot(w, np.angle(h_cheby1), label='Chebyshev I', linewidth=2)
axes[1].plot(w, np.angle(h_cheby2), label='Chebyshev II', linewidth=2)
axes[1].set_title('Phase Response', fontweight='bold')
axes[1].set_xlabel('Frequency (Hz)')
axes[1].set_ylabel('Phase (radians)')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.savefig('outputs/filter_responses.png', dpi=300, bbox_inches='tight')
print("[SUCCESS] Filter responses saved")
print()

# Test with signal
print("=" * 50)
print("APPLYING FILTERS TO TEST SIGNAL")
print("=" * 50)
print()

t = np.linspace(0, 1, fs, endpoint=False)
test_signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 150 * t) + 0.3 * np.sin(2 * np.pi * 300 * t)

filtered_butter = signal.lfilter(b_butter, a_butter, test_signal)
filtered_cheby1 = signal.lfilter(b_cheby1, a_cheby1, test_signal)
filtered_cheby2 = signal.lfilter(b_cheby2, a_cheby2, test_signal)

fig2, axes2 = plt.subplots(4, 1, figsize=(12, 10))
fig2.suptitle('Filter Application on Test Signal', fontsize=16, fontweight='bold')

axes2[0].plot(t[:500], test_signal[:500], linewidth=1.5)
axes2[0].set_title('Original Signal (50Hz + 150Hz + 300Hz)', fontweight='bold')
axes2[0].set_ylabel('Amplitude')
axes2[0].grid(True, alpha=0.3)

axes2[1].plot(t[:500], filtered_butter[:500], linewidth=1.5, color='orange')
axes2[1].set_title('Butterworth Filtered', fontweight='bold')
axes2[1].set_ylabel('Amplitude')
axes2[1].grid(True, alpha=0.3)

axes2[2].plot(t[:500], filtered_cheby1[:500], linewidth=1.5, color='green')
axes2[2].set_title('Chebyshev I Filtered', fontweight='bold')
axes2[2].set_ylabel('Amplitude')
axes2[2].grid(True, alpha=0.3)

axes2[3].plot(t[:500], filtered_cheby2[:500], linewidth=1.5, color='red')
axes2[3].set_title('Chebyshev II Filtered', fontweight='bold')
axes2[3].set_xlabel('Time (s)')
axes2[3].set_ylabel('Amplitude')
axes2[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/filtered_signals.png', dpi=300, bbox_inches='tight')
print("[SUCCESS] Filtered signals saved")
print()
print("=" * 50)
print("EXPERIMENT 4 COMPLETED SUCCESSFULLY!")
print("=" * 50)
