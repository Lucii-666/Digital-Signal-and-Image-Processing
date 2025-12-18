"""
Experiment 5: FIR Filter using Windowing Method
Objective: To design FIR filters using different windowing techniques
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

os.makedirs('outputs', exist_ok=True)

print("=" * 50)
print("EXPERIMENT 5: FIR FILTER DESIGN")
print("=" * 50)

fs = 1000
fc = 100
numtaps = 51

print(f"Sampling Frequency: {fs} Hz")
print(f"Cutoff Frequency: {fc} Hz")
print(f"Number of Taps: {numtaps}")
print()

windows = ['boxcar', 'hamming', 'hann', 'blackman', ('kaiser', 5)]
filters = {}

for window in windows:
    window_name = window if isinstance(window, str) else window[0]
    fir = signal.firwin(numtaps, fc, window=window, fs=fs)
    filters[window_name] = fir
    
    print(f"{window_name.upper()} Window:")
    print(f"  Filter coefficients (first 5): {fir[:5]}")
    print(f"  Filter coefficients (last 5): {fir[-5:]}")
    print()

# Frequency responses
fig, axes = plt.subplots(len(filters), 1, figsize=(12, 10))
fig.suptitle('FIR Filter Magnitude Responses', fontsize=16, fontweight='bold')

for idx, (window_name, fir) in enumerate(filters.items()):
    w, h = signal.freqz(fir, worN=2000, fs=fs)
    axes[idx].plot(w, 20 * np.log10(abs(h)), linewidth=2)
    axes[idx].axvline(fc, color='r', linestyle='--', alpha=0.7)
    axes[idx].set_title(f'{window_name.capitalize()} Window', fontweight='bold')
    axes[idx].set_ylabel('Magnitude (dB)')
    axes[idx].grid(True, alpha=0.3)
    axes[idx].set_ylim(-100, 5)

axes[-1].set_xlabel('Frequency (Hz)')
plt.tight_layout()
plt.savefig('outputs/fir_magnitude_responses.png', dpi=300, bbox_inches='tight')
print("[SUCCESS] Magnitude responses saved")
print()

# Impulse responses
fig2, axes2 = plt.subplots(len(filters), 1, figsize=(12, 10))
fig2.suptitle('FIR Filter Impulse Responses', fontsize=16, fontweight='bold')

for idx, (window_name, fir) in enumerate(filters.items()):
    axes2[idx].stem(range(numtaps), fir, basefmt=' ')
    axes2[idx].set_title(f'{window_name.capitalize()} Window', fontweight='bold')
    axes2[idx].set_ylabel('Amplitude')
    axes2[idx].grid(True, alpha=0.3)

axes2[-1].set_xlabel('Sample Number')
plt.tight_layout()
plt.savefig('outputs/fir_impulse_responses.png', dpi=300, bbox_inches='tight')
print("[SUCCESS] Impulse responses saved")
print()

# Test on signal
t = np.linspace(0, 1, fs, endpoint=False)
test_signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 200 * t)

fig3, axes3 = plt.subplots(3, 1, figsize=(12, 8))
axes3[0].plot(t[:200], test_signal[:200])
axes3[0].set_title('Original Signal (50Hz + 200Hz)', fontweight='bold')
axes3[0].grid(True, alpha=0.3)

filtered = signal.lfilter(filters['hamming'], 1.0, test_signal)
axes3[1].plot(t[:200], filtered[:200], color='orange')
axes3[1].set_title('Filtered (Hamming Window)', fontweight='bold')
axes3[1].grid(True, alpha=0.3)

axes3[2].plot(t[:200], test_signal[:200], label='Original', alpha=0.7)
axes3[2].plot(t[:200], filtered[:200], label='Filtered', linewidth=2)
axes3[2].set_title('Comparison', fontweight='bold')
axes3[2].set_xlabel('Time (s)')
axes3[2].legend()
axes3[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/fir_filtered_signal.png', dpi=300, bbox_inches='tight')
print("[SUCCESS] Filtered signal saved")
print()
print("=" * 50)
print("EXPERIMENT 5 COMPLETED SUCCESSFULLY!")
print("=" * 50)
