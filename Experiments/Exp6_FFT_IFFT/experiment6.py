"""
Experiment 6: FFT and IFFT on Discrete Time Signals
Objective: To perform FFT and IFFT analysis on discrete-time signals
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

os.makedirs('outputs', exist_ok=True)

print("=" * 50)
print("EXPERIMENT 6: FFT AND IFFT ANALYSIS")
print("=" * 50)

# Signal parameters
fs = 1000
N = 1000
t = np.arange(N) / fs

# Create composite signal
f1, f2, f3 = 50, 120, 200
signal_data = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t) + 0.3 * np.sin(2 * np.pi * f3 * t)

print(f"Signal Components: {f1}Hz, {f2}Hz, {f3}Hz")
print(f"Sampling Frequency: {fs}Hz")
print(f"Signal Length: {N} samples")
print()

# FFT
fft_result = np.fft.fft(signal_data)
frequencies = np.fft.fftfreq(N, 1/fs)
magnitude = np.abs(fft_result)
phase = np.angle(fft_result)

print(f"FFT Result:")
print(f"  Length: {len(fft_result)}")
print(f"  Max magnitude: {np.max(magnitude):.2f}")

# Find dominant frequencies
dominant_indices = np.where(magnitude > N/2 * 0.25)[0]
dominant_freqs = np.abs(frequencies[dominant_indices])
dominant_freqs = np.unique(dominant_freqs[dominant_freqs > 0])
print(f"  Dominant frequencies detected: {dominant_freqs[:3].tolist()}")
print()

# IFFT
reconstructed = np.fft.ifft(fft_result).real
error = np.max(np.abs(signal_data - reconstructed))
print(f"Reconstruction Error: {error:.2e}")
print()

# Visualization
fig, axes = plt.subplots(3, 2, figsize=(14, 10))
fig.suptitle('FFT and IFFT Analysis', fontsize=16, fontweight='bold')

# Time domain - original
axes[0, 0].plot(t[:200], signal_data[:200], linewidth=1.5)
axes[0, 0].set_title('Original Signal (Time Domain)', fontweight='bold')
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('Amplitude')
axes[0, 0].grid(True, alpha=0.3)

# Frequency domain - magnitude
positive_freq_indices = frequencies > 0
axes[0, 1].stem(frequencies[positive_freq_indices][:500], magnitude[positive_freq_indices][:500], basefmt=' ')
axes[0, 1].set_title('FFT Magnitude Spectrum', fontweight='bold')
axes[0, 1].set_xlabel('Frequency (Hz)')
axes[0, 1].set_ylabel('Magnitude')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xlim(0, 300)

# Frequency domain - phase
axes[1, 0].plot(frequencies[positive_freq_indices][:500], phase[positive_freq_indices][:500], linewidth=1.5)
axes[1, 0].set_title('FFT Phase Spectrum', fontweight='bold')
axes[1, 0].set_xlabel('Frequency (Hz)')
axes[1, 0].set_ylabel('Phase (radians)')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_xlim(0, 300)

# Reconstructed signal
axes[1, 1].plot(t[:200], reconstructed[:200], linewidth=1.5, color='orange')
axes[1, 1].set_title('Reconstructed Signal (IFFT)', fontweight='bold')
axes[1, 1].set_xlabel('Time (s)')
axes[1, 1].set_ylabel('Amplitude')
axes[1, 1].grid(True, alpha=0.3)

# Comparison
axes[2, 0].plot(t[:200], signal_data[:200], label='Original', linewidth=2)
axes[2, 0].plot(t[:200], reconstructed[:200], '--', label='Reconstructed', linewidth=2)
axes[2, 0].set_title('Original vs Reconstructed', fontweight='bold')
axes[2, 0].set_xlabel('Time (s)')
axes[2, 0].set_ylabel('Amplitude')
axes[2, 0].legend()
axes[2, 0].grid(True, alpha=0.3)

# Error
axes[2, 1].plot(t[:200], signal_data[:200] - reconstructed[:200], color='red', linewidth=1.5)
axes[2, 1].set_title('Reconstruction Error', fontweight='bold')
axes[2, 1].set_xlabel('Time (s)')
axes[2, 1].set_ylabel('Error')
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/fft_ifft_analysis.png', dpi=300, bbox_inches='tight')
print("[SUCCESS] FFT/IFFT analysis saved")
print()

# Different signal types
fig2, axes2 = plt.subplots(4, 2, figsize=(14, 12))
fig2.suptitle('FFT of Different Signals', fontsize=16, fontweight='bold')

signals = {
    'Sine Wave': np.sin(2 * np.pi * 50 * t),
    'Square Wave': signal.square(2 * np.pi * 50 * t),
    'Sawtooth Wave': signal.sawtooth(2 * np.pi * 50 * t),
    'Chirp Signal': signal.chirp(t, f0=20, f1=200, t1=1, method='linear')
}
for idx, (name, sig) in enumerate(signals.items()):
    fft_sig = np.fft.fft(sig)
    mag_sig = np.abs(fft_sig)
    
    axes2[idx, 0].plot(t[:200], sig[:200], linewidth=1.5)
    axes2[idx, 0].set_title(f'{name} - Time Domain', fontweight='bold')
    axes2[idx, 0].set_ylabel('Amplitude')
    axes2[idx, 0].grid(True, alpha=0.3)
    
    axes2[idx, 1].stem(frequencies[positive_freq_indices][:200], mag_sig[positive_freq_indices][:200], basefmt=' ')
    axes2[idx, 1].set_title(f'{name} - Frequency Domain', fontweight='bold')
    axes2[idx, 1].set_ylabel('Magnitude')
    axes2[idx, 1].grid(True, alpha=0.3)
    axes2[idx, 1].set_xlim(0, 300)

axes2[-1, 0].set_xlabel('Time (s)')
axes2[-1, 1].set_xlabel('Frequency (Hz)')

plt.tight_layout()
plt.savefig('outputs/fft_different_signals.png', dpi=300, bbox_inches='tight')
print("[SUCCESS] Different signals FFT saved")
print()
print("=" * 50)
print("EXPERIMENT 6 COMPLETED SUCCESSFULLY!")
print("=" * 50)
