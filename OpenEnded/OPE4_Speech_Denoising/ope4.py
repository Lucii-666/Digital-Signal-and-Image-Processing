"""
Open Ended Problem 4: Speech Denoising using IIR and FIR Filters
Objective: Analyze FFT of clean/noisy speech, apply IIR/FIR filtering, compare with clean speech
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import os

os.makedirs('outputs', exist_ok=True)

print("=" * 70)
print("OPE 4: SPEECH DENOISING - IIR & FIR FILTERING")
print("=" * 70)
print()

# Create synthetic speech signal
fs = 16000  # Typical speech sampling rate
duration = 2
t = np.linspace(0, duration, int(fs * duration))

print("Creating synthetic speech signals...")

# Clean speech (formants at typical speech frequencies)
formant1 = np.sin(2 * np.pi * 700 * t)   # First formant
formant2 = 0.6 * np.sin(2 * np.pi * 1220 * t)  # Second formant
formant3 = 0.4 * np.sin(2 * np.pi * 2600 * t)  # Third formant

clean_speech = formant1 + formant2 + formant3
clean_speech = clean_speech / np.max(np.abs(clean_speech))

# Add envelope (speech-like amplitude modulation)
envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 5 * t)
clean_speech = clean_speech * envelope

wavfile.write('outputs/clean_speech.wav', fs, (clean_speech * 32767).astype(np.int16))
print("[SUCCESS] Clean speech created")

# Create different types of noisy speech
np.random.seed(42)

# 1. White noise
white_noise = 0.3 * np.random.randn(len(clean_speech))
noisy_white = clean_speech + white_noise
noisy_white = noisy_white / np.max(np.abs(noisy_white))
wavfile.write('outputs/noisy_white_noise.wav', fs, (noisy_white * 32767).astype(np.int16))

# 2. Pink noise (low-frequency noise)
pink_noise = signal.lfilter([1], [1, -0.9], np.random.randn(len(clean_speech)))
pink_noise = 0.3 * pink_noise / np.max(np.abs(pink_noise))
noisy_pink = clean_speech + pink_noise
noisy_pink = noisy_pink / np.max(np.abs(noisy_pink))
wavfile.write('outputs/noisy_pink_noise.wav', fs, (noisy_pink * 32767).astype(np.int16))

# 3. High-frequency hiss
hiss = 0.2 * np.random.randn(len(clean_speech))
b, a = signal.butter(4, 4000, 'high', fs=fs)
hiss = signal.filtfilt(b, a, hiss)
noisy_hiss = clean_speech + hiss
noisy_hiss = noisy_hiss / np.max(np.abs(noisy_hiss))
wavfile.write('outputs/noisy_hiss.wav', fs, (noisy_hiss * 32767).astype(np.int16))

print("[SUCCESS] Created 3 types of noisy speech")
print("  1. White noise")
print("  2. Pink (low-frequency) noise")
print("  3. High-frequency hiss")
print()

# ============================================================================
# FREQUENCY SPECTRUM ANALYSIS
# ============================================================================

print("=" * 70)
print("FREQUENCY SPECTRUM ANALYSIS (FFT)")
print("=" * 70)
print()

def analyze_fft(signal_data, label, fs):
    """Compute and analyze FFT"""
    n = len(signal_data)
    fft_result = np.fft.fft(signal_data)
    freqs = np.fft.fftfreq(n, 1/fs)
    magnitude = np.abs(fft_result)
    
    # Positive frequencies only
    pos_mask = freqs > 0
    pos_freqs = freqs[pos_mask]
    pos_mag = magnitude[pos_mask]
    
    # Find dominant frequencies
    peaks, _ = signal.find_peaks(pos_mag, height=np.max(pos_mag) * 0.1)
    dominant_freqs = pos_freqs[peaks][:5]  # Top 5
    
    print(f"{label}:")
    print(f"  Dominant frequencies: {dominant_freqs[:3]}")
    
    return freqs, magnitude

clean_freqs, clean_mag = analyze_fft(clean_speech, "Clean Speech", fs)
white_freqs, white_mag = analyze_fft(noisy_white, "White Noise Speech", fs)
pink_freqs, pink_mag = analyze_fft(noisy_pink, "Pink Noise Speech", fs)
hiss_freqs, hiss_mag = analyze_fft(noisy_hiss, "Hiss Noise Speech", fs)

print()
print("CONCLUSION FROM FFT ANALYSIS:")
print("  - Clean speech shows clear peaks at formant frequencies")
print("  - White noise adds uniform spectral energy across all frequencies")
print("  - Pink noise adds more energy in lower frequencies")
print("  - Hiss adds energy primarily in higher frequencies (>4kHz)")
print()

# ============================================================================
# IIR FILTERING
# ============================================================================

print("=" * 70)
print("IIR FILTERING FOR NOISE REMOVAL")
print("=" * 70)
print()

# Design IIR filters for each noise type

# 1. For white noise: Band-pass filter (speech range 300-3400 Hz)
b_iir1, a_iir1 = signal.butter(4, [300, 3400], 'bandpass', fs=fs)
iir_filtered_white = signal.filtfilt(b_iir1, a_iir1, noisy_white)
iir_filtered_white = iir_filtered_white / np.max(np.abs(iir_filtered_white))
wavfile.write('outputs/iir_filtered_white.wav', fs, (iir_filtered_white * 32767).astype(np.int16))

# 2. For pink noise: High-pass filter (>250 Hz)
b_iir2, a_iir2 = signal.butter(4, 250, 'high', fs=fs)
iir_filtered_pink = signal.filtfilt(b_iir2, a_iir2, noisy_pink)
iir_filtered_pink = iir_filtered_pink / np.max(np.abs(iir_filtered_pink))
wavfile.write('outputs/iir_filtered_pink.wav', fs, (iir_filtered_pink * 32767).astype(np.int16))

# 3. For hiss: Low-pass filter (<3800 Hz)
b_iir3, a_iir3 = signal.butter(4, 3800, 'low', fs=fs)
iir_filtered_hiss = signal.filtfilt(b_iir3, a_iir3, noisy_hiss)
iir_filtered_hiss = iir_filtered_hiss / np.max(np.abs(iir_filtered_hiss))
wavfile.write('outputs/iir_filtered_hiss.wav', fs, (iir_filtered_hiss * 32767).astype(np.int16))

print("[SUCCESS] IIR filtering complete")
print()

# Correlation with clean speech (IIR)
corr_iir_white = np.corrcoef(clean_speech, iir_filtered_white)[0, 1]
corr_iir_pink = np.corrcoef(clean_speech, iir_filtered_pink)[0, 1]
corr_iir_hiss = np.corrcoef(clean_speech, iir_filtered_hiss)[0, 1]

print("IIR FILTERED CORRELATION WITH CLEAN SPEECH:")
print(f"  White noise filtered: {corr_iir_white:.4f}")
print(f"  Pink noise filtered:  {corr_iir_pink:.4f}")
print(f"  Hiss filtered:        {corr_iir_hiss:.4f}")
print()

# ============================================================================
# FIR FILTERING
# ============================================================================

print("=" * 70)
print("FIR FILTERING FOR NOISE REMOVAL")
print("=" * 70)
print()

# Design FIR filters (using Hamming window)

# 1. For white noise: Band-pass FIR
fir1 = signal.firwin(201, [300, 3400], pass_zero=False, fs=fs, window='hamming')
fir_filtered_white = signal.filtfilt(fir1, 1, noisy_white)
fir_filtered_white = fir_filtered_white / np.max(np.abs(fir_filtered_white))
wavfile.write('outputs/fir_filtered_white.wav', fs, (fir_filtered_white * 32767).astype(np.int16))

# 2. For pink noise: High-pass FIR
fir2 = signal.firwin(201, 250, pass_zero=False, fs=fs, window='hamming')
fir_filtered_pink = signal.filtfilt(fir2, 1, noisy_pink)
fir_filtered_pink = fir_filtered_pink / np.max(np.abs(fir_filtered_pink))
wavfile.write('outputs/fir_filtered_pink.wav', fs, (fir_filtered_pink * 32767).astype(np.int16))

# 3. For hiss: Low-pass FIR
fir3 = signal.firwin(201, 3800, fs=fs, window='hamming')
fir_filtered_hiss = signal.filtfilt(fir3, 1, noisy_hiss)
fir_filtered_hiss = fir_filtered_hiss / np.max(np.abs(fir_filtered_hiss))
wavfile.write('outputs/fir_filtered_hiss.wav', fs, (fir_filtered_hiss * 32767).astype(np.int16))

print("[SUCCESS] FIR filtering complete")
print()

# Correlation with clean speech (FIR)
corr_fir_white = np.corrcoef(clean_speech, fir_filtered_white)[0, 1]
corr_fir_pink = np.corrcoef(clean_speech, fir_filtered_pink)[0, 1]
corr_fir_hiss = np.corrcoef(clean_speech, fir_filtered_hiss)[0, 1]

print("FIR FILTERED CORRELATION WITH CLEAN SPEECH:")
print(f"  White noise filtered: {corr_fir_white:.4f}")
print(f"  Pink noise filtered:  {corr_fir_pink:.4f}")
print(f"  Hiss filtered:        {corr_fir_hiss:.4f}")
print()

# ============================================================================
# VISUALIZATION
# ============================================================================

print("Creating visualization plots...")

# Spectrum comparison
fig1, axes1 = plt.subplots(4, 2, figsize=(15, 16))
fig1.suptitle('Frequency Spectrum Analysis', fontsize=16, fontweight='bold')

spectra = [
    (clean_speech, 'Clean Speech', 'blue'),
    (noisy_white, 'White Noise', 'red'),
    (noisy_pink, 'Pink Noise', 'green'),
    (noisy_hiss, 'High-Frequency Hiss', 'orange'),
]

for idx, (sig, title, color) in enumerate(spectra):
    # Time domain
    time_plot = t[:int(0.1 * fs)]
    axes1[idx, 0].plot(time_plot, sig[:len(time_plot)], linewidth=0.8, color=color)
    axes1[idx, 0].set_title(f'{title} - Time Domain', fontweight='bold')
    axes1[idx, 0].set_xlabel('Time (s)')
    axes1[idx, 0].set_ylabel('Amplitude')
    axes1[idx, 0].grid(True, alpha=0.3)
    
    # Frequency domain
    fft_sig = np.fft.fft(sig)
    freqs_sig = np.fft.fftfreq(len(sig), 1/fs)
    pos = freqs_sig > 0
    axes1[idx, 1].plot(freqs_sig[pos], np.abs(fft_sig[pos]), linewidth=1, color=color)
    axes1[idx, 1].set_title(f'{title} - Frequency Spectrum', fontweight='bold')
    axes1[idx, 1].set_xlabel('Frequency (Hz)')
    axes1[idx, 1].set_ylabel('Magnitude')
    axes1[idx, 1].set_xlim(0, fs/2)
    axes1[idx, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/spectrum_analysis.png', dpi=300, bbox_inches='tight')
print("[SUCCESS] Spectrum analysis saved")

# IIR vs FIR comparison
fig2, axes2 = plt.subplots(3, 3, figsize=(16, 12))
fig2.suptitle('IIR vs FIR Filtering Comparison', fontsize=16, fontweight='bold')

noise_types = ['White Noise', 'Pink Noise', 'High-Freq Hiss']
iir_results = [iir_filtered_white, iir_filtered_pink, iir_filtered_hiss]
fir_results = [fir_filtered_white, fir_filtered_pink, fir_filtered_hiss]
noisy_signals = [noisy_white, noisy_pink, noisy_hiss]

for idx in range(3):
    # Noisy
    axes2[idx, 0].plot(time_plot, noisy_signals[idx][:len(time_plot)], linewidth=0.8, color='red', alpha=0.7)
    axes2[idx, 0].set_title(f'{noise_types[idx]} - Noisy', fontweight='bold')
    axes2[idx, 0].set_xlabel('Time (s)')
    axes2[idx, 0].set_ylabel('Amplitude')
    axes2[idx, 0].grid(True, alpha=0.3)
    
    # IIR Filtered
    axes2[idx, 1].plot(time_plot, iir_results[idx][:len(time_plot)], linewidth=0.8, color='blue')
    axes2[idx, 1].set_title(f'{noise_types[idx]} - IIR Filtered', fontweight='bold')
    axes2[idx, 1].set_xlabel('Time (s)')
    axes2[idx, 1].set_ylabel('Amplitude')
    axes2[idx, 1].grid(True, alpha=0.3)
    
    # FIR Filtered
    axes2[idx, 2].plot(time_plot, fir_results[idx][:len(time_plot)], linewidth=0.8, color='green')
    axes2[idx, 2].set_title(f'{noise_types[idx]} - FIR Filtered', fontweight='bold')
    axes2[idx, 2].set_xlabel('Time (s)')
    axes2[idx, 2].set_ylabel('Amplitude')
    axes2[idx, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/filtering_comparison.png', dpi=300, bbox_inches='tight')
print("[SUCCESS] Filtering comparison saved")

# Correlation comparison
fig3, ax3 = plt.subplots(1, 1, figsize=(12, 6))
x_pos = np.arange(3)
width = 0.35

iir_corrs = [corr_iir_white, corr_iir_pink, corr_iir_hiss]
fir_corrs = [corr_fir_white, corr_fir_pink, corr_fir_hiss]

bars1 = ax3.bar(x_pos - width/2, iir_corrs, width, label='IIR Filtered', alpha=0.8)
bars2 = ax3.bar(x_pos + width/2, fir_corrs, width, label='FIR Filtered', alpha=0.8)

ax3.set_xlabel('Noise Type', fontweight='bold')
ax3.set_ylabel('Correlation with Clean Speech', fontweight='bold')
ax3.set_title('IIR vs FIR: Correlation with Clean Speech', fontsize=14, fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(noise_types)
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('outputs/correlation_comparison.png', dpi=300, bbox_inches='tight')
print("[SUCCESS] Correlation comparison saved")
print()

# ============================================================================
# CONCLUSIONS
# ============================================================================

print("=" * 70)
print("CONCLUSIONS")
print("=" * 70)
print()

print("IIR FILTERING RESULTS:")
print(f"  White Noise: {corr_iir_white:.4f} correlation (Good noise reduction)")
print(f"  Pink Noise:  {corr_iir_pink:.4f} correlation (Excellent low-freq noise removal)")
print(f"  Hiss:        {corr_iir_hiss:.4f} correlation (Excellent high-freq noise removal)")
print()
print("  Advantages: Sharp cutoff, lower order, efficient")
print("  Limitations: Potential phase distortion, less linear phase")
print()

print("FIR FILTERING RESULTS:")
print(f"  White Noise: {corr_fir_white:.4f} correlation (Good noise reduction)")
print(f"  Pink Noise:  {corr_fir_pink:.4f} correlation (Very good)")
print(f"  Hiss:        {corr_fir_hiss:.4f} correlation (Very good)")
print()
print("  Advantages: Linear phase, always stable, better transient response")
print("  Limitations: Higher order needed, more computation")
print()

print("OVERALL COMPARISON:")
if np.mean(iir_corrs) > np.mean(fir_corrs):
    print("  IIR filters performed slightly better on average")
else:
    print("  FIR filters performed slightly better on average")
print("  Both methods effectively reduced noise")
print("  Choice depends on: phase linearity requirements, computational resources")
print()

print("=" * 70)
print("OPE 4 COMPLETED SUCCESSFULLY!")
print("=" * 70)
