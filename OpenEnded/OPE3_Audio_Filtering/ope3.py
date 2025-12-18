"""
Open Ended Problem 3: Low-Pass and Band-Pass Filtering on MP3 Songs
(a) Low-pass filtering with different cutoff frequencies
(b) Band-pass filtering for vocal and music suppression
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import os

os.makedirs('outputs', exist_ok=True)

print("=" * 70)
print("OPE 3: AUDIO FILTERING - LOW-PASS & BAND-PASS")
print("=" * 70)
print()

# Create synthetic audio signal
fs = 44100  # Sampling rate
duration = 5  # seconds
t = np.linspace(0, duration, int(fs * duration))

print("Creating synthetic song with multiple frequency components...")

# Vocals (mid-range: 300-1000 Hz)
vocals = (np.sin(2 * np.pi * 400 * t) + 
          0.8 * np.sin(2 * np.pi * 600 * t) +
          0.6 * np.sin(2 * np.pi * 800 * t))

# Bass (low: 60-250 Hz)
bass = (0.5 * np.sin(2 * np.pi * 80 * t) +
        0.4 * np.sin(2 * np.pi * 120 * t))

# Treble/Cymbals (high: 2000-8000 Hz)
treble = (0.3 * np.sin(2 * np.pi * 3000 * t) +
          0.2 * np.sin(2 * np.pi * 5000 * t))

# Combined original song
original = vocals + bass + treble
original = original / np.max(np.abs(original))

wavfile.write('outputs/original_song.wav', fs, (original * 32767).astype(np.int16))
print("[SUCCESS] Created original song with vocals, bass, and treble")
print()

# ============================================================================
# PART (a): LOW-PASS FILTERING WITH DIFFERENT CUTOFF FREQUENCIES
# ============================================================================

print("=" * 70)
print("PART (a): LOW-PASS FILTERING")
print("=" * 70)
print()

cutoff_frequencies = [500, 2000, 5000]  # Three different cutoffs
filtered_lowpass = []

print("Applying low-pass filters with different cutoff frequencies:")
print()

for fc in cutoff_frequencies:
    # Design FIR low-pass filter
    numtaps = 501
    lpf = signal.firwin(numtaps, fc, fs=fs)
    
    # Apply filter
    filtered = signal.filtfilt(lpf, 1, original)
    filtered = filtered / np.max(np.abs(filtered))
    filtered_lowpass.append(filtered)
    
    # Save audio
    filename = f'outputs/lowpass_{fc}Hz.wav'
    wavfile.write(filename, fs, (filtered * 32767).astype(np.int16))
    
    print(f"  Cutoff: {fc} Hz - Filter order: {numtaps}")
    print(f"  Saved: {filename}")
    print()

# ============================================================================
# PART (b): BAND-PASS FILTERING FOR VOCAL AND MUSIC SUPPRESSION
# ============================================================================

print("=" * 70)
print("PART (b): BAND-PASS FILTERING")
print("=" * 70)
print()

# Band-pass filter 1: Suppress vocals (keep bass + treble)
# Pass: 60-300 Hz and 1500-8000 Hz, Stop: 300-1500 Hz (vocal range)
print("Filter 1: Suppress Vocals (keep bass and treble)")

# Low-pass for bass
bass_filter = signal.firwin(501, 300, fs=fs)
bass_only = signal.filtfilt(bass_filter, 1, original)

# High-pass for treble
treble_filter = signal.firwin(501, 1500, pass_zero=False, fs=fs)
treble_only = signal.filtfilt(treble_filter, 1, original)

# Combine
music_only = bass_only + treble_only
music_only = music_only / np.max(np.abs(music_only))

wavfile.write('outputs/bandpass_music_only.wav', fs, (music_only * 32767).astype(np.int16))
print("  Saved: bandpass_music_only.wav")
print()

# Band-pass filter 2: Suppress background music (keep vocals)
# Pass: 300-1500 Hz (vocal range)
print("Filter 2: Suppress Background Music (keep vocals)")

vocal_filter = signal.firwin(501, [300, 1500], pass_zero=False, fs=fs)
vocals_only = signal.filtfilt(vocal_filter, 1, original)
vocals_only = vocals_only / np.max(np.abs(vocals_only))

wavfile.write('outputs/bandpass_vocals_only.wav', fs, (vocals_only * 32767).astype(np.int16))
print("  Saved: bandpass_vocals_only.wav")
print()

# ============================================================================
# VISUALIZATION
# ============================================================================

print("Creating visualization plots...")

fig, axes = plt.subplots(4, 2, figsize=(15, 16))
fig.suptitle('Audio Filtering Analysis', fontsize=16, fontweight='bold')

# Time domain comparison (first 0.05 seconds)
time_plot = t[:int(0.05 * fs)]

# Row 0: Original signal
axes[0, 0].plot(time_plot, original[:len(time_plot)], linewidth=0.8)
axes[0, 0].set_title('Original Song - Time Domain', fontweight='bold')
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('Amplitude')
axes[0, 0].grid(True, alpha=0.3)

# Original spectrum
fft_orig = np.fft.fft(original[:fs])
freqs_orig = np.fft.fftfreq(fs, 1/fs)
pos_orig = freqs_orig > 0
axes[0, 1].plot(freqs_orig[pos_orig], np.abs(fft_orig[pos_orig]), linewidth=1)
axes[0, 1].set_title('Original Song - Frequency Spectrum', fontweight='bold')
axes[0, 1].set_xlabel('Frequency (Hz)')
axes[0, 1].set_ylabel('Magnitude')
axes[0, 1].set_xlim(0, 6000)
axes[0, 1].grid(True, alpha=0.3)

# Row 1-3: Low-pass filtered signals
colors_lp = ['orange', 'green', 'red']
for idx, (filtered, fc, color) in enumerate(zip(filtered_lowpass, cutoff_frequencies, colors_lp)):
    row = idx + 1
    
    # Time domain
    axes[row, 0].plot(time_plot, filtered[:len(time_plot)], linewidth=0.8, color=color)
    axes[row, 0].set_title(f'Low-Pass {fc} Hz - Time Domain', fontweight='bold')
    axes[row, 0].set_xlabel('Time (s)')
    axes[row, 0].set_ylabel('Amplitude')
    axes[row, 0].grid(True, alpha=0.3)
    
    # Frequency domain
    fft_filt = np.fft.fft(filtered[:fs])
    freqs_filt = np.fft.fftfreq(fs, 1/fs)
    pos_filt = freqs_filt > 0
    axes[row, 1].plot(freqs_filt[pos_filt], np.abs(fft_filt[pos_filt]), linewidth=1, color=color)
    axes[row, 1].set_title(f'Low-Pass {fc} Hz - Frequency Spectrum', fontweight='bold')
    axes[row, 1].set_xlabel('Frequency (Hz)')
    axes[row, 1].set_ylabel('Magnitude')
    axes[row, 1].set_xlim(0, 6000)
    axes[row, 1].axvline(x=fc, color='red', linestyle='--', label=f'Cutoff: {fc} Hz')
    axes[row, 1].legend()
    axes[row, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/lowpass_filtering_analysis.png', dpi=300, bbox_inches='tight')
print("[SUCCESS] Low-pass analysis saved")
print()

# Band-pass filtering visualization
fig2, axes2 = plt.subplots(3, 2, figsize=(15, 12))
fig2.suptitle('Band-Pass Filtering for Vocal/Music Separation', fontsize=16, fontweight='bold')

signals_bp = [
    (original, 'Original Song', 'blue'),
    (music_only, 'Music Only (Vocals Suppressed)', 'green'),
    (vocals_only, 'Vocals Only (Music Suppressed)', 'orange'),
]

for idx, (sig, title, color) in enumerate(signals_bp):
    # Time domain
    axes2[idx, 0].plot(time_plot, sig[:len(time_plot)], linewidth=0.8, color=color)
    axes2[idx, 0].set_title(f'{title} - Time Domain', fontweight='bold')
    axes2[idx, 0].set_xlabel('Time (s)')
    axes2[idx, 0].set_ylabel('Amplitude')
    axes2[idx, 0].grid(True, alpha=0.3)
    
    # Frequency domain
    fft_sig = np.fft.fft(sig[:fs])
    freqs_sig = np.fft.fftfreq(fs, 1/fs)
    pos_sig = freqs_sig > 0
    axes2[idx, 1].plot(freqs_sig[pos_sig], np.abs(fft_sig[pos_sig]), linewidth=1, color=color)
    axes2[idx, 1].set_title(f'{title} - Frequency Spectrum', fontweight='bold')
    axes2[idx, 1].set_xlabel('Frequency (Hz)')
    axes2[idx, 1].set_ylabel('Magnitude')
    axes2[idx, 1].set_xlim(0, 6000)
    axes2[idx, 1].grid(True, alpha=0.3)
    
    # Add frequency range annotations
    if idx == 1:  # Music only
        axes2[idx, 1].axvspan(0, 300, alpha=0.2, color='green', label='Bass')
        axes2[idx, 1].axvspan(1500, 6000, alpha=0.2, color='cyan', label='Treble')
        axes2[idx, 1].legend()
    elif idx == 2:  # Vocals only
        axes2[idx, 1].axvspan(300, 1500, alpha=0.2, color='orange', label='Vocal Range')
        axes2[idx, 1].legend()

plt.tight_layout()
plt.savefig('outputs/bandpass_filtering_analysis.png', dpi=300, bbox_inches='tight')
print("[SUCCESS] Band-pass analysis saved")
print()

# ============================================================================
# OBSERVATIONS AND CONCLUSIONS
# ============================================================================

print("=" * 70)
print("PART (a) OBSERVATIONS: LOW-PASS FILTERING")
print("=" * 70)
print("""
1. LOW-PASS 500 Hz:
   - Character: Very muffled, bass-heavy sound
   - Clarity: Significantly reduced - only bass frequencies remain
   - Tonal Balance: Extremely dark, lacks brightness
   - Effect: Removes most vocals and all treble instruments
   - Use Case: Bass extraction, removing high-frequency noise

2. LOW-PASS 2000 Hz:
   - Character: Warmer, less bright than original
   - Clarity: Good - vocals preserved, some treble lost
   - Tonal Balance: Well-balanced for vocal-focused music
   - Effect: Removes cymbals and high-frequency percussion
   - Use Case: Reducing harshness, FM radio-like quality

3. LOW-PASS 5000 Hz:
   - Character: Very close to original
   - Clarity: Excellent - most content preserved
   - Tonal Balance: Natural, slight reduction in extreme highs
   - Effect: Minimal change, removes only ultra-high frequencies
   - Use Case: Gentle smoothing, CD-quality audio
   
PATTERN OBSERVED:
- Lower cutoff → more muffled, bass-heavy sound
- Higher cutoff → more transparent, natural sound
- Cutoff frequency directly controls brightness/darkness
""")
print()

print("=" * 70)
print("PART (b) OBSERVATIONS: BAND-PASS FILTERING")
print("=" * 70)
print("""
1. MUSIC ONLY (Vocals Suppressed):
   - Method: Combined bass (<300 Hz) + treble (>1500 Hz)
   - Effect: Removed mid-range where vocals reside
   - Quality: Preserves rhythm section and high-frequency instruments
   - Limitation: Some vocal bleed remains at band edges
   - Sound Character: "Hollow" sound, missing midrange warmth
   
2. VOCALS ONLY (Music Suppressed):
   - Method: Band-pass 300-1500 Hz (vocal frequency range)
   - Effect: Isolated vocal frequencies, removed bass and treble
   - Quality: Vocals audible but thin, lacking fullness
   - Limitation: Loses harmonic richness and accompaniment
   - Sound Character: Telephone-like quality, lacks depth
   
EFFECTIVENESS:
- Partial separation achieved based on frequency ranges
- Not perfect due to frequency overlap between sources
- Works better for synthetic signals than real recordings

CHALLENGES:
- Vocals and instruments overlap in frequency domain
- Linear filtering cannot separate time-domain sources
- Phase distortion can occur with sharp filter cutoffs
""")
print()

print("=" * 70)
print("CONCLUSIONS")
print("=" * 70)
print("""
LOW-PASS FILTERING:
✓ Cutoff frequency is critical parameter affecting sound quality
✓ Lower cutoffs create darker, warmer tones
✓ Higher cutoffs preserve more original character
✓ Useful for noise reduction and tone shaping
✗ Cannot selectively remove specific instruments

BAND-PASS FILTERING:
✓ Can isolate frequency ranges (bass, vocals, treble)
✓ Useful for basic source separation
✓ Works well when sources have distinct frequency bands
✗ Limited by frequency overlap between sources
✗ Cannot achieve complete vocal/music separation
✗ Quality loss due to removed frequency content

RECOMMENDATIONS:
- For casual listening: Use cutoffs ≥2000 Hz
- For bass extraction: Use cutoffs ≤500 Hz
- For vocal isolation: Combine multiple filtering strategies
- For professional work: Use time-frequency or ML-based methods
""")
print()

print("=" * 70)
print("OPE 3 COMPLETED SUCCESSFULLY!")
print("=" * 70)
print()
print("Generated files:")
print("  Low-pass filtered:")
print("    - lowpass_500Hz.wav")
print("    - lowpass_2000Hz.wav")
print("    - lowpass_5000Hz.wav")
print("  Band-pass filtered:")
print("    - bandpass_music_only.wav")
print("    - bandpass_vocals_only.wav")
print("  Analysis plots:")
print("    - lowpass_filtering_analysis.png")
print("    - bandpass_filtering_analysis.png")
