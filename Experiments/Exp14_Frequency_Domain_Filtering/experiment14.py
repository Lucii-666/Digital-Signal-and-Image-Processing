"""
Experiment 14: Frequency Domain Filtering
Objective: To perform smoothing and sharpening using frequency domain filters
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

os.makedirs('outputs', exist_ok=True)

print("=" * 50)
print("EXPERIMENT 14: FREQUENCY DOMAIN FILTERING")
print("=" * 50)
print()

# Load image
img_path = '../../extracted_images/image_1.png'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    img = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
else:
    img = cv2.resize(img, (512, 512))

print(f"Image shape: {img.shape}")
print()

# Convert to frequency domain
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# Calculate magnitude spectrum
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]) + 1)

print("Frequency Domain:")
print(f"  Magnitude range: [{magnitude_spectrum.min():.2f}, {magnitude_spectrum.max():.2f}]")
print()

rows, cols = img.shape
crow, ccol = rows // 2, cols // 2

# Create filter functions
def create_ideal_lpf(shape, cutoff):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), cutoff, 1, -1)
    return mask

def create_butterworth_lpf(shape, cutoff, order=2):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    u = np.arange(rows).reshape(-1, 1) - crow
    v = np.arange(cols) - ccol
    D = np.sqrt(u**2 + v**2)
    mask = 1 / (1 + (D / cutoff)**(2 * order))
    return mask

def create_gaussian_lpf(shape, cutoff):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    u = np.arange(rows).reshape(-1, 1) - crow
    v = np.arange(cols) - ccol
    D = np.sqrt(u**2 + v**2)
    mask = np.exp(-(D**2) / (2 * cutoff**2))
    return mask

# Cutoff frequencies
cutoff_low = 30
cutoff_high = 20

print("Filters:")
print(f"  Lowpass cutoff: {cutoff_low}")
print(f"  Highpass cutoff: {cutoff_high}")
print()

# Lowpass filters
ideal_lpf = create_ideal_lpf(img.shape, cutoff_low)
butter_lpf = create_butterworth_lpf(img.shape, cutoff_low)
gauss_lpf = create_gaussian_lpf(img.shape, cutoff_low)

# Highpass filters (1 - lowpass)
ideal_hpf = 1 - create_ideal_lpf(img.shape, cutoff_high)
butter_hpf = 1 - create_butterworth_lpf(img.shape, cutoff_high)
gauss_hpf = 1 - create_gaussian_lpf(img.shape, cutoff_high)

# Apply filters
def apply_filter(dft_shift, filter_mask):
    fshift = dft_shift * filter_mask[:, :, np.newaxis]
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
    return np.clip(img_back, 0, 255).astype(np.uint8)

# Apply all filters
ideal_low = apply_filter(dft_shift, ideal_lpf)
butter_low = apply_filter(dft_shift, butter_lpf)
gauss_low = apply_filter(dft_shift, gauss_lpf)

ideal_high = apply_filter(dft_shift, ideal_hpf)
butter_high = apply_filter(dft_shift, butter_hpf)
gauss_high = apply_filter(dft_shift, gauss_hpf)

print("Filters applied successfully")
print()

# Visualization - Frequency Domain
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Frequency Domain Analysis', fontsize=16, fontweight='bold')

axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('Original Image', fontweight='bold')
axes[0, 0].axis('off')

axes[0, 1].imshow(magnitude_spectrum, cmap='gray')
axes[0, 1].set_title('Magnitude Spectrum', fontweight='bold')
axes[0, 1].axis('off')

axes[1, 0].imshow(ideal_lpf, cmap='gray')
axes[1, 0].set_title('Ideal Lowpass Filter', fontweight='bold')
axes[1, 0].axis('off')

axes[1, 1].imshow(gauss_lpf, cmap='gray')
axes[1, 1].set_title('Gaussian Lowpass Filter', fontweight='bold')
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('outputs/frequency_domain.png', dpi=300, bbox_inches='tight')
print("[SUCCESS] Frequency domain analysis saved")
print()

# Lowpass filtering results
fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
fig2.suptitle('Lowpass Filtering (Smoothing)', fontsize=16, fontweight='bold')

lowpass_results = [
    (img, 'Original'),
    (ideal_low, 'Ideal LPF'),
    (butter_low, 'Butterworth LPF'),
    (gauss_low, 'Gaussian LPF'),
]

for idx, (image, title) in enumerate(lowpass_results):
    row, col = idx // 2, idx % 2
    axes2[row, col].imshow(image, cmap='gray')
    axes2[row, col].set_title(title, fontweight='bold')
    axes2[row, col].axis('off')

plt.tight_layout()
plt.savefig('outputs/lowpass_filtering.png', dpi=300, bbox_inches='tight')
print("[SUCCESS] Lowpass filtering saved")
print()

# Highpass filtering results
fig3, axes3 = plt.subplots(2, 2, figsize=(12, 10))
fig3.suptitle('Highpass Filtering (Sharpening)', fontsize=16, fontweight='bold')

highpass_results = [
    (img, 'Original'),
    (ideal_high, 'Ideal HPF'),
    (butter_high, 'Butterworth HPF'),
    (gauss_high, 'Gaussian HPF'),
]

for idx, (image, title) in enumerate(highpass_results):
    row, col = idx // 2, idx % 2
    axes3[row, col].imshow(image, cmap='gray')
    axes3[row, col].set_title(title, fontweight='bold')
    axes3[row, col].axis('off')

plt.tight_layout()
plt.savefig('outputs/highpass_filtering.png', dpi=300, bbox_inches='tight')
print("[SUCCESS] Highpass filtering saved")
print()
print("=" * 50)
print("EXPERIMENT 14 COMPLETED SUCCESSFULLY!")
print("=" * 50)
