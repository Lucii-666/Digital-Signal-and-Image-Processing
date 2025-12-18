"""
Experiment 10: Non-Linear Filters on Images
Objective: To apply various non-linear filters for noise reduction
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
import os

os.makedirs('outputs', exist_ok=True)

print("=" * 50)
print("EXPERIMENT 10: NON-LINEAR FILTERS")
print("=" * 50)
print()

# Load and add noise
img_path = '../../extracted_images/image_1.png'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    img = np.random.randint(0, 256, (512, 512), dtype=np.uint8)

print(f"Image shape: {img.shape}")

# Add salt and pepper noise
noisy = np.copy(img)
prob = 0.05
rnd = np.random.random(img.shape)
noisy[rnd < prob/2] = 0
noisy[rnd > 1 - prob/2] = 255
print(f"Salt & pepper noise added (prob={prob})")
print()

# 1. Median filter
median = cv2.medianBlur(noisy, 5)
print("Median filter: Best for salt & pepper noise")

# 2. Min filter
min_filter = ndimage.minimum_filter(noisy, size=5)
print("Min filter: Reduces bright noise")

# 3. Max filter
max_filter = ndimage.maximum_filter(noisy, size=5)
print("Max filter: Reduces dark noise")

# 4. Midpoint filter
midpoint = ((min_filter.astype(np.float32) + max_filter.astype(np.float32)) / 2).astype(np.uint8)
print("Midpoint filter applied")

# 5. Alpha-trimmed mean
def alpha_trimmed_mean(img, size=5, d=4):
    padded = np.pad(img, size//2, mode='edge')
    result = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            window = padded[i:i+size, j:j+size].flatten()
            sorted_window = np.sort(window)
            trimmed = sorted_window[d//2:-d//2] if d > 0 else sorted_window
            result[i, j] = np.mean(trimmed)
    return result.astype(np.uint8)

alpha_trimmed = alpha_trimmed_mean(noisy, 5, 4)
print("Alpha-trimmed mean filter applied")
print()

# Visualization
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
fig.suptitle('Non-Linear Filters for Noise Reduction', fontsize=16, fontweight='bold')

filters = [
    (img, 'Original'),
    (noisy, 'Noisy (Salt & Pepper)'),
    (median, 'Median Filter'),
    (min_filter, 'Min Filter'),
    (max_filter, 'Max Filter'),
    (midpoint, 'Midpoint Filter'),
    (alpha_trimmed, 'Alpha-Trimmed Mean'),
]

for idx, (image, title) in enumerate(filters):
    row, col = idx // 3, idx % 3
    axes[row, col].imshow(image, cmap='gray')
    axes[row, col].set_title(title, fontweight='bold')
    axes[row, col].axis('off')

# Remove empty subplots
for idx in range(len(filters), 9):
    row, col = idx // 3, idx % 3
    axes[row, col].axis('off')

plt.tight_layout()
plt.savefig('outputs/nonlinear_filters.png', dpi=300, bbox_inches='tight')
print("[SUCCESS] Non-linear filters saved")
print()

# PSNR comparison
def calculate_psnr(img1, img2):
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

print("PSNR Comparison (Higher is better):")
print(f"Noisy image: {calculate_psnr(img, noisy):.2f} dB")
print(f"Median filter: {calculate_psnr(img, median):.2f} dB")
print(f"Min filter: {calculate_psnr(img, min_filter):.2f} dB")
print(f"Max filter: {calculate_psnr(img, max_filter):.2f} dB")
print(f"Midpoint filter: {calculate_psnr(img, midpoint):.2f} dB")
print(f"Alpha-trimmed: {calculate_psnr(img, alpha_trimmed):.2f} dB")
print()
print("=" * 50)
print("EXPERIMENT 10 COMPLETED SUCCESSFULLY!")
print("=" * 50)
