"""
Experiment 9: Smoothing and Sharpening in Spatial Domain
Objective: To apply spatial domain filters for smoothing and sharpening
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

os.makedirs('outputs', exist_ok=True)

print("=" * 50)
print("EXPERIMENT 9: SPATIAL DOMAIN FILTERING")
print("=" * 50)
print()

# Load image
img_path = '../../extracted_images/image_1.png'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    img = np.random.randint(0, 256, (512, 512), dtype=np.uint8)

print(f"Image shape: {img.shape}")
print()

# SMOOTHING FILTERS
print("=" * 50)
print("SMOOTHING FILTERS")
print("=" * 50)

# Mean filter
kernel_size = 5
mean_kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
mean_filtered = cv2.filter2D(img, -1, mean_kernel)
print(f"Mean filter kernel size: {kernel_size}x{kernel_size}")

# Gaussian filter
gaussian_filtered = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
print(f"Gaussian filter applied")

# Median filter
median_filtered = cv2.medianBlur(img, kernel_size)
print(f"Median filter applied")

# Bilateral filter
bilateral_filtered = cv2.bilateralFilter(img, 9, 75, 75)
print(f"Bilateral filter applied (edge-preserving)")
print()

# SHARPENING FILTERS
print("=" * 50)
print("SHARPENING FILTERS")
print("=" * 50)

# Laplacian
laplacian_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
laplacian = cv2.filter2D(img, -1, laplacian_kernel)
sharpened_laplacian = cv2.add(img, laplacian.astype(np.uint8))
print("Laplacian sharpening applied")

# Unsharp masking
gaussian_blur = cv2.GaussianBlur(img, (9, 9), 10.0)
unsharp_mask = cv2.addWeighted(img, 1.5, gaussian_blur, -0.5, 0)
print("Unsharp masking applied")

# High-boost filtering
A = 2.0
high_boost = cv2.addWeighted(img, A, gaussian_blur, -(A-1), 0)
print(f"High-boost filter with A={A}")

# Sobel sharpening
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobel = np.sqrt(sobelx**2 + sobely**2)
sobel = np.clip(sobel, 0, 255).astype(np.uint8)
sobel_sharpened = cv2.add(img, sobel)
print("Sobel-based sharpening applied")
print()

# Visualization - Smoothing
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Smoothing Filters', fontsize=16, fontweight='bold')

smoothing_images = [
    (img, 'Original'),
    (mean_filtered, 'Mean Filter'),
    (gaussian_filtered, 'Gaussian Filter'),
    (median_filtered, 'Median Filter'),
    (bilateral_filtered, 'Bilateral Filter'),
]

for idx, (image, title) in enumerate(smoothing_images):
    row, col = idx // 3, idx % 3
    axes[row, col].imshow(image, cmap='gray')
    axes[row, col].set_title(title, fontweight='bold')
    axes[row, col].axis('off')

axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('outputs/smoothing_filters.png', dpi=300, bbox_inches='tight')
print("[SUCCESS] Smoothing filters saved")
print()

# Visualization - Sharpening
fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))
fig2.suptitle('Sharpening Filters', fontsize=16, fontweight='bold')

sharpening_images = [
    (img, 'Original'),
    (sharpened_laplacian, 'Laplacian Sharpening'),
    (unsharp_mask, 'Unsharp Masking'),
    (high_boost, 'High-Boost Filter'),
    (sobel_sharpened, 'Sobel Sharpening'),
]

for idx, (image, title) in enumerate(sharpening_images):
    row, col = idx // 3, idx % 3
    axes2[row, col].imshow(image, cmap='gray')
    axes2[row, col].set_title(title, fontweight='bold')
    axes2[row, col].axis('off')

axes2[1, 2].axis('off')

plt.tight_layout()
plt.savefig('outputs/sharpening_filters.png', dpi=300, bbox_inches='tight')
print("[SUCCESS] Sharpening filters saved")
print()
print("=" * 50)
print("EXPERIMENT 9 COMPLETED SUCCESSFULLY!")
print("=" * 50)
