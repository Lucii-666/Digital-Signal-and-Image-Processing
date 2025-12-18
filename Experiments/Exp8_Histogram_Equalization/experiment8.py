"""
Experiment 8: Histogram and Histogram Equalization
Objective: To compute and equalize image histograms
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

os.makedirs('outputs', exist_ok=True)

print("=" * 50)
print("EXPERIMENT 8: HISTOGRAM EQUALIZATION")
print("=" * 50)
print()

# Load image
img_path = '../../extracted_images/image_1.png'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    img = np.random.randint(0, 256, (512, 512), dtype=np.uint8)

print(f"Image shape: {img.shape}")
print(f"Intensity range: [{np.min(img)}, {np.max(img)}]")
print()

# Calculate histogram
hist, bins = np.histogram(img.flatten(), 256, [0, 256])

print("Histogram:")
print(f"  Total pixels: {np.sum(hist)}")
print(f"  Most frequent intensity: {np.argmax(hist)}")
print()

# Manual histogram equalization
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max() / cdf.max()
cdf_m = np.ma.masked_equal(cdf, 0)
cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
cdf_final = np.ma.filled(cdf_m, 0).astype('uint8')
img_eq_manual = cdf_final[img]

print("Manual Equalization:")
print(f"  Equalized range: [{np.min(img_eq_manual)}, {np.max(img_eq_manual)}]")
print()

# OpenCV histogram equalization
img_eq_cv = cv2.equalizeHist(img)
print("OpenCV Equalization:")
print(f"  Equalized range: [{np.min(img_eq_cv)}, {np.max(img_eq_cv)}]")
print()

# CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
img_clahe = clahe.apply(img)
print("CLAHE:")
print(f"  Range: [{np.min(img_clahe)}, {np.max(img_clahe)}]")
print()

# Visualization
fig, axes = plt.subplots(3, 2, figsize=(12, 12))
fig.suptitle('Histogram Equalization', fontsize=16, fontweight='bold')

# Original
axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('Original Image', fontweight='bold')
axes[0, 0].axis('off')

axes[0, 1].hist(img.flatten(), 256, [0, 256], color='blue', alpha=0.7)
axes[0, 1].set_title('Original Histogram', fontweight='bold')
axes[0, 1].set_xlabel('Intensity')
axes[0, 1].set_ylabel('Frequency')

# Manual equalization
axes[1, 0].imshow(img_eq_manual, cmap='gray')
axes[1, 0].set_title('Manual Histogram Equalization', fontweight='bold')
axes[1, 0].axis('off')

axes[1, 1].hist(img_eq_manual.flatten(), 256, [0, 256], color='green', alpha=0.7)
axes[1, 1].set_title('Equalized Histogram (Manual)', fontweight='bold')
axes[1, 1].set_xlabel('Intensity')
axes[1, 1].set_ylabel('Frequency')

# CLAHE
axes[2, 0].imshow(img_clahe, cmap='gray')
axes[2, 0].set_title('CLAHE Result', fontweight='bold')
axes[2, 0].axis('off')

axes[2, 1].hist(img_clahe.flatten(), 256, [0, 256], color='red', alpha=0.7)
axes[2, 1].set_title('CLAHE Histogram', fontweight='bold')
axes[2, 1].set_xlabel('Intensity')
axes[2, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('outputs/histogram_equalization.png', dpi=300, bbox_inches='tight')
print("[SUCCESS] Histogram equalization saved")
print()

# CDF comparison
fig2, ax = plt.subplots(figsize=(10, 6))
ax.plot(cdf_normalized, color='b', label='Original CDF')
hist_eq, _ = np.histogram(img_eq_manual.flatten(), 256, [0, 256])
cdf_eq = hist_eq.cumsum()
cdf_eq_normalized = cdf_eq * hist_eq.max() / cdf_eq.max()
ax.plot(cdf_eq_normalized, color='r', label='Equalized CDF')
ax.set_title('CDF Comparison', fontweight='bold', fontsize=14)
ax.set_xlabel('Intensity')
ax.set_ylabel('Cumulative Count')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('outputs/cdf_comparison.png', dpi=300, bbox_inches='tight')
print("[SUCCESS] CDF comparison saved")
print()
print("=" * 50)
print("EXPERIMENT 8 COMPLETED SUCCESSFULLY!")
print("=" * 50)
