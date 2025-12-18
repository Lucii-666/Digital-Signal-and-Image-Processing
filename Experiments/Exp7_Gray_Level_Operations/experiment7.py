"""
Experiment 7: Gray Level Operations on Images
Objective: To perform various gray level transformations on images
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

os.makedirs('outputs', exist_ok=True)

print("=" * 50)
print("EXPERIMENT 7: GRAY LEVEL TRANSFORMATIONS")
print("=" * 50)
print()

# Load image
img_path = '../../extracted_images/image_1.png'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    img = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
    print("Using generated image")
else:
    print(f"Loading image: image_1.png")

print(f"Image shape: {img.shape}")
print(f"Data type: {img.dtype}")
print(f"Min value: {np.min(img)}, Max value: {np.max(img)}")
print()

# 1. Negative transformation
print("=" * 50)
print("NEGATIVE TRANSFORMATION")
print("=" * 50)
negative = 255 - img
print(f"Negative - Min: {np.min(negative)}, Max: {np.max(negative)}")
print()

# 2. Log transformation
print("=" * 50)
print("LOG TRANSFORMATION")
print("=" * 50)
c = 255 / np.log(1 + np.max(img))
log_transform = c * np.log(1 + img.astype(np.float32))
log_transform = np.clip(log_transform, 0, 255).astype(np.uint8)
print(f"Log - c constant: {c:.2f}")
print(f"Log - Min: {np.min(log_transform)}, Max: {np.max(log_transform)}")
print()

# 3. Power-law (Gamma) transformation
print("=" * 50)
print("POWER-LAW (GAMMA) TRANSFORMATIONS")
print("=" * 50)
gamma_values = [0.5, 1.5, 2.5]
gamma_images = []

for gamma in gamma_values:
    gamma_corrected = np.power(img / 255.0, gamma) * 255
    gamma_corrected = gamma_corrected.astype(np.uint8)
    gamma_images.append(gamma_corrected)
    print(f"Gamma={gamma}: Min={np.min(gamma_corrected)}, Max={np.max(gamma_corrected)}")
print()

# 4. Contrast stretching
print("=" * 50)
print("CONTRAST STRETCHING")
print("=" * 50)
r_min, r_max = np.min(img), np.max(img)
stretched = ((img - r_min) / (r_max - r_min) * 255).astype(np.uint8)
print(f"Original range: [{r_min}, {r_max}]")
print(f"Stretched range: [{np.min(stretched)}, {np.max(stretched)}]")
print()

# 5. Intensity level slicing
print("=" * 50)
print("INTENSITY LEVEL SLICING")
print("=" * 50)
lower, upper = 100, 200
sliced = np.copy(img)
mask = (img >= lower) & (img <= upper)
sliced[mask] = 255
sliced[~mask] = 0
print(f"Slicing range: [{lower}, {upper}]")
print()

# Visualization
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
fig.suptitle('Gray Level Transformations', fontsize=16, fontweight='bold')

images = [
    (img, 'Original'),
    (negative, 'Negative'),
    (log_transform, 'Log Transform'),
    (gamma_images[0], f'Gamma = {gamma_values[0]}'),
    (gamma_images[1], f'Gamma = {gamma_values[1]}'),
    (gamma_images[2], f'Gamma = {gamma_values[2]}'),
    (stretched, 'Contrast Stretched'),
    (sliced, 'Intensity Sliced'),
]

for idx, (image, title) in enumerate(images):
    row, col = idx // 3, idx % 3
    axes[row, col].imshow(image, cmap='gray')
    axes[row, col].set_title(title, fontweight='bold')
    axes[row, col].axis('off')

# Remove empty subplot
axes[2, 2].axis('off')

plt.tight_layout()
plt.savefig('outputs/gray_level_operations.png', dpi=300, bbox_inches='tight')
print("[SUCCESS] Gray level operations saved")
print()

# Transformation functions plot
fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
fig2.suptitle('Transformation Functions', fontsize=16, fontweight='bold')

r = np.arange(0, 256)

# Negative
axes2[0, 0].plot(r, 255 - r, linewidth=2)
axes2[0, 0].set_title('Negative: s = 255 - r', fontweight='bold')
axes2[0, 0].set_xlabel('Input Intensity (r)')
axes2[0, 0].set_ylabel('Output Intensity (s)')
axes2[0, 0].grid(True, alpha=0.3)

# Log
c_val = 255 / np.log(256)
axes2[0, 1].plot(r, c_val * np.log(1 + r), linewidth=2)
axes2[0, 1].set_title('Log: s = c * log(1 + r)', fontweight='bold')
axes2[0, 1].set_xlabel('Input Intensity (r)')
axes2[0, 1].set_ylabel('Output Intensity (s)')
axes2[0, 1].grid(True, alpha=0.3)

# Gamma
for gamma in [0.5, 1.0, 1.5, 2.5]:
    axes2[1, 0].plot(r, 255 * np.power(r / 255, gamma), label=f'γ = {gamma}', linewidth=2)
axes2[1, 0].set_title('Power Law: s = r^γ', fontweight='bold')
axes2[1, 0].set_xlabel('Input Intensity (r)')
axes2[1, 0].set_ylabel('Output Intensity (s)')
axes2[1, 0].legend()
axes2[1, 0].grid(True, alpha=0.3)

# Contrast stretching
axes2[1, 1].plot(r, r, linewidth=2, label='Original')
r_stretched = ((r - 50) / (200 - 50) * 255).clip(0, 255)
axes2[1, 1].plot(r, r_stretched, linewidth=2, label='Stretched [50, 200] → [0, 255]')
axes2[1, 1].set_title('Contrast Stretching', fontweight='bold')
axes2[1, 1].set_xlabel('Input Intensity (r)')
axes2[1, 1].set_ylabel('Output Intensity (s)')
axes2[1, 1].legend()
axes2[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/transformation_functions.png', dpi=300, bbox_inches='tight')
print("[SUCCESS] Transformation functions saved")
print()
print("=" * 50)
print("EXPERIMENT 7 COMPLETED SUCCESSFULLY!")
print("=" * 50)
