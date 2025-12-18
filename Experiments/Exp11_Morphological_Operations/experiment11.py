"""
Experiment 11: Morphological Operations
Objective: To perform morphological operations - dilation, erosion, opening, closing
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

os.makedirs('outputs', exist_ok=True)

print("=" * 50)
print("EXPERIMENT 11: MORPHOLOGICAL OPERATIONS")
print("=" * 50)
print()

# Create or load binary image
img_path = '../../extracted_images/image_1.png'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    img = np.random.randint(0, 2, (400, 400), dtype=np.uint8) * 255
else:
    img = cv2.resize(img, (400, 400))
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

print(f"Image shape: {img.shape}")
print()

# Structuring elements
kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

print("Structuring Elements:")
print("- Rectangular 5x5")
print("- Elliptical 5x5")
print("- Cross 5x5")
print()

# Basic morphological operations
erosion = cv2.erode(img, kernel_rect, iterations=1)
dilation = cv2.dilate(img, kernel_rect, iterations=1)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_rect)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_rect)

print("Operations performed:")
print("- Erosion: Shrinks objects")
print("- Dilation: Expands objects")
print("- Opening: Erosion then Dilation")
print("- Closing: Dilation then Erosion")
print()

# Advanced morphological operations
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel_rect)
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel_rect)
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel_rect)

print("Advanced Operations:")
print("- Gradient: Dilation - Erosion")
print("- Top Hat: Original - Opening")
print("- Black Hat: Closing - Original")
print()

# Visualization
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
fig.suptitle('Morphological Operations', fontsize=16, fontweight='bold')

operations = [
    (img, 'Original'),
    (erosion, 'Erosion'),
    (dilation, 'Dilation'),
    (opening, 'Opening'),
    (closing, 'Closing'),
    (gradient, 'Gradient'),
    (tophat, 'Top Hat'),
    (blackhat, 'Black Hat'),
]

for idx, (image, title) in enumerate(operations):
    row, col = idx // 3, idx % 3
    axes[row, col].imshow(image, cmap='gray')
    axes[row, col].set_title(title, fontweight='bold')
    axes[row, col].axis('off')

axes[2, 2].axis('off')

plt.tight_layout()
plt.savefig('outputs/morphological_operations.png', dpi=300, bbox_inches='tight')
print("[SUCCESS] Morphological operations saved")
print()

# Different structuring elements
fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))
fig2.suptitle('Different Structuring Elements', fontsize=16, fontweight='bold')

erosion_rect = cv2.erode(img, kernel_rect, iterations=1)
erosion_ellipse = cv2.erode(img, kernel_ellipse, iterations=1)
erosion_cross = cv2.erode(img, kernel_cross, iterations=1)

dilation_rect = cv2.dilate(img, kernel_rect, iterations=1)
dilation_ellipse = cv2.dilate(img, kernel_ellipse, iterations=1)
dilation_cross = cv2.dilate(img, kernel_cross, iterations=1)

se_operations = [
    (erosion_rect, 'Erosion - Rect'),
    (erosion_ellipse, 'Erosion - Ellipse'),
    (erosion_cross, 'Erosion - Cross'),
    (dilation_rect, 'Dilation - Rect'),
    (dilation_ellipse, 'Dilation - Ellipse'),
    (dilation_cross, 'Dilation - Cross'),
]

for idx, (image, title) in enumerate(se_operations):
    row, col = idx // 3, idx % 3
    axes2[row, col].imshow(image, cmap='gray')
    axes2[row, col].set_title(title, fontweight='bold')
    axes2[row, col].axis('off')

plt.tight_layout()
plt.savefig('outputs/structuring_elements.png', dpi=300, bbox_inches='tight')
print("[SUCCESS] Structuring elements comparison saved")
print()
print("=" * 50)
print("EXPERIMENT 11 COMPLETED SUCCESSFULLY!")
print("=" * 50)
