"""
Experiment 13: Boundary Extraction
Objective: To extract boundaries using morphological and edge detection methods
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

os.makedirs('outputs', exist_ok=True)

print("=" * 50)
print("EXPERIMENT 13: BOUNDARY EXTRACTION")
print("=" * 50)
print()

# Load and prepare image
img_path = '../../extracted_images/image_1.png'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    img = np.random.randint(0, 2, (400, 400), dtype=np.uint8) * 255
else:
    img = cv2.resize(img, (400, 400))
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

print(f"Image shape: {img.shape}")
print()

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# Method 1: Morphological boundary (original - erosion)
print("Method 1: Morphological Boundary")
erosion = cv2.erode(img, kernel)
boundary_morph = img - erosion
print(f"  Boundary pixels: {np.sum(boundary_morph > 0)}")
print()

# Method 2: Internal and external boundaries
print("Method 2: Internal & External Boundaries")
internal_boundary = img - erosion
dilation = cv2.dilate(img, kernel)
external_boundary = dilation - img
print(f"  Internal boundary pixels: {np.sum(internal_boundary > 0)}")
print(f"  External boundary pixels: {np.sum(external_boundary > 0)}")
print()

# Method 3: Morphological gradient
print("Method 3: Morphological Gradient")
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
print(f"  Gradient boundary pixels: {np.sum(gradient > 0)}")
print()

# Method 4: Edge detection on grayscale
print("Method 4: Edge Detection Methods")
img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) if cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) is not None else img
img_gray = cv2.resize(img_gray, (400, 400))

sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
sobel = np.sqrt(sobel_x**2 + sobel_y**2)
sobel = np.clip(sobel, 0, 255).astype(np.uint8)

canny = cv2.Canny(img_gray, 50, 150)

laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
laplacian = np.clip(np.abs(laplacian), 0, 255).astype(np.uint8)

print(f"  Sobel edges detected")
print(f"  Canny edges: {np.sum(canny > 0)} pixels")
print(f"  Laplacian edges detected")
print()

# Method 5: Contour detection
print("Method 5: Contour Detection")
contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_img = np.zeros_like(img)
cv2.drawContours(contour_img, contours, -1, 255, 1)
print(f"  Number of contours: {len(contours)}")
print(f"  Contour pixels: {np.sum(contour_img > 0)}")
print()

# Visualization
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
fig.suptitle('Boundary Extraction Methods', fontsize=16, fontweight='bold')

boundaries = [
    (img, 'Original Binary'),
    (boundary_morph, 'Morphological Boundary'),
    (internal_boundary, 'Internal Boundary'),
    (external_boundary, 'External Boundary'),
    (gradient, 'Morphological Gradient'),
    (sobel, 'Sobel Edges'),
    (canny, 'Canny Edges'),
    (laplacian, 'Laplacian Edges'),
    (contour_img, 'Contours'),
]

for idx, (image, title) in enumerate(boundaries):
    row, col = idx // 3, idx % 3
    axes[row, col].imshow(image, cmap='gray')
    axes[row, col].set_title(title, fontweight='bold')
    axes[row, col].axis('off')

plt.tight_layout()
plt.savefig('outputs/boundary_extraction.png', dpi=300, bbox_inches='tight')
print("[SUCCESS] Boundary extraction saved")
print()

# Comparison plot
fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
fig2.suptitle('Comparison of Edge Detection Methods', fontsize=16, fontweight='bold')

edges = [
    (sobel, 'Sobel'),
    (canny, 'Canny'),
    (laplacian, 'Laplacian'),
    (gradient, 'Morphological Gradient'),
]

for idx, (image, title) in enumerate(edges):
    row, col = idx // 2, idx % 2
    axes2[row, col].imshow(image, cmap='gray')
    axes2[row, col].set_title(title, fontweight='bold')
    axes2[row, col].axis('off')

plt.tight_layout()
plt.savefig('outputs/edge_comparison.png', dpi=300, bbox_inches='tight')
print("[SUCCESS] Edge comparison saved")
print()
print("=" * 50)
print("EXPERIMENT 13 COMPLETED SUCCESSFULLY!")
print("=" * 50)
