"""
Experiment 12: Hit-or-Miss Transformation
Objective: To perform hit-or-miss transformation for pattern detection
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
import os

os.makedirs('outputs', exist_ok=True)

print("=" * 50)
print("EXPERIMENT 12: HIT-OR-MISS TRANSFORMATION")
print("=" * 50)
print()

# Create binary image with patterns
img = np.zeros((200, 200), dtype=np.uint8)
# Add some patterns
img[50:52, 50:52] = 255  # Isolated point
img[100:102, 100:110] = 255  # Horizontal line
img[150, 150] = 255  # Isolated point
cv2.rectangle(img, (30, 30), (70, 70), 255, 2)  # Rectangle

print(f"Image shape: {img.shape}")
print(f"Foreground pixels: {np.sum(img == 255)}")
print()

# Structuring elements for hit-or-miss
# Detect isolated points
se_hit_point = np.array([[0, 0, 0],
                         [0, 1, 0],
                         [0, 0, 0]], dtype=np.uint8)

se_miss_point = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]], dtype=np.uint8)

# Detect corners
se_hit_corner = np.array([[1, 1, 0],
                          [1, 0, 0],
                          [0, 0, 0]], dtype=np.uint8)

se_miss_corner = np.array([[0, 0, 1],
                           [0, 0, 1],
                           [1, 1, 1]], dtype=np.uint8)

print("Structuring Elements Defined:")
print("- Isolated Point Detection")
print("- Corner Detection")
print()

# Perform hit-or-miss transformation
def hit_or_miss(image, hit_se, miss_se):
    hit = cv2.erode(image, hit_se)
    miss = cv2.erode(255 - image, miss_se)
    return cv2.bitwise_and(hit, miss)

hmt_points = hit_or_miss(img, se_hit_point, se_miss_point)
hmt_corners = hit_or_miss(img, se_hit_corner, se_miss_corner)

print("Hit-or-Miss Results:")
print(f"- Isolated points detected: {np.sum(hmt_points > 0)}")
print(f"- Corners detected: {np.sum(hmt_corners > 0)}")
print()

# Using scipy
binary_img = (img > 0).astype(int)
structure1 = np.array([[0, 0, 0],
                       [0, 1, 0],
                       [0, 0, 0]])
hmt_scipy = ndimage.binary_hit_or_miss(binary_img, structure1=structure1).astype(np.uint8) * 255

print(f"SciPy HMT result: {np.sum(hmt_scipy > 0)} patterns detected")
print()

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Hit-or-Miss Transformation', fontsize=16, fontweight='bold')

images = [
    (img, 'Original Binary Image'),
    (hmt_points, 'Isolated Points'),
    (hmt_corners, 'Corners'),
    (hmt_scipy, 'SciPy HMT'),
]

for idx, (image, title) in enumerate(images):
    row, col = idx // 3, idx % 3
    axes[row, col].imshow(image, cmap='gray')
    axes[row, col].set_title(title, fontweight='bold')
    axes[row, col].axis('off')

# Structuring elements visualization
axes[1, 1].imshow(se_hit_point * 100 + se_miss_point * 50, cmap='viridis')
axes[1, 1].set_title('Point Detection SE', fontweight='bold')
axes[1, 1].axis('off')

axes[1, 2].imshow(se_hit_corner * 100 + se_miss_corner * 50, cmap='viridis')
axes[1, 2].set_title('Corner Detection SE', fontweight='bold')
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('outputs/hit_or_miss.png', dpi=300, bbox_inches='tight')
print("[SUCCESS] Hit-or-miss results saved")
print()
print("=" * 50)
print("EXPERIMENT 12 COMPLETED SUCCESSFULLY!")
print("=" * 50)
