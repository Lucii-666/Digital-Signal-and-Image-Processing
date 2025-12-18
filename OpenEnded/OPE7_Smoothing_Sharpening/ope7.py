"""
Open Ended Problem 7: Smoothing and Sharpening Techniques
Objective: Apply multiple smoothing and sharpening filters to enhance images
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

os.makedirs('outputs', exist_ok=True)

print("=" * 70)
print("OPE 7: SMOOTHING & SHARPENING TECHNIQUES")
print("=" * 70)
print()

# Create test images with different characteristics
print("Creating test images...")

# Image 1: Noisy image
np.random.seed(42)
img1_clean = np.random.randint(50, 200, (256, 256), dtype=np.uint8)
noise = np.random.normal(0, 25, (256, 256))
img1 = np.clip(img1_clean + noise, 0, 255).astype(np.uint8)
cv2.imwrite('outputs/test1_noisy.png', img1)
print("[SUCCESS] Image 1: Noisy image created")

# Image 2: Blurry image
img2 = cv2.GaussianBlur(img1_clean, (15, 15), 5)
cv2.imwrite('outputs/test2_blurry.png', img2)
print("[SUCCESS] Image 2: Blurry image created")

# Image 3: Image with edges (checkerboard)
img3 = np.zeros((256, 256), dtype=np.uint8)
square_size = 32
for i in range(0, 256, square_size):
    for j in range(0, 256, square_size):
        if (i // square_size + j // square_size) % 2 == 0:
            img3[i:i+square_size, j:j+square_size] = 200
        else:
            img3[i:i+square_size, j:j+square_size] = 50
cv2.imwrite('outputs/test3_edges.png', img3)
print("[SUCCESS] Image 3: Edge pattern created")

print()

# ============================================================================
# SMOOTHING FILTERS
# ============================================================================

print("=" * 70)
print("SMOOTHING FILTERS - Noise Reduction")
print("=" * 70)
print()

# Test on noisy image
test_smooth = img1.copy()

# 1. Mean/Average Filter
mean_filtered = cv2.blur(test_smooth, (5, 5))
cv2.imwrite('outputs/smoothed_mean.png', mean_filtered)
print("1. Mean Filter (5x5):")
print("   - Replaces pixel with average of neighborhood")
print("   - Reduces noise but blurs edges")

# 2. Gaussian Filter
gaussian_filtered = cv2.GaussianBlur(test_smooth, (5, 5), 1.5)
cv2.imwrite('outputs/smoothed_gaussian.png', gaussian_filtered)
print("2. Gaussian Filter (5x5, σ=1.5):")
print("   - Weighted average using Gaussian kernel")
print("   - Better edge preservation than mean filter")

# 3. Median Filter
median_filtered = cv2.medianBlur(test_smooth, 5)
cv2.imwrite('outputs/smoothed_median.png', median_filtered)
print("3. Median Filter (5x5):")
print("   - Non-linear filter, replaces with median value")
print("   - Excellent for salt-and-pepper noise")
print("   - Preserves edges well")

# 4. Bilateral Filter
bilateral_filtered = cv2.bilateralFilter(test_smooth, 9, 75, 75)
cv2.imwrite('outputs/smoothed_bilateral.png', bilateral_filtered)
print("4. Bilateral Filter (d=9, σ_color=75, σ_space=75):")
print("   - Edge-preserving smoothing")
print("   - Considers both spatial and intensity similarity")
print("   - Best edge preservation among smoothing filters")

# 5. Box Filter
box_filtered = cv2.boxFilter(test_smooth, -1, (5, 5))
cv2.imwrite('outputs/smoothed_box.png', box_filtered)
print("5. Box Filter (5x5):")
print("   - Similar to mean but can be normalized differently")
print("   - Fast computation")

print()

# ============================================================================
# SHARPENING FILTERS
# ============================================================================

print("=" * 70)
print("SHARPENING FILTERS - Edge Enhancement")
print("=" * 70)
print()

# Test on blurry image
test_sharpen = img2.copy()

# 1. Laplacian Sharpening
laplacian = cv2.Laplacian(test_sharpen, cv2.CV_64F)
laplacian_sharpened = test_sharpen - laplacian
laplacian_sharpened = np.clip(laplacian_sharpened, 0, 255).astype(np.uint8)
cv2.imwrite('outputs/sharpened_laplacian.png', laplacian_sharpened)
print("1. Laplacian Sharpening:")
print("   - Uses second derivative for edge detection")
print("   - Output = Original - Laplacian")
print("   - Enhances all edges")

# 2. Unsharp Masking
blurred = cv2.GaussianBlur(test_sharpen, (9, 9), 10.0)
unsharp_mask = cv2.addWeighted(test_sharpen, 1.5, blurred, -0.5, 0)
cv2.imwrite('outputs/sharpened_unsharp.png', unsharp_mask)
print("2. Unsharp Masking:")
print("   - Output = α*Original - β*Blurred")
print("   - α=1.5, β=0.5")
print("   - Controlled sharpening, widely used")

# 3. High-Boost Filtering
high_boost = cv2.addWeighted(test_sharpen, 2.0, blurred, -1.0, 0)
cv2.imwrite('outputs/sharpened_highboost.png', high_boost)
print("3. High-Boost Filtering:")
print("   - More aggressive than unsharp masking")
print("   - Amplification factor A=2.0")
print("   - Emphasizes high-frequency components")

# 4. Sobel Edge Enhancement
sobel_x = cv2.Sobel(test_sharpen, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(test_sharpen, cv2.CV_64F, 0, 1, ksize=3)
sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
sobel_enhanced = test_sharpen + 0.3 * sobel_combined
sobel_enhanced = np.clip(sobel_enhanced, 0, 255).astype(np.uint8)
cv2.imwrite('outputs/sharpened_sobel.png', sobel_enhanced)
print("4. Sobel Edge Enhancement:")
print("   - Gradient-based edge detection")
print("   - Combines horizontal and vertical edges")
print("   - Good directional edge enhancement")

# 5. Custom Sharpening Kernel
kernel_sharpen = np.array([
    [-1, -1, -1],
    [-1,  9, -1],
    [-1, -1, -1]
])
custom_sharpened = cv2.filter2D(test_sharpen, -1, kernel_sharpen)
cv2.imwrite('outputs/sharpened_custom.png', custom_sharpened)
print("5. Custom Sharpening Kernel:")
print("   - User-defined 3x3 kernel")
print("   - Center weight = 9, neighbors = -1")
print("   - Direct convolution approach")

print()

# ============================================================================
# COMBINED EFFECTS
# ============================================================================

print("=" * 70)
print("COMBINED SMOOTHING + SHARPENING")
print("=" * 70)
print()

# Start with noisy blurry image
noisy_blurry = cv2.GaussianBlur(img1, (7, 7), 2)

# First denoise, then sharpen
step1 = cv2.bilateralFilter(noisy_blurry, 9, 75, 75)  # Denoise
step2 = cv2.addWeighted(step1, 1.5, cv2.GaussianBlur(step1, (5, 5), 2), -0.5, 0)  # Sharpen
cv2.imwrite('outputs/combined_denoise_sharpen.png', step2)
print("Combined Enhancement:")
print("  Step 1: Bilateral filter (noise reduction)")
print("  Step 2: Unsharp masking (sharpening)")
print("  Result: Clean and sharp image")
print()

# ============================================================================
# VISUALIZATION
# ============================================================================

print("Creating comprehensive visualizations...")

# Figure 1: Smoothing filters comparison
fig1, axes1 = plt.subplots(2, 3, figsize=(15, 10))
fig1.suptitle('Smoothing Filters Comparison (Noisy Image)', fontsize=16, fontweight='bold')

smoothed_images = [
    (test_smooth, 'Original (Noisy)'),
    (mean_filtered, 'Mean Filter'),
    (gaussian_filtered, 'Gaussian Filter'),
    (median_filtered, 'Median Filter'),
    (bilateral_filtered, 'Bilateral Filter'),
    (box_filtered, 'Box Filter'),
]

for idx, (img, title) in enumerate(smoothed_images):
    row, col = idx // 3, idx % 3
    axes1[row, col].imshow(img, cmap='gray')
    axes1[row, col].set_title(title, fontweight='bold')
    axes1[row, col].axis('off')
    
    # Add PSNR if not original
    if idx > 0:
        mse = np.mean((img1_clean.astype(float) - img.astype(float)) ** 2)
        if mse > 0:
            psnr = 10 * np.log10(255**2 / mse)
            axes1[row, col].text(0.5, -0.1, f'PSNR: {psnr:.2f} dB', 
                                ha='center', transform=axes1[row, col].transAxes)

plt.tight_layout()
plt.savefig('outputs/smoothing_comparison.png', dpi=300, bbox_inches='tight')
print("[SUCCESS] Smoothing comparison saved")

# Figure 2: Sharpening filters comparison
fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))
fig2.suptitle('Sharpening Filters Comparison (Blurry Image)', fontsize=16, fontweight='bold')

sharpened_images = [
    (test_sharpen, 'Original (Blurry)'),
    (laplacian_sharpened, 'Laplacian Sharpening'),
    (unsharp_mask, 'Unsharp Masking'),
    (high_boost, 'High-Boost Filter'),
    (sobel_enhanced, 'Sobel Enhancement'),
    (custom_sharpened, 'Custom Kernel'),
]

for idx, (img, title) in enumerate(sharpened_images):
    row, col = idx // 3, idx % 3
    axes2[row, col].imshow(img, cmap='gray')
    axes2[row, col].set_title(title, fontweight='bold')
    axes2[row, col].axis('off')

plt.tight_layout()
plt.savefig('outputs/sharpening_comparison.png', dpi=300, bbox_inches='tight')
print("[SUCCESS] Sharpening comparison saved")

# Figure 3: Edge preservation comparison
fig3, axes3 = plt.subplots(2, 3, figsize=(15, 10))
fig3.suptitle('Edge Preservation Analysis (Checkerboard Pattern)', fontsize=16, fontweight='bold')

# Add noise to checkerboard
noisy_edges = np.clip(img3 + np.random.normal(0, 20, img3.shape), 0, 255).astype(np.uint8)

edge_test_filters = [
    (noisy_edges, 'Noisy Original'),
    (cv2.blur(noisy_edges, (5, 5)), 'Mean (Poor Edge)'),
    (cv2.GaussianBlur(noisy_edges, (5, 5), 1.5), 'Gaussian (Moderate)'),
    (cv2.medianBlur(noisy_edges, 5), 'Median (Good Edge)'),
    (cv2.bilateralFilter(noisy_edges, 9, 75, 75), 'Bilateral (Best Edge)'),
    (img3, 'Clean Reference'),
]

for idx, (img, title) in enumerate(edge_test_filters):
    row, col = idx // 3, idx % 3
    axes3[row, col].imshow(img, cmap='gray')
    axes3[row, col].set_title(title, fontweight='bold')
    axes3[row, col].axis('off')

plt.tight_layout()
plt.savefig('outputs/edge_preservation_comparison.png', dpi=300, bbox_inches='tight')
print("[SUCCESS] Edge preservation comparison saved")

# Figure 4: Combined workflow
fig4, axes4 = plt.subplots(1, 4, figsize=(16, 5))
fig4.suptitle('Complete Enhancement Workflow', fontsize=16, fontweight='bold')

workflow = [
    (noisy_blurry, 'Noisy + Blurry'),
    (step1, 'After Denoising'),
    (step2, 'After Sharpening'),
    (img1_clean, 'Original Clean'),
]

for idx, (img, title) in enumerate(workflow):
    axes4[idx].imshow(img, cmap='gray')
    axes4[idx].set_title(title, fontweight='bold', fontsize=12)
    axes4[idx].axis('off')

plt.tight_layout()
plt.savefig('outputs/enhancement_workflow.png', dpi=300, bbox_inches='tight')
print("[SUCCESS] Enhancement workflow saved")
print()

# ============================================================================
# DETAILED OBSERVATIONS
# ============================================================================

print("=" * 70)
print("DETAILED OBSERVATIONS")
print("=" * 70)
print("""
SMOOTHING FILTERS:

1. MEAN FILTER:
   Effect: Moderate noise reduction
   Edge Quality: Significant blurring
   Speed: Fast
   Best For: Quick smoothing when edges not critical

2. GAUSSIAN FILTER:
   Effect: Smooth noise reduction
   Edge Quality: Better than mean, still blurs
   Speed: Fast
   Best For: General-purpose smoothing

3. MEDIAN FILTER:
   Effect: Excellent salt-and-pepper noise removal
   Edge Quality: Good edge preservation
   Speed: Moderate (depends on window size)
   Best For: Impulse noise, edge-preserving smoothing

4. BILATERAL FILTER:
   Effect: Strong noise reduction
   Edge Quality: Excellent edge preservation
   Speed: Slow (computationally intensive)
   Best For: High-quality denoising with edge preservation

5. BOX FILTER:
   Effect: Similar to mean
   Edge Quality: Moderate blurring
   Speed: Very fast
   Best For: Real-time applications

SHARPENING FILTERS:

1. LAPLACIAN SHARPENING:
   Effect: Enhances all edges uniformly
   Quality: Can amplify noise
   Strength: Moderate
   Best For: General sharpening

2. UNSHARP MASKING:
   Effect: Controlled edge enhancement
   Quality: Natural appearance
   Strength: Adjustable via parameters
   Best For: Professional image enhancement

3. HIGH-BOOST FILTERING:
   Effect: Aggressive sharpening
   Quality: Can over-sharpen
   Strength: High
   Best For: Very blurry images

4. SOBEL ENHANCEMENT:
   Effect: Directional edge enhancement
   Quality: Emphasizes specific edge directions
   Strength: Moderate
   Best For: Specific edge emphasis

5. CUSTOM KERNEL:
   Effect: User-controllable
   Quality: Depends on kernel design
   Strength: Fully adjustable
   Best For: Specialized applications

EDGE PRESERVATION RANKING:
1. Bilateral Filter (Best)
2. Median Filter (Very Good)
3. Gaussian Filter (Moderate)
4. Mean Filter (Poor)
5. Box Filter (Poor)

SHARPENING EFFECTIVENESS:
1. Unsharp Masking (Most Balanced)
2. High-Boost (Most Aggressive)
3. Laplacian (Good General Use)
4. Sobel (Directional)
5. Custom (Variable)
""")
print()

print("=" * 70)
print("CONCLUSIONS")
print("=" * 70)
print("""
SMOOTHING:
✓ Bilateral filter best for quality (slow)
✓ Median filter best for impulse noise
✓ Gaussian filter best balance speed/quality
✓ Always trade-off between noise reduction and edge preservation
✓ Filter size affects blur amount

SHARPENING:
✓ Unsharp masking most versatile
✓ Must denoise before sharpening to avoid amplifying noise
✓ Over-sharpening creates halos and artifacts
✓ Different methods emphasize different features
✓ Parameters critical for good results

COMBINED WORKFLOW:
✓ Always denoise first, then sharpen
✓ Use edge-preserving smoothing (bilateral/median)
✓ Apply moderate sharpening (unsharp masking)
✓ Iterative refinement often needed
✓ Visual inspection important

RECOMMENDATIONS:
- Noisy images: Bilateral → Unsharp masking
- Blurry images: Direct sharpening (high-boost)
- Both issues: Median → Unsharp masking
- Real-time: Gaussian → Laplacian
- High quality: Bilateral → Unsharp with tuning
""")
print()

print("=" * 70)
print("OPE 7 COMPLETED SUCCESSFULLY!")
print("=" * 70)
