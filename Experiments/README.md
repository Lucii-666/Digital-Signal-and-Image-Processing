# Digital Signal and Image Processing - 14 Experiments

This folder contains 14 comprehensive experiments covering signal processing and image processing topics.

## Experiments Overview

### Signal Processing (Experiments 1-6)

1. **Analysis of Different Signals** - Generate and analyze sine, cosine, square, sawtooth, and triangular waves
2. **Linear Convolution** - Demonstrate convolution of discrete signals
3. **Cross-Correlation and Circular Convolution** - Compare different correlation and convolution methods
4. **Butterworth and Chebyshev Filters** - Design and apply IIR filters
5. **FIR Filter using Windowing Method** - Design FIR filters with various window functions
6. **FFT and IFFT** - Frequency domain analysis and reconstruction

### Image Processing (Experiments 7-14)

7. **Gray Level Operations** - Negative, log, power-law, contrast stretching
8. **Histogram Equalization** - Histogram analysis and enhancement
9. **Spatial Domain Filtering** - Smoothing and sharpening filters
10. **Non-Linear Filters** - Median, min, max, alpha-trimmed mean filters
11. **Morphological Operations** - Dilation, erosion, opening, closing
12. **Hit-or-Miss Transformation** - Pattern detection in binary images
13. **Boundary Extraction** - Various edge detection and boundary extraction methods
14. **Frequency Domain Filtering** - Ideal, Butterworth, and Gaussian filters

## Setup

### Install Required Packages

```powershell
& "D:/College/SEM 5/DSIP/Practical Codes/.venv/Scripts/python.exe" -m pip install numpy matplotlib opencv-python scipy
```

## Running Experiments

### Run All Experiments

```powershell
& "D:/College/SEM 5/DSIP/Practical Codes/.venv/Scripts/python.exe" run_all_experiments.py
```

### Run Individual Experiment

Navigate to the experiment folder and run:

```powershell
cd Exp1_Signal_Analysis
& "D:/College/SEM 5/DSIP/Practical Codes/.venv/Scripts/python.exe" experiment1.py
```

## Output Structure

Each experiment creates an `outputs` folder containing:
- Generated plots (PNG format, 300 DPI)
- Result visualizations
- Comparative analyses

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- OpenCV (cv2)
- SciPy

## Notes

- Experiments 7-14 use images from the `extracted_images` folder if available
- Synthetic data is generated if no images are found
- All plots are saved at high resolution (300 DPI) for reports
- Each experiment includes detailed console output with calculations and statistics
