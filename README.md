# Image Deblurring Framework

This repository provides a comprehensive framework for image deblurring using various restoration algorithms. The framework includes implementations of classic deblurring methods like Wiener Deconvolution, Richardson-Lucy Deconvolution, and Total Variation Deblurring.

## GitHub Link for Results and Report

```
https://github.com/AadishSethiya/CS-736-Project.git
```

## Features

- Multiple deblurring algorithms with customizable parameters
- Motion blur simulation with adjustable kernel size and angle
- Gaussian noise addition
- Quantitative evaluation using LPIPS (Learned Perceptual Image Patch Similarity)
- Visualization tools for results comparison
- Batch processing capability for multiple images

## Installation

### Requirements

```
numpy
scipy
opencv-python (cv2)
matplotlib
torch
lpips
torchvision
```

You can install the required packages using pip:

```bash
pip install numpy scipy opencv-python matplotlib torch lpips torchvision
```

## Usage

### Basic Usage

```bash
python main.py --image path/to/image.jpg --method wiener
```

### Command Line Arguments

- `--image`: Path to input image (if not provided, all images in data directory are processed)
- `--data_dir`: Directory containing input images (default: '../data')
- `--kernel_size`: Size of motion blur kernel (default: 20)
- `--angle`: Angle of motion blur in degrees (default: 0)
- `--noise_sigma`: Standard deviation of Gaussian noise (default: 3)
- `--method`: Deblurring method to use (choices: 'wiener', 'richardson_lucy', 'total_variation')
- `--output`: Output directory for results (default: '../results')
- `--display`: Display results after processing
- `--no_save`: Skip saving results to disk

### Method-specific Parameters

#### Wiener Deconvolution
- `--nsr`: Noise-to-signal ratio (default: 0.01)
- `--pad_size`: Padding size to reduce boundary artifacts (default: 30)

#### Richardson-Lucy Deconvolution
- `--rl_iterations`: Maximum number of iterations (default: 50)

#### Total Variation Deblurring
- `--tv_lambda`: Regularization parameter (default: 0.01)
- `--tv_iterations`: Maximum number of iterations (default: 500)
- `--pad_size`: Padding size to reduce boundary artifacts (default: 30)

### Examples

```bash
# Use Wiener deconvolution with custom parameters
python main.py --image sample.jpg --method wiener --kernel_size 15 --angle 45 --nsr 0.02 --display

# Use Richardson-Lucy deconvolution
python main.py --image sample.jpg --method richardson_lucy --rl_iterations 100 --noise_sigma 2

# Use Total Variation deblurring
python main.py --image sample.jpg --method total_variation --tv_lambda 0.005 --tv_iterations 300

# Process all images in the data directory
python main.py --data_dir ./my_images --method wiener --output ./my_results
```

## Project Structure

- `main.py`: Entry point of the application
- `utils/`: Utility functions for image processing and visualization
  - `image_utils.py`: Image loading, saving, blur generation
  - `metrics.py`: LPIPS evaluation metrics
  - `visualization.py`: Functions for displaying and saving results
- `deblur/`: Deblurring algorithm implementations
  - `base.py`: Abstract base class for deblurring methods
  - `wiener.py`: Wiener deconvolution implementation
  - `richardson_lucy.py`: Richardson-Lucy deconvolution implementation
  - `total_variation.py`: Total Variation regularization implementation

## Results

For each processed image, the framework generates:
- A comparison figure showing the original, blurred, and deblurred images
- A visualization of the point spread function (PSF)
- Metrics file with LPIPS scores and improvement measurements
- Execution time statistics

Results are organized in the output directory by method and image name.