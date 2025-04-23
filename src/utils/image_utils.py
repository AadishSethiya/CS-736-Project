import numpy as np
from scipy import ndimage
import cv2
import os

def generate_motion_blur_kernel(kernel_size=15, angle=45):
    if kernel_size % 2 == 0:
        kernel_size += 1

    # create horizontal line
    k = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    k[kernel_size//2, :] = 1.0

    # rotate to desired angle
    k = ndimage.rotate(k, angle, reshape=False, order=1, mode='constant', cval=0.0)

    # normalize
    return k / k.sum()

def apply_motion_blur(image, kernel_size=15, angle=45):
    kernel = generate_motion_blur_kernel(kernel_size, angle)
    if image.ndim == 3:
        blurred = np.stack([
            ndimage.convolve(image[...,c].astype(np.float32), kernel, mode='reflect')
            for c in range(3)
        ], axis=-1)
    else:
        blurred = ndimage.convolve(image.astype(np.float32), kernel, mode='reflect')

    return np.clip(blurred, 0, 255).astype(np.uint8), kernel

def add_gaussian_noise(image, mean=0, sigma=10):
    image_float = image.astype(np.float32)
    noise = np.random.normal(mean, sigma, image.shape)
    noisy_image = image_float + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

def save_kernel_visualization(kernel, output_path):
    # Normalize kernel for better visualization
    kernel_vis = kernel.copy()
    if kernel_vis.max() > 0:
        kernel_vis = kernel_vis / kernel_vis.max() * 255
        
    # Resize for better visibility (optional)
    kernel_size = kernel.shape[0]
    scale_factor = max(1, int(200 / kernel_size))
    kernel_vis_large = cv2.resize(
        kernel_vis, 
        (kernel_size * scale_factor, kernel_size * scale_factor), 
        interpolation=cv2.INTER_NEAREST
    )
    
    cv2.imwrite(output_path, kernel_vis_large)

def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
    return image

def save_images(images_dict, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for name, image in images_dict.items():
        output_path = os.path.join(output_dir, f"{name}.png")
        cv2.imwrite(output_path, image)