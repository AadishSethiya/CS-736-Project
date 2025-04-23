import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def display_results(images, titles, metrics=None, figsize=(15, 10), save_path=None):
    num_images = len(images)
    fig, axes = plt.subplots(1, num_images, figsize=figsize)
    
    if num_images == 1:
        axes = [axes]
    
    for i, (img, title) in enumerate(zip(images, titles)):
        if len(img.shape) == 3:
            display_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            display_img = img
        
        axes[i].imshow(display_img, cmap='gray')
        
        # Add metrics if provided
        if metrics is not None and i < len(metrics) and metrics[i] is not None:
            metric_str = '\n'.join([f"{k}: {v:.4f}" for k, v in metrics[i].items()])
            axes[i].set_title(f"{title}\n{metric_str}")
        else:
            axes[i].set_title(title)
            
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def create_comparison_figure(original, blurred, deblurred, method_name, metrics, output_path):
    plt.figure(figsize=(18, 6))
    
    # Original image
    plt.subplot(1, 3, 1)
    if len(original.shape) == 3:
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_RGB2GRAY), cmap='gray')
    else:
        plt.imshow(original, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')
    
    # Blurred image
    plt.subplot(1, 3, 2)
    if len(blurred.shape) == 3:
        plt.imshow(cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY), cmap='gray')
    else:
        plt.imshow(blurred, cmap='gray')
    plt.title(f"Blurred & Noisy Image\nLPIPS: {metrics['blurred']['lpips']:.2f}")
    plt.axis('off')
    
    # Deblurred image
    plt.subplot(1, 3, 3)
    if len(deblurred.shape) == 3:
        plt.imshow(cv2.cvtColor(deblurred, cv2.COLOR_RGB2GRAY), cmap='gray')
    else:
        plt.imshow(deblurred, cmap='gray')
    plt.title(f"Deblurred Image ({method_name})\nLPIPS: {metrics['deblurred']['lpips']:.2f}")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def visualize_kernel(kernel, title="Point Spread Function (PSF)", figsize=(6, 6), save_path=None):
    plt.figure(figsize=figsize)
    plt.imshow(kernel, cmap='viridis')
    plt.colorbar(label='Value')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.close()
    
    return save_path