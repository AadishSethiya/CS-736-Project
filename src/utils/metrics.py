import numpy as np
import torch
import lpips
from torchvision import transforms

# LPIPS model (loaded once globally)
lpips_model = lpips.LPIPS(net='alex')

def to_tensor_lpips(img):
    # Convert a NumPy image (H, W, C) to normalized torch tensor in [-1, 1].
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])
    return transform(img).unsqueeze(0)

def calculate_lpips(img1, img2):
    img1_tensor = to_tensor_lpips(img1.astype(np.uint8))
    img2_tensor = to_tensor_lpips(img2.astype(np.uint8))
    
    with torch.no_grad():
        score = lpips_model(img1_tensor, img2_tensor)
    return score.item()

def evaluate_deblurring(original, blurred, deblurred):
    blurred_lpips = calculate_lpips(original, blurred)
    deblurred_lpips = calculate_lpips(original, deblurred)

    return {
        "blurred": {
            "lpips": blurred_lpips
        },
        "deblurred": {
            "lpips": deblurred_lpips
        },
        "improvement": {
            "lpips": blurred_lpips - deblurred_lpips  # lower is better
        }
    }
