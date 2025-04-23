import numpy as np
from scipy import signal, ndimage
from .base import DeblurringMethod

class RichardsonLucyDeconvolution(DeblurringMethod):
    
    def __init__(self):
        super().__init__("Richardson-Lucy Deconvolution")
    
    def get_default_params(self):
        return {
            "iterations": 30,
            "clip": True,
            "epsilon": 1e-10,  # Small constant to avoid division by zero
            "convergence_threshold": 1e-4,  # Threshold for convergence check
            "check_convergence": True  # Whether to check for convergence
        }
    
    def deblur(self, blurred_image, kernel, **kwargs):
        params = self.get_default_params()
        params.update(kwargs)
        
        iterations = params["iterations"]
        clip = params["clip"]
        epsilon = params["epsilon"]
        convergence_threshold = params["convergence_threshold"]
        check_convergence = params["check_convergence"]
        
        # Handle multi-channel images
        if len(blurred_image.shape) == 3:
            deblurred = np.zeros_like(blurred_image, dtype=np.float32)
            actual_iterations = []
            
            for i in range(blurred_image.shape[2]):
                channel_result, channel_iterations = self._richardson_lucy_channel(
                    blurred_image[:, :, i], kernel, iterations, epsilon, 
                    convergence_threshold, check_convergence
                )
                deblurred[:, :, i] = channel_result
                actual_iterations.append(channel_iterations)
                
            # Report average iterations across channels
            avg_iterations = sum(actual_iterations) / len(actual_iterations)
            print(f"Richardson-Lucy converged after an average of {avg_iterations:.1f} iterations")
        else:
            deblurred, actual_iterations = self._richardson_lucy_channel(
                blurred_image, kernel, iterations, epsilon,
                convergence_threshold, check_convergence
            )
            print(f"Richardson-Lucy converged after {actual_iterations} iterations")
        
        if clip:
            deblurred = np.clip(deblurred, 0, 255).astype(np.uint8)
        
        return deblurred
    
    def _richardson_lucy_channel(self, blurred_channel, psf, iterations, epsilon=1e-10, 
                                convergence_threshold=1e-4, check_convergence=True):
        # Convert to float for numerical stability
        blurred_channel = blurred_channel.astype(np.float32)
        psf = psf.astype(np.float32)
        psf = psf / psf.sum()
        
        # Create a reversed PSF for the convolution step
        psf_flipped = np.flip(psf)
        
        # Initialization: start with the mean of blurred image as the estimate
        estimate = np.full(blurred_channel.shape, np.mean(blurred_channel), dtype=np.float32)
        
        prev_estimate = None
        relative_change = float('inf')
        actual_iterations = 0
        
        for i in range(iterations):
            if check_convergence:
                prev_estimate = estimate.copy()
                
            # Step 1: Compute the predicted blurred image by convolving the current estimate with the PSF
            predicted_blurred = signal.convolve2d(estimate, psf, mode='same', boundary='symm')
            
            # Step 2: Compute the ratio between the observed blurred image and the predicted blurred image
            ratio = blurred_channel / (predicted_blurred + epsilon)
            
            # Step 3: Convolve the ratio with the flipped PSF
            correction = signal.convolve2d(ratio, psf_flipped, mode='same', boundary='symm')
            
            # Step 4: Update the current estimate
            estimate = estimate * correction
            
            actual_iterations = i + 1
            
            if check_convergence and i > 0:
                if np.sum(np.abs(prev_estimate)) > epsilon:
                    relative_change = np.sum(np.abs(estimate - prev_estimate)) / np.sum(np.abs(prev_estimate))
                    if relative_change < convergence_threshold:
                        break
        
        return estimate, actual_iterations