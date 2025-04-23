import numpy as np
from scipy import fft
from .base import DeblurringMethod

class TotalVariationDeblurring(DeblurringMethod):
    
    def __init__(self):
        super().__init__("Total Variation Deblurring")
    
    def get_default_params(self):
        return {
            "lambda_param": 0.01,  # Regularization parameter
            "iterations": 50,      # Number of iterations
            "clip": True,          # Whether to clip values to [0, 255]
            "epsilon": 1e-6,       # Small constant to avoid division by zero in gradient calculation
            "step_size": 0.1,      # Step size for gradient descent
            "pad_size": 30         # Padding size to reduce boundary artifacts
        }
    
    def deblur(self, blurred_image, kernel, **kwargs):
        params = self.get_default_params()
        params.update(kwargs)
        
        lambda_param = params["lambda_param"]
        iterations = params["iterations"]
        clip = params["clip"]
        epsilon = params["epsilon"]
        step_size = params["step_size"]
        pad_size = params["pad_size"]
        
        # Handle multi-channel images
        if len(blurred_image.shape) == 3:
            deblurred = np.zeros_like(blurred_image, dtype=np.float32)
            for i in range(blurred_image.shape[2]):
                deblurred[:, :, i] = self._tv_channel(
                    blurred_image[:, :, i], kernel, lambda_param, iterations,
                    epsilon, step_size, pad_size
                )
        else:
            deblurred = self._tv_channel(
                blurred_image, kernel, lambda_param, iterations,
                epsilon, step_size, pad_size
            )
        
        if clip:
            deblurred = np.clip(deblurred, 0, 255).astype(np.uint8)
        
        return deblurred
    
    def _tv_channel(self, blurred_channel, psf, lambda_param, iterations, 
                   epsilon, step_size, pad_size):
        # Convert to float for numerical stability
        blurred_channel = blurred_channel.astype(np.float32)
        psf = psf.astype(np.float32)
        
        original_height, original_width = blurred_channel.shape
        
        # Pad the image and PSF to reduce boundary effects
        padded_blurred = self._pad_image(blurred_channel, pad_size)
        padded_psf = self._pad_psf(psf, padded_blurred.shape)
        
        # Move to frequency domain
        blurred_fft = fft.fft2(padded_blurred)
        psf_fft = fft.fft2(fft.ifftshift(padded_psf))
        psf_fft_conj = np.conj(psf_fft)
        
        # Initialize the estimate with the blurred image
        estimate = padded_blurred.copy()
        prev_cost = np.inf
        
        for i in range(iterations):
            # Calculate TV gradient: div(∇u / |∇u|)
            grad_x, grad_y = self._gradient(estimate)
            grad_norm = np.sqrt(grad_x**2 + grad_y**2 + epsilon)
            div_term = self._divergence(grad_x / grad_norm, grad_y / grad_norm)
            
            # Data fidelity gradient in spatial domain: H^T * (H*u - g)
            estimate_fft = fft.fft2(estimate)
            diff_fft = estimate_fft * psf_fft - blurred_fft
            data_grad = np.real(fft.ifft2(diff_fft * psf_fft_conj))
            
            # Total gradient: data fidelity + TV regularization
            gradient = data_grad - lambda_param * div_term
            
            # Update estimate with gradient descent
            estimate -= step_size * gradient
            
            # Every 10 iterations, check for convergence
            if i % 10 == 0:
                diff_fft = estimate_fft * psf_fft - blurred_fft
                data_term = np.sum(np.abs(diff_fft)**2)
                tv_term = np.sum(np.sqrt(grad_x**2 + grad_y**2 + epsilon))
                current_cost = data_term + lambda_param * tv_term
                
                if prev_cost > epsilon and not np.isnan(prev_cost) and not np.isinf(prev_cost):
                    improvement = (prev_cost - current_cost) / prev_cost
                else:
                    improvement = 0  # Default value when previous cost is invalid
                
                prev_cost = current_cost
                
                if improvement < 1e-4 and i > 10:
                    print(f"TV deblurring converged after {i} iterations")
                    break

        deblurred = self._unpad_image(estimate, original_height, original_width, pad_size)
        
        return deblurred
    
    def _gradient(self, image):
        # Compute x and y gradients using forward differences
        grad_x = np.zeros_like(image)
        grad_y = np.zeros_like(image)
        
        grad_x[:, :-1] = image[:, 1:] - image[:, :-1]
        grad_y[:-1, :] = image[1:, :] - image[:-1, :]
        
        return grad_x, grad_y
    
    def _divergence(self, grad_x, grad_y):
        # Compute divergence using backward differences
        div_x = np.zeros_like(grad_x)
        div_y = np.zeros_like(grad_y)
        
        div_x[:, 1:] = grad_x[:, 1:] - grad_x[:, :-1]
        div_y[1:, :] = grad_y[1:, :] - grad_y[:-1, :]
        
        # Handle boundary conditions
        div_x[:, 0] = grad_x[:, 0]
        div_y[0, :] = grad_y[0, :]
        
        return div_x + div_y
    
    def _pad_image(self, image, pad_size):
        # Create padded image with reflective boundaries
        padded = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size)), mode='reflect')
        
        return padded
    
    def _pad_psf(self, psf, output_shape):
        # Create zero-padded PSF
        psf_padded = np.zeros(output_shape, dtype=np.float32)
        
        # Calculate center for positioning the PSF
        psf_center = np.array(psf.shape) // 2
        output_center = np.array(output_shape) // 2
        
        # Calculate start indices
        start_x = output_center[1] - psf_center[1]
        start_y = output_center[0] - psf_center[0]
        
        # Place PSF in center of padded array
        psf_padded[start_y:start_y+psf.shape[0], start_x:start_x+psf.shape[1]] = psf
        
        return psf_padded
    
    def _unpad_image(self, padded_image, original_height, original_width, pad_size):
        # Extract original image from padded version
        return padded_image[pad_size:pad_size+original_height, pad_size:pad_size+original_width]