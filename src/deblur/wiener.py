import numpy as np
from scipy import fft
from .base import DeblurringMethod

class WienerDeconvolution(DeblurringMethod):
    
    def __init__(self):
        super().__init__("Wiener Deconvolution")
    
    def get_default_params(self):
        return {
            "nsr": 0.01,
            "clip": True,
            "pad_size": 30
        }
    
    def deblur(self, blurred_image, kernel, **kwargs):
        params = self.get_default_params()
        params.update(kwargs)
        
        nsr = params["nsr"]
        clip = params["clip"]
        pad_size = params["pad_size"]
        
        # Handle multi-channel images
        if len(blurred_image.shape) == 3:
            deblurred = np.zeros_like(blurred_image, dtype=np.float32)
            for i in range(blurred_image.shape[2]):
                deblurred[:, :, i] = self._wiener_channel(blurred_image[:, :, i], kernel, nsr, pad_size)
        else:
            deblurred = self._wiener_channel(blurred_image, kernel, nsr, pad_size)
        
        if clip:
            deblurred = np.clip(deblurred, 0, 255).astype(np.uint8)
        
        return deblurred
    
    def _wiener_channel(self, blurred_channel, psf, nsr, pad_size=None):
        # Convert inputs to float
        blurred_channel = blurred_channel.astype(np.float32)
        psf = psf.astype(np.float32)
        
        if pad_size is None:
            pad_size = max(psf.shape) * 2
        original_height, original_width = blurred_channel.shape
        
        # Apply padding to reduce boundary artifacts
        padded_image = self._pad_image(blurred_channel, pad_size)
        
        # Pad the PSF to match the padded image size
        padded_height, padded_width = padded_image.shape
        psf_padded = np.zeros((padded_height, padded_width), dtype=np.float32)
        psf_center = np.array(psf.shape) // 2
        psf_start = np.array((padded_height, padded_width)) // 2 - psf_center
        
        psf_padded[psf_start[0]:psf_start[0]+psf.shape[0], 
                psf_start[1]:psf_start[1]+psf.shape[1]] = psf
        
        # Perform FFT on the padded image and PSF
        blurred_fft = fft.fft2(padded_image)
        psf_fft = fft.fft2(fft.ifftshift(psf_padded))
        
        # Wiener deconvolution in frequency domain
        # G(u,v) = F(u,v) * H*(u,v) / (|H(u,v)|^2 + K)
        # where K is the noise-to-signal power ratio
        psf_fft_conj = np.conj(psf_fft)
        wiener_filter = psf_fft_conj / (np.abs(psf_fft)**2 + nsr)
        deblurred_fft = blurred_fft * wiener_filter
        
        # Return to spatial domain
        deblurred_padded = np.real(fft.ifft2(deblurred_fft))
        
        # Crop the image back to original size (remove padding)
        deblurred = self._unpad_image(deblurred_padded, original_height, original_width, pad_size)
        
        return deblurred
    
    def _pad_image(self, image, pad_size):
        # Create padded image
        padded = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size)), mode='reflect')
        return padded
    
    def _unpad_image(self, padded_image, original_height, original_width, pad_size):
        return padded_image[pad_size:pad_size+original_height, pad_size:pad_size+original_width]