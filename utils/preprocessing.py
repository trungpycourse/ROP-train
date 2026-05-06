import cv2
import numpy as np
from typing import Union, Tuple
import torch
from PIL import Image

def gamma_correction(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Apply gamma correction to image.
    
    Args:
        image (np.ndarray): Input image (0-255 uint8 or 0-1 float)
        gamma (float): Gamma value. Values > 1 make image darker, < 1 make it brighter
    
    Returns:
        np.ndarray: Gamma corrected image
    """
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
        
    corrected = np.power(image, gamma)
    
    if image.dtype == np.uint8:
        corrected = (corrected * 255).astype(np.uint8)
    
    return corrected

def clahe_enhancement(image, 
                     clip_limit: float = 2.0,
                     tile_grid_size: Tuple[int, int] = (8, 8)):
    """Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Args:
        image: Input image (PIL Image or numpy array)
        clip_limit (float): Contrast limit for CLAHE
        tile_grid_size (tuple): Size of grid for histogram equalization
    
    Returns:
        Same type as input: Enhanced image
    """
    # Convert PIL Image to numpy array if necessary
    if hasattr(image, 'convert'):  # Check if it's a PIL Image
        is_pil = True
        image_np = np.array(image)
    else:
        is_pil = False
        image_np = image

    if len(image_np.shape) == 3:
        # Convert to LAB color space
        lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        cl = clahe.apply(l)
        
        # Merge channels
        enhanced_lab = cv2.merge((cl, a, b))
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced = clahe.apply(image_np)
    
    # Convert back to PIL Image if input was PIL
    if is_pil:
        from PIL import Image
        enhanced = Image.fromarray(enhanced)
    
    return enhanced

def dehazing(image: np.ndarray, 
            omega: float = 0.95, 
            t0: float = 0.1, 
            radius: int = 7) -> np.ndarray:
    """Remove haze from image using Dark Channel Prior method.
    
    Args:
        image (np.ndarray): Input RGB image
        omega (float): Haze removal factor (0-1)
        t0 (float): Minimum transmission
        radius (int): Radius for dark channel calculation
    
    Returns:
        np.ndarray: Dehazed image
    """
    if len(image.shape) != 3:
        raise ValueError("Input image must be RGB")
    
    # Convert to float
    image = image.astype(np.float32) / 255.0
    
    # Calculate dark channel
    dark_channel = np.min(image, axis=2)
    dark_channel = cv2.erode(dark_channel, cv2.getStructuringElement(cv2.MORPH_RECT, (radius, radius)))
    
    # Estimate atmospheric light
    flat_dark = dark_channel.flatten()
    flat_image = image.reshape(-1, 3)
    num_pixels = flat_dark.size
    num_bright = int(num_pixels * 0.001)
    bright_idx = np.argpartition(flat_dark, -num_bright)[-num_bright:]
    atmospheric = np.max(flat_image[bright_idx], axis=0)
    
    # Estimate transmission
    transmission = 1 - omega * dark_channel
    transmission = np.maximum(transmission, t0)
    
    # Recover image
    result = np.empty_like(image)
    for i in range(3):
        result[:, :, i] = (image[:, :, i] - atmospheric[i]) / transmission + atmospheric[i]
    
    # Clip values and convert back to uint8
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    return result

def normalize_image(image: np.ndarray,
                   mean: Union[float, Tuple[float, ...]] = None,
                   std: Union[float, Tuple[float, ...]] = None) -> np.ndarray:
    """Normalize image by mean and standard deviation.
    
    Args:
        image (np.ndarray): Input image (0-255 uint8 or 0-1 float)
        mean (float or tuple): Mean for normalization. If None, computed from image
        std (float or tuple): Standard deviation for normalization. If None, computed from image
    
    Returns:
        np.ndarray: Normalized image
    """
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    
    # If mean/std not provided, compute from image
    if mean is None:
        mean = image.mean(axis=(0, 1))
    if std is None:
        std = image.std(axis=(0, 1))
    
    # Normalize
    normalized = (image - mean) / std
    return normalized

def equalize_hist_rgb(image: np.ndarray) -> np.ndarray:
    """Equalize histogram for RGB image while preserving colors.
    
    Args:
        image (np.ndarray): Input RGB image
    
    Returns:
        np.ndarray: Histogram equalized image
    """
    if len(image.shape) != 3:
        return cv2.equalizeHist(image)
    
    # Convert to YUV
    yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    
    # Equalize Y channel
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    
    # Convert back to RGB
    result = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
    return result

class ImagePreprocessor:
    """Class to handle image preprocessing pipeline."""
    
    def __init__(self, size=None):
        """
        Args:
            size (tuple, optional): Target size for resizing
        """
        self.size = size
        self.transforms = []
    
    def add_transform(self, transform_fn, **kwargs):
        """Add a transform to the pipeline.
        
        Args:
            transform_fn: Transform function to add
            **kwargs: Arguments for the transform function
        """
        self.transforms.append((transform_fn, kwargs))
    
    def __call__(self, image):
        """Apply all transforms in sequence.
        
        Args:
            image (np.ndarray): Input image
        
        Returns:
            np.ndarray: Processed image
        """
        if self.size:
            image = cv2.resize(image, self.size)
        
        for transform_fn, kwargs in self.transforms:
            image = transform_fn(image, **kwargs)
        
        return image