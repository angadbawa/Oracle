"""Utility functions for Oracle project."""

import numpy as np
import torch
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Union, Optional, Any
import logging
from functools import wraps
import time

logger = logging.getLogger('oracle.utils')

def safe_execute(func, default_value=None, log_errors=True):
    """
    Decorator for safe function execution with error handling.
    
    Args:
        func: Function to wrap
        default_value: Value to return on error
        log_errors: Whether to log errors
        
    Returns:
        Wrapped function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if log_errors:
                logger.error(f"Error in {func.__name__}: {e}")
            return default_value
    return wrapper

def timer(func):
    """
    Decorator to measure function execution time.
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper

def validate_image(image: Union[str, Path, Image.Image, np.ndarray]) -> Optional[Image.Image]:
    """
    Validate and convert image to PIL Image.
    
    Args:
        image: Input image in various formats
        
    Returns:
        PIL Image or None if invalid
    """
    try:
        if isinstance(image, (str, Path)):
            path = Path(image)
            if not path.exists():
                logger.error(f"Image file not found: {path}")
                return None
            return Image.open(path).convert('RGB')
        
        elif isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            return Image.fromarray(image)
        
        elif isinstance(image, Image.Image):
            return image.convert('RGB')
        
        else:
            logger.error(f"Unsupported image type: {type(image)}")
            return None
            
    except Exception as e:
        logger.error(f"Error validating image: {e}")
        return None

def resize_image(image: Image.Image, target_size: Tuple[int, int], maintain_aspect: bool = True) -> Image.Image:
    """
    Resize image to target size.
    
    Args:
        image: PIL Image
        target_size: Target (width, height)
        maintain_aspect: Whether to maintain aspect ratio
        
    Returns:
        Resized PIL Image
    """
    if maintain_aspect:
        image.thumbnail(target_size, Image.Resampling.LANCZOS)
        
        # Create new image with target size and paste resized image
        new_image = Image.new('RGB', target_size, (255, 255, 255))
        paste_x = (target_size[0] - image.width) // 2
        paste_y = (target_size[1] - image.height) // 2
        new_image.paste(image, (paste_x, paste_y))
        
        return new_image
    else:
        return image.resize(target_size, Image.Resampling.LANCZOS)

def numpy_to_pil(array: np.ndarray) -> Image.Image:
    """
    Convert numpy array to PIL Image.
    
    Args:
        array: Numpy array
        
    Returns:
        PIL Image
    """
    if array.dtype != np.uint8:
        array = (array * 255).astype(np.uint8)
    
    if len(array.shape) == 2:  # Grayscale
        return Image.fromarray(array, mode='L')
    elif len(array.shape) == 3:  # RGB
        return Image.fromarray(array, mode='RGB')
    else:
        raise ValueError(f"Unsupported array shape: {array.shape}")

def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """
    Convert PIL Image to numpy array.
    
    Args:
        image: PIL Image
        
    Returns:
        Numpy array
    """
    return np.array(image)

def prepare_mask(mask: Union[Image.Image, np.ndarray], target_size: Optional[Tuple[int, int]] = None) -> Image.Image:
    """
    Prepare mask for inpainting.
    
    Args:
        mask: Input mask
        target_size: Target size for resizing
        
    Returns:
        Processed mask as PIL Image
    """
    # Convert to PIL if needed
    if isinstance(mask, np.ndarray):
        mask = numpy_to_pil(mask)
    
    # Convert to grayscale
    mask = mask.convert('L')
    
    # Resize if needed
    if target_size:
        mask = mask.resize(target_size, Image.Resampling.NEAREST)
    
    return mask

def check_device_availability() -> str:
    """
    Check available compute device.
    
    Returns:
        Device string ('cuda' or 'cpu')
    """
    if torch.cuda.is_available():
        device = 'cuda'
        logger.info(f"CUDA available - GPU: {torch.cuda.get_device_name()}")
    else:
        device = 'cpu'
        logger.info("Using CPU device")
    
    return device

def get_memory_usage() -> dict:
    """
    Get current memory usage statistics.
    
    Returns:
        Dictionary with memory statistics
    """
    stats = {}
    
    if torch.cuda.is_available():
        stats['gpu_allocated'] = torch.cuda.memory_allocated() / 1024**3  # GB
        stats['gpu_reserved'] = torch.cuda.memory_reserved() / 1024**3   # GB
        stats['gpu_max_allocated'] = torch.cuda.max_memory_allocated() / 1024**3  # GB
    
    # Could add system memory stats here if needed
    return stats

def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU memory cache cleared")

def validate_file_size(file_path: Union[str, Path], max_size_mb: float = 10.0) -> bool:
    """
    Validate file size.
    
    Args:
        file_path: Path to file
        max_size_mb: Maximum size in MB
        
    Returns:
        True if file size is valid
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return False
        
        size_mb = path.stat().st_size / (1024 * 1024)
        return size_mb <= max_size_mb
        
    except Exception as e:
        logger.error(f"Error checking file size: {e}")
        return False

def create_output_filename(base_name: str, suffix: str = "", extension: str = "png") -> str:
    """
    Create unique output filename with timestamp.
    
    Args:
        base_name: Base filename
        suffix: Optional suffix
        extension: File extension
        
    Returns:
        Unique filename
    """
    timestamp = int(time.time())
    if suffix:
        return f"{base_name}_{suffix}_{timestamp}.{extension}"
    else:
        return f"{base_name}_{timestamp}.{extension}"

def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if needed.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
