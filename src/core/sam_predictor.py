"""SAM (Segment Anything Model) predictor module."""

import numpy as np
import torch
from PIL import Image
from typing import List, Tuple, Optional, Union
import logging

from ..utils.config import get_config
from ..utils.helpers import safe_execute, timer, validate_image, pil_to_numpy, numpy_to_pil

logger = logging.getLogger('oracle.sam')

class SAMPredictor:
    """Segment Anything Model predictor wrapper."""
    
    def __init__(self, config_override: Optional[dict] = None):
        """
        Initialize SAM predictor.
        
        Args:
            config_override: Optional configuration overrides
        """
        self.config = get_config()
        if config_override:
            self.config.update(config_override)
        
        self.sam_config = self.config.sam_config
        self.predictor = None
        self.selected_pixels = []
        
        self._load_model()
    
    def _load_model(self):
        """Load SAM model and predictor."""
        try:
            from segment_anything import SamPredictor, sam_model_registry
            
            model_type = self.sam_config.get('model_type', 'vit_h')
            checkpoint_path = self.sam_config.get('checkpoint_path', './weights/sam_vit_h_4b8939.pth')
            device = self.sam_config.get('device', 'cpu')
            
            logger.info(f"Loading SAM model: {model_type} from {checkpoint_path}")
            
            # Load model
            sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
            sam.to(device=device)
            
            # Create predictor
            self.predictor = SamPredictor(sam_model=sam)
            
            logger.info("SAM model loaded successfully")
            
        except ImportError as e:
            logger.error(f"Failed to import segment_anything: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load SAM model: {e}")
            raise
    
    @timer
    @safe_execute
    def set_image(self, image: Union[str, Image.Image, np.ndarray]) -> bool:
        """
        Set image for SAM predictor.
        
        Args:
            image: Input image
            
        Returns:
            True if successful
        """
        if self.predictor is None:
            logger.error("SAM predictor not initialized")
            return False
        
        # Validate and convert image
        pil_image = validate_image(image)
        if pil_image is None:
            return False
        
        # Convert to numpy array
        image_array = pil_to_numpy(pil_image)
        
        # Set image in predictor
        self.predictor.set_image(image=image_array)
        
        # Reset selected pixels
        self.selected_pixels = []
        
        logger.info(f"Image set for SAM predictor: {image_array.shape}")
        return True
    
    def add_point(self, x: int, y: int) -> None:
        """
        Add point for segmentation.
        
        Args:
            x: X coordinate
            y: Y coordinate
        """
        self.selected_pixels.append([x, y])
        logger.debug(f"Added point: ({x}, {y})")
    
    def clear_points(self) -> None:
        """Clear all selected points."""
        self.selected_pixels = []
        logger.debug("Cleared all points")
    
    @timer
    @safe_execute
    def predict_mask(self, 
                    points: Optional[List[Tuple[int, int]]] = None,
                    multimask_output: Optional[bool] = None) -> Optional[Image.Image]:
        """
        Predict segmentation mask.
        
        Args:
            points: List of (x, y) points. Uses selected_pixels if None.
            multimask_output: Whether to output multiple masks
            
        Returns:
            Segmentation mask as PIL Image or None if failed
        """
        if self.predictor is None:
            logger.error("SAM predictor not initialized")
            return None
        
        # Use provided points or selected pixels
        if points is not None:
            input_points = np.array(points)
        elif self.selected_pixels:
            input_points = np.array(self.selected_pixels)
        else:
            logger.warning("No points provided for segmentation")
            return None
        
        # Create labels (all positive points)
        input_labels = np.ones(shape=(input_points.shape[0]))
        
        # Get multimask setting
        if multimask_output is None:
            multimask_output = self.config.get('processing.sam.multimask_output', False)
        
        try:
            # Predict masks
            masks, scores, logits = self.predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=multimask_output
            )
            
            # Select best mask (first one if single mask, highest score if multiple)
            if multimask_output and len(masks) > 1:
                best_idx = np.argmax(scores)
                mask = masks[best_idx]
                logger.info(f"Selected mask {best_idx} with score {scores[best_idx]:.3f}")
            else:
                mask = masks[0]
            
            # Convert to PIL Image
            mask_image = numpy_to_pil(mask.astype(np.uint8) * 255)
            
            logger.info(f"Generated mask from {len(input_points)} points")
            return mask_image
            
        except Exception as e:
            logger.error(f"Mask prediction failed: {e}")
            return None
    
    @safe_execute
    def predict_from_click(self, image: Union[str, Image.Image, np.ndarray], 
                          x: int, y: int) -> Optional[Image.Image]:
        """
        Predict mask from single click (convenience method).
        
        Args:
            image: Input image
            x: X coordinate of click
            y: Y coordinate of click
            
        Returns:
            Segmentation mask as PIL Image or None if failed
        """
        # Set image
        if not self.set_image(image):
            return None
        
        # Predict mask from single point
        return self.predict_mask(points=[(x, y)])
    
    def get_model_info(self) -> dict:
        """
        Get information about loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_type': self.sam_config.get('model_type'),
            'checkpoint_path': self.sam_config.get('checkpoint_path'),
            'device': self.sam_config.get('device'),
            'is_loaded': self.predictor is not None,
            'selected_points': len(self.selected_pixels)
        }
