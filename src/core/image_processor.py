from PIL import Image
from typing import Optional, Union, Tuple, List
import logging

from .sam_predictor import SAMPredictor
from .diffusion_pipeline import DiffusionInpainter
from ..utils.config import get_config
from ..utils.helpers import safe_execute, timer, validate_image, create_output_filename, ensure_directory

logger = logging.getLogger('oracle.processor')

class ImageProcessor:
    """Main image processing orchestrator combining SAM and Stable Diffusion."""
    
    def __init__(self, config_override: Optional[dict] = None):
        """
        Initialize image processor.
        
        Args:
            config_override: Optional configuration overrides
        """
        self.config = get_config()
        if config_override:
            self.config.update(config_override)
        
        # Initialize components
        self.sam_predictor = SAMPredictor(config_override)
        self.diffusion_inpainter = DiffusionInpainter(config_override)
        
        logger.info("Image processor initialized")
    
    @timer
    @safe_execute
    def process_click_to_inpaint(self,
                                image: Union[str, Image.Image],
                                x: int,
                                y: int,
                                prompt: str,
                                negative_prompt: Optional[str] = None,
                                save_output: bool = True,
                                output_dir: Optional[str] = None) -> Optional[dict]:
        """
        Complete pipeline: click to generate mask, then inpaint.
        
        Args:
            image: Input image
            x: X coordinate of click
            y: Y coordinate of click
            prompt: Inpainting prompt
            negative_prompt: Negative prompt
            save_output: Whether to save output images
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary with results or None if failed
        """
        logger.info(f"Starting click-to-inpaint pipeline: ({x}, {y}) - '{prompt}'")
        
        # Validate image
        pil_image = validate_image(image)
        if pil_image is None:
            logger.error("Invalid input image")
            return None
        
        # Step 1: Generate mask from click
        mask = self.sam_predictor.predict_from_click(pil_image, x, y)
        if mask is None:
            logger.error("Failed to generate mask from click")
            return None
        
        logger.info("Mask generated successfully")
        
        # Step 2: Perform inpainting
        inpainted_image = self.diffusion_inpainter.inpaint_simple(prompt, pil_image, mask)
        if inpainted_image is None:
            logger.error("Failed to perform inpainting")
            return None
        
        logger.info("Inpainting completed successfully")
        
        # Prepare results
        results = {
            'original_image': pil_image,
            'mask': mask,
            'inpainted_image': inpainted_image,
            'click_coordinates': (x, y),
            'prompt': prompt,
            'negative_prompt': negative_prompt
        }
        
        # Save outputs if requested
        if save_output:
            saved_paths = self._save_results(results, output_dir)
            results['saved_paths'] = saved_paths
        
        return results
    
    @safe_execute
    def process_mask_to_inpaint(self,
                               image: Union[str, Image.Image],
                               mask: Union[str, Image.Image],
                               prompt: str,
                               negative_prompt: Optional[str] = None,
                               save_output: bool = True,
                               output_dir: Optional[str] = None) -> Optional[dict]:
        """
        Process with provided mask (skip SAM step).
        
        Args:
            image: Input image
            mask: Segmentation mask
            prompt: Inpainting prompt
            negative_prompt: Negative prompt
            save_output: Whether to save output images
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary with results or None if failed
        """
        logger.info(f"Starting mask-to-inpaint pipeline: '{prompt}'")
        
        # Validate inputs
        pil_image = validate_image(image)
        pil_mask = validate_image(mask)
        
        if pil_image is None or pil_mask is None:
            logger.error("Invalid input image or mask")
            return None
        
        # Perform inpainting
        inpainted_image = self.diffusion_inpainter.inpaint_simple(prompt, pil_image, pil_mask)
        if inpainted_image is None:
            logger.error("Failed to perform inpainting")
            return None
        
        logger.info("Inpainting completed successfully")
        
        # Prepare results
        results = {
            'original_image': pil_image,
            'mask': pil_mask,
            'inpainted_image': inpainted_image,
            'prompt': prompt,
            'negative_prompt': negative_prompt
        }
        
        # Save outputs if requested
        if save_output:
            saved_paths = self._save_results(results, output_dir)
            results['saved_paths'] = saved_paths
        
        return results
    
    @safe_execute
    def generate_mask_only(self,
                          image: Union[str, Image.Image],
                          points: List[Tuple[int, int]],
                          save_output: bool = True,
                          output_dir: Optional[str] = None) -> Optional[dict]:
        """
        Generate segmentation mask only.
        
        Args:
            image: Input image
            points: List of (x, y) points
            save_output: Whether to save output
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary with results or None if failed
        """
        logger.info(f"Generating mask from {len(points)} points")
        
        # Validate image
        pil_image = validate_image(image)
        if pil_image is None:
            logger.error("Invalid input image")
            return None
        
        # Set image and generate mask
        if not self.sam_predictor.set_image(pil_image):
            return None
        
        mask = self.sam_predictor.predict_mask(points=points)
        if mask is None:
            logger.error("Failed to generate mask")
            return None
        
        logger.info("Mask generated successfully")
        
        # Prepare results
        results = {
            'original_image': pil_image,
            'mask': mask,
            'points': points
        }
        
        # Save outputs if requested
        if save_output:
            saved_paths = self._save_mask_result(results, output_dir)
            results['saved_paths'] = saved_paths
        
        return results
    
    def _save_results(self, results: dict, output_dir: Optional[str] = None) -> dict:
        """
        Save processing results to files.
        
        Args:
            results: Results dictionary
            output_dir: Output directory
            
        Returns:
            Dictionary with saved file paths
        """
        if output_dir is None:
            output_dir = self.config.get('paths.output_dir', './outputs')
        
        output_path = ensure_directory(output_dir)
        
        saved_paths = {}
        
        try:
            # Save original image
            if 'original_image' in results:
                original_path = output_path / create_output_filename('original', extension='png')
                results['original_image'].save(original_path)
                saved_paths['original'] = str(original_path)
            
            # Save mask
            if 'mask' in results:
                mask_path = output_path / create_output_filename('mask', extension='png')
                results['mask'].save(mask_path)
                saved_paths['mask'] = str(mask_path)
            
            # Save inpainted image
            if 'inpainted_image' in results:
                inpainted_path = output_path / create_output_filename('inpainted', extension='png')
                results['inpainted_image'].save(inpainted_path)
                saved_paths['inpainted'] = str(inpainted_path)
            
            logger.info(f"Results saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
        
        return saved_paths
    
    def _save_mask_result(self, results: dict, output_dir: Optional[str] = None) -> dict:
        """
        Save mask-only results.
        
        Args:
            results: Results dictionary
            output_dir: Output directory
            
        Returns:
            Dictionary with saved file paths
        """
        if output_dir is None:
            output_dir = self.config.get('paths.output_dir', './outputs')
        
        output_path = ensure_directory(output_dir)
        
        saved_paths = {}
        
        try:
            # Save original image
            if 'original_image' in results:
                original_path = output_path / create_output_filename('original', extension='png')
                results['original_image'].save(original_path)
                saved_paths['original'] = str(original_path)
            
            # Save mask
            if 'mask' in results:
                mask_path = output_path / create_output_filename('mask', extension='png')
                results['mask'].save(mask_path)
                saved_paths['mask'] = str(mask_path)
            
            logger.info(f"Mask results saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save mask results: {e}")
        
        return saved_paths
    
    def get_system_info(self) -> dict:
        """
        Get information about all system components.
        
        Returns:
            Dictionary with system information
        """
        return {
            'sam': self.sam_predictor.get_model_info(),
            'diffusion': self.diffusion_inpainter.get_pipeline_info(),
            'config': {
                'image_size': self.config.get('image.default_size'),
                'output_dir': self.config.get('paths.output_dir')
            }
        }
    
    def clear_memory(self):
        """Clear memory caches."""
        self.diffusion_inpainter.clear_memory()
        logger.info("Memory caches cleared")
