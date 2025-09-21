import torch
from PIL import Image
from typing import Optional, Union, Tuple
import logging

from ..utils.config import get_config
from ..utils.helpers import safe_execute, timer, validate_image, resize_image, prepare_mask

logger = logging.getLogger('oracle.diffusion')

class DiffusionInpainter:
    """Stable Diffusion inpainting pipeline wrapper."""
    
    def __init__(self, config_override: Optional[dict] = None):
        """
        Initialize diffusion pipeline.
        
        Args:
            config_override: Optional configuration overrides
        """
        self.config = get_config()
        if config_override:
            self.config.update(config_override)
        
        self.diffusion_config = self.config.diffusion_config
        self.pipeline = None
        
        self._load_pipeline()
    
    def _load_pipeline(self):
        """Load Stable Diffusion inpainting pipeline."""
        try:
            from diffusers import StableDiffusionInpaintPipeline
            
            model_name = self.diffusion_config.get('model_name', 'stabilityai/stable-diffusion-2-inpainting')
            device = self.diffusion_config.get('device', 'cpu')
            
            logger.info(f"Loading Stable Diffusion model: {model_name}")
            
            # Load pipeline
            self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
                safety_checker=None,  # Disable safety checker for faster inference
                requires_safety_checker=False
            )
            
            # Move to device
            self.pipeline = self.pipeline.to(device)
            
            # Enable memory efficient attention if configured
            if self.config.get('performance.enable_memory_efficient_attention', True):
                try:
                    self.pipeline.enable_attention_slicing()
                    logger.info("Enabled attention slicing for memory efficiency")
                except:
                    logger.warning("Could not enable attention slicing")
            
            # Enable CPU offload if configured
            if self.config.get('performance.enable_cpu_offload', False) and device == 'cuda':
                try:
                    self.pipeline.enable_model_cpu_offload()
                    logger.info("Enabled model CPU offload")
                except:
                    logger.warning("Could not enable CPU offload")
            
            logger.info("Stable Diffusion pipeline loaded successfully")
            
        except ImportError as e:
            logger.error(f"Failed to import diffusers: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load Stable Diffusion pipeline: {e}")
            raise
    
    @timer
    @safe_execute
    def inpaint(self,
                prompt: str,
                image: Union[str, Image.Image],
                mask: Union[str, Image.Image],
                negative_prompt: Optional[str] = None,
                num_inference_steps: Optional[int] = None,
                guidance_scale: Optional[float] = None,
                strength: Optional[float] = None,
                target_size: Optional[Tuple[int, int]] = None) -> Optional[Image.Image]:
        """
        Perform inpainting on image.
        
        Args:
            prompt: Text prompt for inpainting
            image: Input image
            mask: Inpainting mask
            negative_prompt: Negative prompt (what to avoid)
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for prompt adherence
            strength: Strength of inpainting
            target_size: Target size for processing
            
        Returns:
            Inpainted image or None if failed
        """
        if self.pipeline is None:
            logger.error("Diffusion pipeline not initialized")
            return None
        
        # Validate inputs
        if not prompt or not prompt.strip():
            logger.error("Empty prompt provided")
            return None
        
        # Validate and convert image
        pil_image = validate_image(image)
        if pil_image is None:
            logger.error("Invalid input image")
            return None
        
        # Validate and convert mask
        pil_mask = validate_image(mask)
        if pil_mask is None:
            logger.error("Invalid mask image")
            return None
        
        # Get processing parameters
        if target_size is None:
            target_size = tuple(self.config.get('image.default_size', [512, 512]))
        
        if num_inference_steps is None:
            num_inference_steps = self.config.get('processing.diffusion.num_inference_steps', 20)
        
        if guidance_scale is None:
            guidance_scale = self.config.get('processing.diffusion.guidance_scale', 7.5)
        
        if strength is None:
            strength = self.config.get('processing.diffusion.strength', 0.8)
        
        try:
            # Resize images to target size
            pil_image = resize_image(pil_image, target_size, maintain_aspect=False)
            pil_mask = prepare_mask(pil_mask, target_size)
            
            logger.info(f"Starting inpainting: '{prompt}' - Steps: {num_inference_steps}, "
                       f"Guidance: {guidance_scale}, Strength: {strength}")
            
            # Perform inpainting
            result = self.pipeline(
                prompt=prompt,
                image=pil_image,
                mask_image=pil_mask,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=strength,
                generator=torch.Generator().manual_seed(42)  # For reproducible results
            )
            
            inpainted_image = result.images[0]
            
            logger.info("Inpainting completed successfully")
            return inpainted_image
            
        except Exception as e:
            logger.error(f"Inpainting failed: {e}")
            return None
    
    @safe_execute
    def inpaint_simple(self,
                      prompt: str,
                      image: Union[str, Image.Image],
                      mask: Union[str, Image.Image]) -> Optional[Image.Image]:
        """
        Simple inpainting with default parameters (convenience method).
        
        Args:
            prompt: Text prompt for inpainting
            image: Input image
            mask: Inpainting mask
            
        Returns:
            Inpainted image or None if failed
        """
        return self.inpaint(prompt, image, mask)
    
    def get_pipeline_info(self) -> dict:
        """
        Get information about loaded pipeline.
        
        Returns:
            Dictionary with pipeline information
        """
        return {
            'model_name': self.diffusion_config.get('model_name'),
            'device': self.diffusion_config.get('device'),
            'is_loaded': self.pipeline is not None,
            'default_steps': self.config.get('processing.diffusion.num_inference_steps', 20),
            'default_guidance': self.config.get('processing.diffusion.guidance_scale', 7.5),
            'default_strength': self.config.get('processing.diffusion.strength', 0.8)
        }
    
    def clear_memory(self):
        """Clear GPU memory if using CUDA."""
        if self.pipeline is not None and self.diffusion_config.get('device') == 'cuda':
            try:
                torch.cuda.empty_cache()
                logger.info("Cleared GPU memory cache")
            except:
                logger.warning("Could not clear GPU memory cache")
