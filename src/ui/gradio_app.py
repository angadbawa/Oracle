import gradio as gr
import numpy as np
from PIL import Image
from typing import Optional, Tuple
import logging

from ..core.image_processor import ImageProcessor
from ..utils.config import get_config
from ..utils.logger import setup_logging

logger = logging.getLogger('oracle.ui')

class OracleGradioApp:
    """Gradio web application for Oracle."""
    
    def __init__(self, config_override: Optional[dict] = None):
        """
        Initialize Gradio app.
        
        Args:
            config_override: Optional configuration overrides
        """
        self.config = get_config()
        if config_override:
            self.config.update(config_override)
        
        self.ui_config = self.config.ui_config
        self.processor = ImageProcessor(config_override)
        
        # State variables
        self.current_image = None
        self.current_mask = None
        self.selected_pixels = []
        
        logger.info("Gradio app initialized")
    
    def on_image_select(self, image: Image.Image, evt: gr.SelectData) -> Optional[Image.Image]:
        """
        Handle image click event to generate mask.
        
        Args:
            image: Selected image
            evt: Selection event data
            
        Returns:
            Generated mask or None
        """
        if image is None:
            return None
        
        try:
            # Get click coordinates
            x, y = evt.index
            
            logger.info(f"Image clicked at: ({x}, {y})")
            
            # Store current image
            self.current_image = image
            
            # Generate mask using SAM
            if not self.processor.sam_predictor.set_image(image):
                logger.error("Failed to set image in SAM predictor")
                return None
            
            # Add point and predict mask
            self.processor.sam_predictor.add_point(x, y)
            mask = self.processor.sam_predictor.predict_mask()
            
            if mask is not None:
                self.current_mask = mask
                logger.info("Mask generated successfully")
                return mask
            else:
                logger.error("Failed to generate mask")
                return None
                
        except Exception as e:
            logger.error(f"Error in image selection: {e}")
            return None
    
    def on_inpaint_click(self, 
                        prompt: str, 
                        image: Optional[Image.Image], 
                        mask: Optional[Image.Image],
                        negative_prompt: str = "",
                        num_steps: int = 20,
                        guidance_scale: float = 7.5) -> Optional[Image.Image]:
        """
        Handle inpainting button click.
        
        Args:
            prompt: Inpainting prompt
            image: Input image
            mask: Segmentation mask
            negative_prompt: Negative prompt
            num_steps: Number of inference steps
            guidance_scale: Guidance scale
            
        Returns:
            Inpainted image or None
        """
        if not prompt or not prompt.strip():
            logger.warning("Empty prompt provided")
            return None
        
        if image is None:
            logger.warning("No image provided")
            return None
        
        if mask is None:
            logger.warning("No mask provided")
            return None
        
        try:
            logger.info(f"Starting inpainting with prompt: '{prompt}'")
            
            # Perform inpainting
            result = self.processor.diffusion_inpainter.inpaint(
                prompt=prompt,
                image=image,
                mask=mask,
                negative_prompt=negative_prompt if negative_prompt.strip() else None,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale
            )
            
            if result is not None:
                logger.info("Inpainting completed successfully")
                return result
            else:
                logger.error("Inpainting failed")
                return None
                
        except Exception as e:
            logger.error(f"Error in inpainting: {e}")
            return None
    
    def on_clear_points(self) -> Tuple[Optional[Image.Image], str]:
        """
        Clear selected points and reset mask.
        
        Returns:
            Tuple of (None, status message)
        """
        self.processor.sam_predictor.clear_points()
        self.selected_pixels = []
        self.current_mask = None
        
        logger.info("Points and mask cleared")
        return None, "Points cleared"
    
    def on_system_info(self) -> str:
        """
        Get system information.
        
        Returns:
            System information as formatted string
        """
        try:
            info = self.processor.get_system_info()
            
            info_text = "🔧 System Information:\n\n"
            
            # SAM info
            sam_info = info.get('sam', {})
            info_text += f"📸 SAM Model:\n"
            info_text += f"  • Type: {sam_info.get('model_type', 'Unknown')}\n"
            info_text += f"  • Device: {sam_info.get('device', 'Unknown')}\n"
            info_text += f"  • Loaded: {'✅' if sam_info.get('is_loaded') else '❌'}\n"
            info_text += f"  • Selected Points: {sam_info.get('selected_points', 0)}\n\n"
            
            # Diffusion info
            diff_info = info.get('diffusion', {})
            info_text += f"🎨 Stable Diffusion:\n"
            info_text += f"  • Model: {diff_info.get('model_name', 'Unknown')}\n"
            info_text += f"  • Device: {diff_info.get('device', 'Unknown')}\n"
            info_text += f"  • Loaded: {'✅' if diff_info.get('is_loaded') else '❌'}\n"
            info_text += f"  • Default Steps: {diff_info.get('default_steps', 20)}\n\n"
            
            # Config info
            config_info = info.get('config', {})
            info_text += f"⚙️ Configuration:\n"
            info_text += f"  • Image Size: {config_info.get('image_size', [512, 512])}\n"
            info_text += f"  • Output Dir: {config_info.get('output_dir', './outputs')}\n"
            
            return info_text
            
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            return f"Error getting system information: {e}"
    
    def create_interface(self) -> gr.Blocks:
        """
        Create Gradio interface.
        
        Returns:
            Gradio Blocks interface
        """
        # Get UI configuration
        title = self.ui_config.get('gradio', {}).get('title', 'Oracle: SAM + Stable Diffusion')
        description = self.ui_config.get('gradio', {}).get('description', 
                                                          'Interactive image editing with AI')
        
        with gr.Blocks(title=title, theme=gr.themes.Soft()) as interface:
            
            # Header
            gr.Markdown(f"# {title}")
            gr.Markdown(f"*{description}*")
            
            with gr.Row():
                # Left column - Input
                with gr.Column(scale=1):
                    gr.Markdown("## 📸 Input Image")
                    input_image = gr.Image(
                        type='pil',
                        label="Click on the image to select area for inpainting",
                        height=400
                    )
                    
                    with gr.Row():
                        clear_btn = gr.Button("🗑️ Clear Points", variant="secondary")
                        info_btn = gr.Button("ℹ️ System Info", variant="secondary")
                
                # Middle column - Mask
                with gr.Column(scale=1):
                    gr.Markdown("## 🎭 Generated Mask")
                    mask_image = gr.Image(
                        type='pil',
                        label="Segmentation mask (auto-generated)",
                        height=400,
                        interactive=False
                    )
                
                # Right column - Output
                with gr.Column(scale=1):
                    gr.Markdown("## 🎨 Inpainted Result")
                    output_image = gr.Image(
                        type='pil',
                        label="Inpainting result",
                        height=400,
                        interactive=False
                    )
            
            # Controls
            with gr.Row():
                with gr.Column():
                    prompt_input = gr.Textbox(
                        label="✨ Inpainting Prompt",
                        placeholder="Describe what you want to generate in the selected area...",
                        lines=2
                    )
                    
                    negative_prompt_input = gr.Textbox(
                        label="🚫 Negative Prompt (Optional)",
                        placeholder="Describe what you want to avoid...",
                        lines=1
                    )
            
            # Advanced settings
            with gr.Accordion("🔧 Advanced Settings", open=False):
                with gr.Row():
                    steps_slider = gr.Slider(
                        minimum=10,
                        maximum=50,
                        value=20,
                        step=1,
                        label="Inference Steps"
                    )
                    
                    guidance_slider = gr.Slider(
                        minimum=1.0,
                        maximum=20.0,
                        value=7.5,
                        step=0.5,
                        label="Guidance Scale"
                    )
            
            # Action button
            inpaint_btn = gr.Button("🚀 Generate Inpainting", variant="primary", size="lg")
            
            # Status and info
            with gr.Row():
                status_text = gr.Textbox(
                    label="📊 Status",
                    interactive=False,
                    lines=1
                )
            
            with gr.Accordion("📋 System Information", open=False):
                info_text = gr.Textbox(
                    label="System Info",
                    interactive=False,
                    lines=10
                )
            
            # Event handlers
            input_image.select(
                fn=self.on_image_select,
                inputs=[input_image],
                outputs=[mask_image]
            )
            
            inpaint_btn.click(
                fn=self.on_inpaint_click,
                inputs=[
                    prompt_input,
                    input_image,
                    mask_image,
                    negative_prompt_input,
                    steps_slider,
                    guidance_slider
                ],
                outputs=[output_image]
            )
            
            clear_btn.click(
                fn=self.on_clear_points,
                outputs=[mask_image, status_text]
            )
            
            info_btn.click(
                fn=self.on_system_info,
                outputs=[info_text]
            )
            
            # Examples
            with gr.Accordion("💡 Example Prompts", open=False):
                gr.Examples(
                    examples=[
                        ["a beautiful flower garden"],
                        ["a modern building with glass windows"],
                        ["a cat sitting on a chair"],
                        ["abstract colorful patterns"],
                        ["a sunset over mountains"],
                    ],
                    inputs=[prompt_input]
                )
        
        return interface
    
    def launch(self, 
               share: Optional[bool] = None,
               server_name: Optional[str] = None,
               server_port: Optional[int] = None,
               **kwargs) -> None:
        """
        Launch Gradio interface.
        
        Args:
            share: Whether to create public link
            server_name: Server hostname
            server_port: Server port
            **kwargs: Additional Gradio launch arguments
        """
        # Get launch configuration
        gradio_config = self.ui_config.get('gradio', {})
        
        if share is None:
            share = gradio_config.get('share', False)
        
        if server_name is None:
            server_name = gradio_config.get('server_name', '127.0.0.1')
        
        if server_port is None:
            server_port = gradio_config.get('server_port', 7860)
        
        # Create and launch interface
        interface = self.create_interface()
        
        logger.info(f"Launching Gradio app on {server_name}:{server_port}")
        
        interface.launch(
            share=share,
            server_name=server_name,
            server_port=server_port,
            **kwargs
        )
