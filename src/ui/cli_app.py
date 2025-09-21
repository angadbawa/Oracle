import argparse
import sys
from pathlib import Path
from typing import Optional
import logging

from ..core.image_processor import ImageProcessor
from ..utils.config import get_config
from ..utils.logger import setup_logging
from ..utils.helpers import validate_file_size

logger = logging.getLogger('oracle.cli')

class OracleCLI:
    """Command line interface for Oracle."""
    
    def __init__(self, config_override: Optional[dict] = None):
        """
        Initialize CLI.
        
        Args:
            config_override: Optional configuration overrides
        """
        self.config = get_config()
        if config_override:
            self.config.update(config_override)
        
        self.processor = ImageProcessor(config_override)
        
        logger.info("CLI initialized")
    
    def inpaint_from_click(self, 
                          image_path: str,
                          x: int,
                          y: int,
                          prompt: str,
                          negative_prompt: Optional[str] = None,
                          output_dir: Optional[str] = None,
                          steps: int = 20,
                          guidance: float = 7.5) -> bool:
        """
        Perform inpainting from click coordinates.
        
        Args:
            image_path: Path to input image
            x: X coordinate
            y: Y coordinate  
            prompt: Inpainting prompt
            negative_prompt: Negative prompt
            output_dir: Output directory
            steps: Number of inference steps
            guidance: Guidance scale
            
        Returns:
            True if successful
        """
        try:
            # Validate input file
            if not Path(image_path).exists():
                logger.error(f"Image file not found: {image_path}")
                return False
            
            max_size = self.config.get('image.max_file_size_mb', 10)
            if not validate_file_size(image_path, max_size):
                logger.error(f"Image file too large (max {max_size}MB)")
                return False
            
            logger.info(f"Processing image: {image_path}")
            logger.info(f"Click coordinates: ({x}, {y})")
            logger.info(f"Prompt: '{prompt}'")
            
            # Process image
            results = self.processor.process_click_to_inpaint(
                image=image_path,
                x=x,
                y=y,
                prompt=prompt,
                negative_prompt=negative_prompt,
                save_output=True,
                output_dir=output_dir
            )
            
            if results is None:
                logger.error("Processing failed")
                return False
            
            # Print results
            saved_paths = results.get('saved_paths', {})
            print(f"\n✅ Processing completed successfully!")
            print(f"📁 Output directory: {output_dir or self.config.get('paths.output_dir')}")
            
            if 'original' in saved_paths:
                print(f"📸 Original: {saved_paths['original']}")
            if 'mask' in saved_paths:
                print(f"🎭 Mask: {saved_paths['mask']}")
            if 'inpainted' in saved_paths:
                print(f"🎨 Result: {saved_paths['inpainted']}")
            
            return True
            
        except Exception as e:
            logger.error(f"CLI processing failed: {e}")
            return False
    
    def inpaint_from_mask(self,
                         image_path: str,
                         mask_path: str,
                         prompt: str,
                         negative_prompt: Optional[str] = None,
                         output_dir: Optional[str] = None,
                         steps: int = 20,
                         guidance: float = 7.5) -> bool:
        """
        Perform inpainting with provided mask.
        
        Args:
            image_path: Path to input image
            mask_path: Path to mask image
            prompt: Inpainting prompt
            negative_prompt: Negative prompt
            output_dir: Output directory
            steps: Number of inference steps
            guidance: Guidance scale
            
        Returns:
            True if successful
        """
        try:
            # Validate input files
            if not Path(image_path).exists():
                logger.error(f"Image file not found: {image_path}")
                return False
            
            if not Path(mask_path).exists():
                logger.error(f"Mask file not found: {mask_path}")
                return False
            
            logger.info(f"Processing image: {image_path}")
            logger.info(f"Using mask: {mask_path}")
            logger.info(f"Prompt: '{prompt}'")
            
            # Process image
            results = self.processor.process_mask_to_inpaint(
                image=image_path,
                mask=mask_path,
                prompt=prompt,
                negative_prompt=negative_prompt,
                save_output=True,
                output_dir=output_dir
            )
            
            if results is None:
                logger.error("Processing failed")
                return False
            
            # Print results
            saved_paths = results.get('saved_paths', {})
            print(f"\n✅ Processing completed successfully!")
            print(f"📁 Output directory: {output_dir or self.config.get('paths.output_dir')}")
            
            if 'original' in saved_paths:
                print(f"📸 Original: {saved_paths['original']}")
            if 'mask' in saved_paths:
                print(f"🎭 Mask: {saved_paths['mask']}")
            if 'inpainted' in saved_paths:
                print(f"🎨 Result: {saved_paths['inpainted']}")
            
            return True
            
        except Exception as e:
            logger.error(f"CLI processing failed: {e}")
            return False
    
    def generate_mask_only(self,
                          image_path: str,
                          coordinates: str,
                          output_dir: Optional[str] = None) -> bool:
        """
        Generate mask only from coordinates.
        
        Args:
            image_path: Path to input image
            coordinates: Comma-separated coordinates "x1,y1,x2,y2,..."
            output_dir: Output directory
            
        Returns:
            True if successful
        """
        try:
            # Validate input file
            if not Path(image_path).exists():
                logger.error(f"Image file not found: {image_path}")
                return False
            
            # Parse coordinates
            try:
                coords = [int(x) for x in coordinates.split(',')]
                if len(coords) % 2 != 0:
                    raise ValueError("Coordinates must be pairs")
                
                points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
                
            except ValueError as e:
                logger.error(f"Invalid coordinates format: {e}")
                return False
            
            logger.info(f"Processing image: {image_path}")
            logger.info(f"Points: {points}")
            
            # Generate mask
            results = self.processor.generate_mask_only(
                image=image_path,
                points=points,
                save_output=True,
                output_dir=output_dir
            )
            
            if results is None:
                logger.error("Mask generation failed")
                return False
            
            # Print results
            saved_paths = results.get('saved_paths', {})
            print(f"\n✅ Mask generated successfully!")
            print(f"📁 Output directory: {output_dir or self.config.get('paths.output_dir')}")
            
            if 'original' in saved_paths:
                print(f"📸 Original: {saved_paths['original']}")
            if 'mask' in saved_paths:
                print(f"🎭 Mask: {saved_paths['mask']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Mask generation failed: {e}")
            return False
    
    def show_system_info(self):
        """Display system information."""
        try:
            info = self.processor.get_system_info()
            
            print("\n🔧 Oracle System Information")
            print("=" * 40)
            
            # SAM info
            sam_info = info.get('sam', {})
            print(f"\n📸 SAM Model:")
            print(f"  Type: {sam_info.get('model_type', 'Unknown')}")
            print(f"  Checkpoint: {sam_info.get('checkpoint_path', 'Unknown')}")
            print(f"  Device: {sam_info.get('device', 'Unknown')}")
            print(f"  Loaded: {'✅' if sam_info.get('is_loaded') else '❌'}")
            
            # Diffusion info
            diff_info = info.get('diffusion', {})
            print(f"\n🎨 Stable Diffusion:")
            print(f"  Model: {diff_info.get('model_name', 'Unknown')}")
            print(f"  Device: {diff_info.get('device', 'Unknown')}")
            print(f"  Loaded: {'✅' if diff_info.get('is_loaded') else '❌'}")
            print(f"  Default Steps: {diff_info.get('default_steps', 20)}")
            print(f"  Default Guidance: {diff_info.get('default_guidance', 7.5)}")
            
            # Config info
            config_info = info.get('config', {})
            print(f"\n⚙️ Configuration:")
            print(f"  Image Size: {config_info.get('image_size', [512, 512])}")
            print(f"  Output Dir: {config_info.get('output_dir', './outputs')}")
            
            print()
            
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            print(f"❌ Error getting system information: {e}")

def setup_argument_parser() -> argparse.ArgumentParser:
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Oracle: SAM + Stable Diffusion CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inpaint from click coordinates
  python -m oracle.cli inpaint-click image.jpg 100 150 "a red flower"
  
  # Inpaint with existing mask
  python -m oracle.cli inpaint-mask image.jpg mask.jpg "a blue sky"
  
  # Generate mask only
  python -m oracle.cli mask-only image.jpg "100,150,200,250"
  
  # Show system information
  python -m oracle.cli info
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Inpaint from click command
    click_parser = subparsers.add_parser('inpaint-click', help='Inpaint from click coordinates')
    click_parser.add_argument('image', help='Input image path')
    click_parser.add_argument('x', type=int, help='X coordinate')
    click_parser.add_argument('y', type=int, help='Y coordinate')
    click_parser.add_argument('prompt', help='Inpainting prompt')
    click_parser.add_argument('--negative', help='Negative prompt')
    click_parser.add_argument('--output-dir', help='Output directory')
    click_parser.add_argument('--steps', type=int, default=20, help='Inference steps')
    click_parser.add_argument('--guidance', type=float, default=7.5, help='Guidance scale')
    
    # Inpaint from mask command
    mask_parser = subparsers.add_parser('inpaint-mask', help='Inpaint with provided mask')
    mask_parser.add_argument('image', help='Input image path')
    mask_parser.add_argument('mask', help='Mask image path')
    mask_parser.add_argument('prompt', help='Inpainting prompt')
    mask_parser.add_argument('--negative', help='Negative prompt')
    mask_parser.add_argument('--output-dir', help='Output directory')
    mask_parser.add_argument('--steps', type=int, default=20, help='Inference steps')
    mask_parser.add_argument('--guidance', type=float, default=7.5, help='Guidance scale')
    
    # Generate mask only command
    mask_only_parser = subparsers.add_parser('mask-only', help='Generate mask only')
    mask_only_parser.add_argument('image', help='Input image path')
    mask_only_parser.add_argument('coordinates', help='Comma-separated coordinates (x1,y1,x2,y2,...)')
    mask_only_parser.add_argument('--output-dir', help='Output directory')
    
    # System info command
    info_parser = subparsers.add_parser('info', help='Show system information')
    
    # Global options
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Logging level')
    
    return parser

def main():
    """Main CLI entry point."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    
    # Load configuration
    config_override = None
    if args.config:
        from ..utils.config import reload_config
        reload_config(args.config)
    
    # Initialize CLI
    try:
        cli = OracleCLI(config_override)
    except Exception as e:
        logger.error(f"Failed to initialize Oracle CLI: {e}")
        sys.exit(1)
    
    # Handle commands
    success = False
    
    if args.command == 'inpaint-click':
        success = cli.inpaint_from_click(
            image_path=args.image,
            x=args.x,
            y=args.y,
            prompt=args.prompt,
            negative_prompt=args.negative,
            output_dir=args.output_dir,
            steps=args.steps,
            guidance=args.guidance
        )
    
    elif args.command == 'inpaint-mask':
        success = cli.inpaint_from_mask(
            image_path=args.image,
            mask_path=args.mask,
            prompt=args.prompt,
            negative_prompt=args.negative,
            output_dir=args.output_dir,
            steps=args.steps,
            guidance=args.guidance
        )
    
    elif args.command == 'mask-only':
        success = cli.generate_mask_only(
            image_path=args.image,
            coordinates=args.coordinates,
            output_dir=args.output_dir
        )
    
    elif args.command == 'info':
        cli.show_system_info()
        success = True
    
    else:
        parser.print_help()
        success = False
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
