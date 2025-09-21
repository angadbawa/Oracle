#!/usr/bin/env python3
"""
Oracle: SAM + Stable Diffusion - Unified Entry Point

Interactive image inpainting combining Segment Anything Model (SAM) 
with Stable Diffusion for powerful image editing capabilities.

Usage:
    # Launch Gradio web interface
    python main.py web
    
    # Command line inpainting from click
    python main.py cli inpaint-click image.jpg 100 150 "a red flower"
    
    # Command line inpainting with mask
    python main.py cli inpaint-mask image.jpg mask.jpg "a blue sky"
    
    # Generate mask only
    python main.py cli mask-only image.jpg "100,150,200,250"
    
    # Show system information
    python main.py info
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def setup_argument_parser() -> argparse.ArgumentParser:
    """Setup unified command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Oracle: SAM + Stable Diffusion - Interactive Image Inpainting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🎨 Oracle combines Segment Anything Model (SAM) with Stable Diffusion
   for powerful interactive image inpainting capabilities.

Examples:
  # Launch web interface
  python main.py web
  
  # CLI inpainting from click coordinates  
  python main.py cli inpaint-click image.jpg 100 150 "a beautiful flower"
  
  # CLI inpainting with existing mask
  python main.py cli inpaint-mask image.jpg mask.jpg "a sunset sky"
  
  # Generate segmentation mask only
  python main.py cli mask-only image.jpg "100,150,200,250"
  
  # Show system information
  python main.py info

🔧 Configuration:
  Use --config to specify custom configuration file.
  Default configuration is loaded from config/default.yaml
        """
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='Interface mode')
    
    # ===== WEB INTERFACE =====
    web_parser = subparsers.add_parser('web', help='Launch Gradio web interface')
    web_parser.add_argument('--port', type=int, default=7860, help='Server port')
    web_parser.add_argument('--host', default='127.0.0.1', help='Server host')
    web_parser.add_argument('--share', action='store_true', help='Create public Gradio link')
    web_parser.add_argument('--no-browser', action='store_true', help='Don\'t open browser automatically')
    
    # ===== CLI INTERFACE =====
    cli_parser = subparsers.add_parser('cli', help='Command line interface')
    cli_subparsers = cli_parser.add_subparsers(dest='cli_command', help='CLI commands')
    
    # CLI: Inpaint from click
    click_parser = cli_subparsers.add_parser('inpaint-click', help='Inpaint from click coordinates')
    click_parser.add_argument('image', help='Input image path')
    click_parser.add_argument('x', type=int, help='X coordinate')
    click_parser.add_argument('y', type=int, help='Y coordinate')
    click_parser.add_argument('prompt', help='Inpainting prompt')
    click_parser.add_argument('--negative', help='Negative prompt')
    click_parser.add_argument('--output-dir', help='Output directory')
    click_parser.add_argument('--steps', type=int, default=20, help='Inference steps (10-50)')
    click_parser.add_argument('--guidance', type=float, default=7.5, help='Guidance scale (1.0-20.0)')
    
    # CLI: Inpaint from mask
    mask_parser = cli_subparsers.add_parser('inpaint-mask', help='Inpaint with provided mask')
    mask_parser.add_argument('image', help='Input image path')
    mask_parser.add_argument('mask', help='Mask image path')
    mask_parser.add_argument('prompt', help='Inpainting prompt')
    mask_parser.add_argument('--negative', help='Negative prompt')
    mask_parser.add_argument('--output-dir', help='Output directory')
    mask_parser.add_argument('--steps', type=int, default=20, help='Inference steps (10-50)')
    mask_parser.add_argument('--guidance', type=float, default=7.5, help='Guidance scale (1.0-20.0)')
    
    # CLI: Generate mask only
    mask_only_parser = cli_subparsers.add_parser('mask-only', help='Generate segmentation mask only')
    mask_only_parser.add_argument('image', help='Input image path')
    mask_only_parser.add_argument('coordinates', help='Comma-separated coordinates (x1,y1,x2,y2,...)')
    mask_only_parser.add_argument('--output-dir', help='Output directory')
    
    # ===== SYSTEM INFO =====
    info_parser = subparsers.add_parser('info', help='Show system information')
    
    # ===== GLOBAL OPTIONS =====
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Logging level')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto',
                       help='Compute device (auto-detects by default)')
    
    return parser

def handle_web_interface(args):
    """Handle web interface mode."""
    try:
        from src.ui.gradio_app import OracleGradioApp
        from src.utils.logger import setup_logging
        
        # Setup logging
        setup_logging(level=args.log_level)
        
        # Create configuration override if needed
        config_override = {}
        if args.device != 'auto':
            config_override = {
                'models': {
                    'sam': {'device': args.device},
                    'stable_diffusion': {'device': args.device}
                }
            }
        
        # Initialize and launch app
        app = OracleGradioApp(config_override if config_override else None)
        
        print(f"🚀 Launching Oracle web interface...")
        print(f"📍 Server: http://{args.host}:{args.port}")
        print(f"🔧 Device: {args.device}")
        print(f"📊 Log Level: {args.log_level}")
        
        app.launch(
            share=args.share,
            server_name=args.host,
            server_port=args.port,
            inbrowser=not args.no_browser
        )
        
        return True
        
    except KeyboardInterrupt:
        print("\n👋 Oracle web interface stopped")
        return True
    except Exception as e:
        print(f"❌ Failed to launch web interface: {e}")
        return False

def handle_cli_interface(args):
    """Handle CLI interface mode."""
    try:
        from src.ui.cli_app import OracleCLI
        from src.utils.logger import setup_logging
        
        # Setup logging
        setup_logging(level=args.log_level)
        
        # Create configuration override if needed
        config_override = {}
        if args.device != 'auto':
            config_override = {
                'models': {
                    'sam': {'device': args.device},
                    'stable_diffusion': {'device': args.device}
                }
            }
        
        # Initialize CLI
        cli = OracleCLI(config_override if config_override else None)
        
        # Handle CLI commands
        if args.cli_command == 'inpaint-click':
            return cli.inpaint_from_click(
                image_path=args.image,
                x=args.x,
                y=args.y,
                prompt=args.prompt,
                negative_prompt=args.negative,
                output_dir=args.output_dir,
                steps=args.steps,
                guidance=args.guidance
            )
        
        elif args.cli_command == 'inpaint-mask':
            return cli.inpaint_from_mask(
                image_path=args.image,
                mask_path=args.mask,
                prompt=args.prompt,
                negative_prompt=args.negative,
                output_dir=args.output_dir,
                steps=args.steps,
                guidance=args.guidance
            )
        
        elif args.cli_command == 'mask-only':
            return cli.generate_mask_only(
                image_path=args.image,
                coordinates=args.coordinates,
                output_dir=args.output_dir
            )
        
        else:
            print("❌ Unknown CLI command")
            return False
            
    except Exception as e:
        print(f"❌ CLI error: {e}")
        return False

def handle_system_info(args):
    """Handle system info display."""
    try:
        from src.ui.cli_app import OracleCLI
        from src.utils.logger import setup_logging
        
        # Setup logging (quiet for info display)
        setup_logging(level='WARNING')
        
        # Create configuration override if needed
        config_override = {}
        if args.device != 'auto':
            config_override = {
                'models': {
                    'sam': {'device': args.device},
                    'stable_diffusion': {'device': args.device}
                }
            }
        
        # Initialize CLI and show info
        cli = OracleCLI(config_override if config_override else None)
        cli.show_system_info()
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to get system info: {e}")
        return False

def main():
    """Main entry point."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Handle configuration file
    if args.config:
        try:
            from src.utils.config import reload_config
            reload_config(args.config)
            print(f"📄 Loaded configuration: {args.config}")
        except Exception as e:
            print(f"⚠️  Failed to load config file: {e}")
    
    # Handle different modes
    success = False
    
    if args.mode == 'web':
        success = handle_web_interface(args)
    
    elif args.mode == 'cli':
        success = handle_cli_interface(args)
    
    elif args.mode == 'info':
        success = handle_system_info(args)
    
    else:
        # No mode specified, show help
        print("🎨 Oracle: SAM + Stable Diffusion")
        print("Interactive image inpainting with AI")
        print()
        parser.print_help()
        print()
        print("💡 Quick start:")
        print("   python main.py web              # Launch web interface")
        print("   python main.py info             # Show system info")
        print("   python main.py cli --help       # CLI help")
        success = False
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
