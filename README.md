# 🎨 Oracle: SAM + Stable Diffusion

**Interactive image inpainting combining Segment Anything Model (SAM) with Stable Diffusion for powerful AI-driven image editing.**

<div align="center">

![Oracle Demo](./output.jpg)

*Oracle enables users to create realistic image edits with a single click*

</div>

## ✨ Features

- **🖱️ Click-to-Segment**: Interactive pixel selection with SAM
- **🎨 AI Inpainting**: Text-guided image generation with Stable Diffusion  
- **🌐 Web Interface**: User-friendly Gradio web app
- **⚡ CLI Tools**: Command-line interface for batch processing
- **⚙️ Configurable**: YAML-based configuration system
- **🔧 Modular**: Clean, maintainable code architecture
- **📊 Logging**: Comprehensive logging and error handling

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/angadbawa/Oracle.git
cd Oracle

# Install dependencies
pip install -r requirements.txt

# Download SAM weights (if not already present)
# Place sam_vit_h_4b8939.pth in ./weights/ directory
```

### 2. Launch Web Interface

```bash
# Start Gradio web app
python main.py web

# Custom host/port
python main.py web --host 0.0.0.0 --port 8080

# Create public link
python main.py web --share
```

### 3. Command Line Usage

```bash
# Inpaint from click coordinates
python main.py cli inpaint-click image.jpg 100 150 "a beautiful red flower"

# Inpaint with existing mask
python main.py cli inpaint-mask image.jpg mask.jpg "a sunset sky"

# Generate mask only
python main.py cli mask-only image.jpg "100,150,200,250"

# Show system information
python main.py info
```

## 🏗️ Architecture

### **Modular Structure**
```
Oracle/
├── src/
│   ├── core/                     # Core AI modules
│   │   ├── sam_predictor.py      # SAM model wrapper
│   │   ├── diffusion_pipeline.py # Stable Diffusion wrapper
│   │   └── image_processor.py    # Main orchestrator
│   ├── ui/                       # User interfaces
│   │   ├── gradio_app.py         # Web interface
│   │   └── cli_app.py            # Command line interface
│   ├── utils/                    # Utilities
│   │   ├── config.py             # Configuration management
│   │   ├── logger.py             # Logging setup
│   │   └── helpers.py            # Helper functions
│   └── config/
│       └── default.yaml          # Configuration file
├── main.py                       # Unified entry point
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

### **Key Components**

- **`SAMPredictor`**: Segment Anything Model wrapper for interactive segmentation
- **`DiffusionInpainter`**: Stable Diffusion pipeline for text-guided inpainting
- **`ImageProcessor`**: Main orchestrator combining SAM + Diffusion
- **`OracleGradioApp`**: Web interface with interactive UI
- **`OracleCLI`**: Command-line interface for automation

## ⚙️ Configuration

Oracle uses YAML configuration files for easy customization:

```yaml
# src/config/default.yaml
models:
  sam:
    model_type: "vit_h"
    checkpoint_path: "./weights/sam_vit_h_4b8939.pth"
    device: "cpu"  # or "cuda"
  
  stable_diffusion:
    model_name: "stabilityai/stable-diffusion-2-inpainting"
    device: "cpu"  # or "cuda"

image:
  default_size: [512, 512]
  max_file_size_mb: 10

processing:
  diffusion:
    num_inference_steps: 20
    guidance_scale: 7.5
    strength: 0.8
```

### **Custom Configuration**

```bash
# Use custom config file
python main.py web --config my_config.yaml

# Override device settings
python main.py web --device cuda
```

## 🖥️ Web Interface

The Gradio web interface provides an intuitive way to use Oracle:

1. **📸 Upload Image**: Load your image
2. **🖱️ Click to Segment**: Click on areas to segment with SAM
3. **✨ Enter Prompt**: Describe what you want to generate
4. **🎨 Generate**: Create AI-powered inpainting results

### **Features**
- Interactive point selection
- Real-time mask generation
- Advanced parameter controls
- System information display
- Example prompts

## 💻 CLI Interface

Powerful command-line tools for automation and batch processing:

### **Inpaint from Click**
```bash
python main.py cli inpaint-click \
  image.jpg 150 200 \
  "a beautiful sunset" \
  --negative "blurry, low quality" \
  --steps 30 \
  --guidance 8.0 \
  --output-dir ./results
```

### **Inpaint with Mask**
```bash
python main.py cli inpaint-mask \
  image.jpg mask.jpg \
  "a field of flowers" \
  --output-dir ./results
```

### **Generate Mask Only**
```bash
python main.py cli mask-only \
  image.jpg "100,150,200,250,300,350" \
  --output-dir ./masks
```

## 🔧 Advanced Usage

### **Python API**

```python
from src import ImageProcessor

# Initialize processor
processor = ImageProcessor()

# Process image with click coordinates
results = processor.process_click_to_inpaint(
    image="image.jpg",
    x=150, y=200,
    prompt="a beautiful garden",
    save_output=True
)

# Access results
original = results['original_image']
mask = results['mask'] 
inpainted = results['inpainted_image']
```

### **Custom Configuration**

```python
from src.utils.config import get_config

# Load custom config
config = get_config("custom_config.yaml")

# Override settings
config.set('models.sam.device', 'cuda')
config.set('processing.diffusion.num_inference_steps', 30)
```

## 📊 System Requirements

### **Minimum Requirements**
- Python 3.8+
- 8GB RAM
- 2GB disk space

### **Recommended**
- Python 3.9+
- 16GB+ RAM
- NVIDIA GPU with 8GB+ VRAM
- CUDA 11.8+

### **GPU Acceleration**

For faster inference, install CUDA-enabled PyTorch:

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Update config to use GPU
python main.py web --device cuda
```

## 🐛 Troubleshooting

### **Common Issues**

1. **SAM weights not found**
   ```bash
   # Download SAM weights
   wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
   mkdir -p weights && mv sam_vit_h_4b8939.pth weights/
   ```

2. **Out of memory errors**
   ```yaml
   # Enable memory optimizations in config
   performance:
     enable_memory_efficient_attention: true
     enable_cpu_offload: true
   ```

3. **Slow inference**
   ```bash
   # Use GPU acceleration
   python main.py web --device cuda
   
   # Reduce inference steps
   python main.py cli inpaint-click image.jpg 100 150 "prompt" --steps 10
   ```

### **Logging**

Enable debug logging for troubleshooting:

```bash
python main.py web --log-level DEBUG
```

## 🚀 Usage Examples

### **Web Interface**
```bash
# Basic launch
python main.py web

# Advanced launch with custom settings
python main.py web --host 0.0.0.0 --port 8080 --device cuda --share
```

### **CLI Operations**
```bash
# System information
python main.py info

# Batch processing with advanced settings
python main.py cli inpaint-click \
  input.jpg 200 300 "vibrant flowers in a garden" \
  --negative "blurry, low quality, distorted" \
  --steps 25 --guidance 7.5 \
  --output-dir ./batch_results
```