# **Qwen Image LoRA DLC**

A powerful Gradio application for generating images using the Qwen-Image model with various LoRA (Low-Rank Adaptation) styles. This tool provides both speed-optimized and quality-focused generation modes with a curated collection of artistic styles.

https://github.com/user-attachments/assets/ae5e8d2e-6ff3-40e8-9a52-f650edb6f370

### Image Examples

| ![Image 1](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/SyJm8LHzDJvpKWiqUOUeD.png) | ![Image 2](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/iELVSR9rCMsvATP9ivAil.png) |
|------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| ![Image 3](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/UPeWsiPLYg9nWt0Lks4j2.png) | ![Image 4](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/sLbC26jv9lQqpTtTnWmfO.png) |
| ![Image 1](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/SyJm8LHzDJvpKWiqUOUeD.png) | ![Image 2](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/iELVSR9rCMsvATP9ivAil.png) |
|------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| ![Image 3](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/UPeWsiPLYg9nWt0Lks4j2.png) | ![Image 4](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/sLbC26jv9lQqpTtTnWmfO.png) |


## Features

### Core Functionality
- **Dual Generation Modes**:
  - Fast Mode: 8 steps with Lightning LoRA for quick results
  - Base Mode: 48 steps for maximum quality output
- **Multiple Aspect Ratios**: Support for 1:1, 16:9, 9:16, 4:3, 3:4, 3:2, and 2:3 ratios
- **Custom LoRA Support**: Load any Qwen-Image compatible LoRA from Hugging Face
- **Advanced Controls**: Fine-tune generation with guidance scale, steps, seed, and LoRA scaling

> Links
> - **GitHub Repository**: https://github.com/PRITHIVSAKTHIUR/Qwen-Image-LoRA-DLC
> - **Hugging Face Space**: https://huggingface.co/spaces/prithivMLmods/Qwen-Image-LoRA-DLC


### Pre-loaded LoRA Styles
The application comes with a curated collection of artistic styles:

1. **Camera Pixel Style** - Game Boy camera aesthetic with grayscale pixel effects
2. **Studio Realism** - Professional studio-quality realistic imagery  
3. **Sketch Smudge** - Artistic sketch and smudge effects
4. **Qwen Anime** - High-quality anime style illustrations
5. **Fragmented Portraiture** - Abstract fragmented portrait effects
6. **Synthetic Face** - AI-generated synthetic facial features
7. **Macne Style Enhancer** - Enhanced character style rendering
8. **Qwen Glitch** - Digital glitch and distortion effects
9. **Modern Anime Lora** - Contemporary Japanese anime styling
10. **Apple QuickTake 150** - Vintage digital camera aesthetic

## Installation

### Requirements
```bash
pip install torch torchvision
pip install diffusers transformers
pip install gradio spaces
pip install huggingface_hub
pip install Pillow numpy
pip install requests
```

### Hardware Requirements
- **GPU**: CUDA-compatible GPU recommended (automatically detects and uses CUDA if available)
- **VRAM**: Minimum 8GB recommended for optimal performance
- **Storage**: At least 10GB free space for model weights

## Usage

### Basic Operation

1. **Select a LoRA Style**: Click on any style from the gallery to activate it
2. **Enter Your Prompt**: Type your desired image description in the prompt field
3. **Choose Generation Mode**:
   - Fast (8 steps): Quick generation with Lightning LoRA
   - Base (48 steps): Higher quality, slower generation
4. **Set Aspect Ratio**: Choose from available ratios based on your needs
5. **Generate**: Click the Generate button or press Enter in the prompt field

### Advanced Settings

#### Guidance Scale (True CFG)
- Range: 1.0 - 5.0
- Lower values for speed mode, higher for quality
- Controls how closely the model follows your prompt

#### Steps
- Range: 4 - 50 steps
- Automatically adjusted based on generation mode
- More steps generally mean higher quality

#### Seed Control
- **Randomize Seed**: Enable for varied results each generation
- **Manual Seed**: Set specific seed for reproducible results
- Range: 0 to 2,147,483,647

#### LoRA Scale
- Range: 0 - 2.0
- Controls the strength of the LoRA style application
- 1.0 is the default strength

### Custom LoRA Integration

The application supports loading custom Qwen-Image compatible LoRAs:

1. **Hugging Face Repository**: Enter username/repository-name format
2. **Direct Safetensors URL**: Paste direct link to .safetensors file
3. **Automatic Validation**: The system validates LoRA compatibility
4. **Trigger Word Detection**: Automatically extracts trigger words from model cards

#### Supported Formats
- `username/repository-name`
- `https://huggingface.co/username/repository-name`
- `https://huggingface.co/username/repository-name/resolve/main/model.safetensors`

## Technical Details

### Model Architecture
- **Base Model**: Qwen/Qwen-Image
- **Scheduler**: FlowMatchEulerDiscreteScheduler with exponential time shifting
- **Precision**: bfloat16 for optimal VRAM usage
- **Lightning LoRA**: lightx2v/Qwen-Image-Lightning for fast generation

### Performance Optimizations
- **GPU Memory Management**: Automatic CUDA detection and optimization
- **LoRA Caching**: Efficient loading and unloading of LoRA weights
- **Batch Processing**: Optimized for single image generation
- **Timer Monitoring**: Built-in performance tracking

### File Structure
```
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
└── README.md             # This documentation
```

## Troubleshooting

### Common Issues

**CUDA Out of Memory**
- Reduce image resolution
- Close other GPU-intensive applications
- Use smaller LoRA scale values

**LoRA Loading Errors**
- Ensure the LoRA is compatible with Qwen-Image
- Check Hugging Face repository accessibility
- Verify .safetensors file exists

**Slow Generation**
- Use Fast mode for quicker results
- Reduce number of inference steps
- Check GPU utilization

### Error Messages
- **"Invalid LoRA"**: The provided LoRA is not compatible with Qwen-Image
- **"You must select a LoRA"**: Choose a style from the gallery before generating
- **"CUDA not available"**: The application will fall back to CPU mode (slower)

## Development

### Code Structure
The application is built with:
- **Gradio**: Web interface framework
- **Diffusers**: Hugging Face diffusion pipeline
- **PyTorch**: Deep learning backend
- **Spaces**: GPU acceleration decorator

### Key Functions
- `process_adapter_generation()`: Main generation pipeline
- `handle_lora_selection()`: LoRA selection logic
- `incorporate_custom_adapter()`: Custom LoRA validation and loading
- `compute_image_dimensions()`: Aspect ratio calculations

## License

This project is licensed under the Apache License 2.0.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Acknowledgments

- Qwen team for the base image generation model
- Lightning LoRA developers for fast inference optimization
- All LoRA creators in the curated collection
- Hugging Face for hosting and infrastructure
