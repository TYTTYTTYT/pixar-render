# Pixar-Render

A Python library for rendering text into visual representations as pixel tensors. This project provides rendering functions for the PIXAR project, converting text strings into images with configurable fonts, colors, and patch-based representations suitable for vision-language models.

## Features

- Convert text to pixel-based tensor representations
- Configurable font rendering with PangoCairo backend
- Support for batch processing
- Attention mask generation for sequence models
- Patch-based encoding with customizable patch sizes
- Image export capabilities (PIL and file output)
- Configuration save/load functionality
- Text encoding slicing and insertion operations
- White space reduction for compact representations

## Installation

Install from PyPI:

```bash
pip install Pixar-Render
```

Or install from source:

```bash
git clone https://github.com/TYTTYTTYT/Pixar-Render.git
cd Pixar-Render
pip install -e .
```

## Quick Start

### Basic Usage

```python
from pixar_render import PixarProcessor

# Initialize the processor with default settings
processor = PixarProcessor()

# Render a single text string
text = "Hello, World!"
encoding = processor.render(text)

# Access the pixel values and attention mask
print(encoding.pixel_values.shape)  # torch.Tensor: [batch_size, channels, height, width]
print(encoding.attention_mask.shape)  # torch.Tensor: [batch_size, seq_length]
print(encoding.num_text_patches)  # List of patch counts per text
```

### Batch Processing

```python
from pixar_render import PixarProcessor

processor = PixarProcessor()

# Render multiple texts at once
texts = [
    "First sentence.",
    "Second sentence with more text.",
    "Third one."
]
encoding = processor.render(texts)

print(encoding.pixel_values.shape)  # [3, 3, 24, 12696]
print(encoding.num_text_patches)  # Number of text patches for each input
```

### Custom Configuration

```python
from pixar_render import PixarProcessor

# Initialize with custom settings
processor = PixarProcessor(
    font_size=12,                    # Larger font size
    font_color="blue",               # Blue text
    background_color="lightyellow",  # Light yellow background
    pixels_per_patch=32,             # 32 pixels per patch instead of 24
    max_seq_length=1024,             # Maximum numer of patches
    dpi=240                          # Higher DPI for better quality
)

text = "Custom styled text"
encoding = processor.render(text)
```

### Converting to PIL Images

```python
from pixar_render import PixarProcessor

processor = PixarProcessor()
text = "Visualize this text"
encoding = processor.render(text)

# Convert to PIL images (returns a list of PIL.Image objects)
images = processor.convert_to_pil(encoding, square=True, contour=False)

# Display or save the first image
images[0].show()
images[0].save("output.png")
```

### Saving Images to Directory

```python
from pixar_render import PixarProcessor

processor = PixarProcessor()
texts = ["First text", "Second text", "Third text"]
encoding = processor.render(texts)

# Save all rendered images to a directory
processor.save_as_images(
    encoding,
    dir_path="./output_images",
    square=True,      # Reshape to square format
    contour=False     # Don't add contours
)
# This creates: output_images/0.png, output_images/1.png, output_images/2.png
```

### Adding Contours

```python
from pixar_render import PixarProcessor

# Initialize with contour settings
processor = PixarProcessor(
    contour_r=1.0,           # Red channel
    contour_g=0.0,           # Green channel
    contour_b=0.0,           # Blue channel (red contours)
    contour_alpha=0.7,       # Contour transparency
    contour_width=2,         # Contour line width
    patch_len=1              # Patches per contour cell
)

text = "Text with contours"
encoding = processor.render(text)

# Convert to image with contours
images = processor.convert_to_pil(encoding, square=True, contour=True)
images[0].save("contoured_output.png")
```

### Working with Multi-turn Conversations

```python
from pixar_render import PixarProcessor

processor = PixarProcessor()

# Render conversation turns as tuples
conversation = [
    ("User: Hello!", "Assistant: Hi there!"),
    ("User: How are you?", "Assistant: I'm doing well!")
]

encoding = processor.render(conversation)
print(encoding.sep_patches)  # Shows separator patch positions
```

### Slicing Encodings

```python
from pixar_render import PixarProcessor

processor = PixarProcessor()
text = "This is a long piece of text"
encoding = processor.render(text)

# Extract patches from index 5 to 15
sliced_encoding = processor.slice(encoding, start=5, end=15)

print(sliced_encoding.pixel_values.shape)
print(sliced_encoding.num_text_patches)
```

### Inserting Encodings

```python
from pixar_render import PixarProcessor

processor = PixarProcessor()

# Create base encoding
base_text = "Hello ___ World"
base_encoding = processor.render(base_text)

# Create text to insert
insert_text = "Beautiful"
insert_encoding = processor.render(insert_text)

# Insert at specific patch positions (e.g., patches 6-10)
combined = processor.insert(base_encoding, start=6, end=10, inserted=insert_encoding)
```

### Reducing White Space

```python
from pixar_render import PixarProcessor

processor = PixarProcessor()
text = "Text with    lots    of    spaces"
encoding = processor.render(text)

# Reduce consecutive white pixels to maximum of 5
compact_encoding = processor.reduce_white_space(encoding, max_white_space=5)

# Display the image
processor.convert_to_pil(compact_encoding)[0]
```

### Saving and Loading Configuration

```python
from pixar_render import PixarProcessor

# Create processor with custom settings
processor = PixarProcessor(
    font_size=10,
    dpi=200,
    pixels_per_patch=28,
    max_seq_length=1024
)

# Save configuration
processor.save_conf("./config")
# Creates: ./config/pixar_processor_conf.json

# Later, load the same configuration
loaded_processor = PixarProcessor.load_conf("./config")
```

### Using with PyTorch Models

```python
import torch
from pixar_render import PixarProcessor

processor = PixarProcessor(device='cuda:0')

# Render text
texts = ["Training sample 1", "Training sample 2"]
encoding = processor.render(texts)

# Move to device
encoding = encoding.to('cuda:0')

# Use in your model
# pixel_values: [batch_size, 3, height, width]
# attention_mask: [batch_size, seq_length]
output = your_vision_model(
    pixel_values=encoding.pixel_values,
    attention_mask=encoding.attention_mask
)
```

### Binary Mode

```python
from pixar_render import PixarProcessor

# Render in binary mode (black and white only)
processor = PixarProcessor(binary=True)

text = "Binary rendered text"
encoding = processor.render(text)

# Pixel values will be 0 or 1
images = processor.convert_to_pil(encoding)
images[0].save("binary_output.png")
```

## API Reference

### PixarProcessor

**`__init__`** parameters:
- `font_file` (str): Font file name (default: 'GoNotoCurrent.ttf')
- `font_size` (int): Font size in points (default: 8)
- `font_color` (str): Text color (default: "black")
- `background_color` (str): Background color (default: "white")
- `binary` (bool): Binarize output (default: False)
- `rgb` (bool): Use RGB mode (default: True)
- `dpi` (int): Dots per inch (default: 180)
- `pad_size` (int): Padding size (default: 3)
- `pixels_per_patch` (int): Pixels per patch (default: 24)
- `max_seq_length` (int): Maximum sequence length (default: 529)
- `fallback_fonts_dir` (str | None): Directory for fallback fonts
- `patch_len` (int): Patch length (default: 1)
- `contour_r` (float): Red component of contour (default: 0.0)
- `contour_g` (float): Green component of contour (default: 0.0)
- `contour_b` (float): Blue component of contour (default: 0.0)
- `contour_alpha` (float): Contour transparency (default: 0.7)
- `contour_width` (int): Contour line width (default: 1)
- `device` (str | int): Processing device (default: 'cpu')

**Methods:**
- `render(text)`: Render text to PixarEncoding
- `convert_to_pil(encoding, square, contour)`: Convert to PIL Images
- `save_as_images(encoding, dir_path, square, contour)`: Save images to directory
- `slice(encoding, start, end)`: Extract patch range
- `insert(encoding, start, end, inserted)`: Insert encoding into another
- `reduce_white_space(encoding, max_white_space)`: Reduce white space
- `save_conf(dir_path)`: Save configuration to JSON
- `load_conf(dir_path)`: Load configuration from JSON (classmethod)

### PixarEncoding

Dataclass containing:
- `pixel_values` (torch.Tensor): Rendered pixel values [batch, channels, height, width]
- `attention_mask` (torch.Tensor): Attention mask [batch, seq_length]
- `num_text_patches` (List[int]): Number of text patches per sample
- `sep_patches` (List[List[int]]): Separator patch positions per sample

**Methods:**
- `to(device)`: Move tensors to device
- `clone()`: Create a deep copy

## Requirements

- Python = 3.11
- numpy
- torch
- torchvision
- pillow
- PangoCairo (for text rendering)

## License

Apache License 2.0

## Links

- Homepage: https://github.com/TYTTYTTYT/Pixar-Render
- Bug Tracker: https://github.com/TYTTYTTYT/Pixar-Render/issues
- PyPI: https://pypi.org/project/Pixar-Render/

## Author

Yintao Tai (tai.yintao@gmail.com)
