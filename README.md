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
- Packed training windows: several documents rendered into ONE window with
  EOS separators and exact segment boundaries (`content_sized=True`)
- Single-channel fast path for greyscale consumers (`render(..., channels=1)`)

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
print(encoding.pixel_values.dtype)  # torch.uint8, raw pixels in [0, 255] on CPU (see note below)
print(encoding.attention_mask.shape)  # torch.Tensor: [batch_size, seq_length]
print(encoding.attention_mask.sum(dim=1))  # number of non-padding patches per text
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

print(encoding.pixel_values.shape)  # [3, 3, 24, W] — W is cropped to the longest text
print(encoding.attention_mask.sum(dim=1))  # number of text patches for each input
```

### Packed Training Windows

For LM pretraining, several documents can be packed into a single fixed-width
window instead of one document per sequence. Pass a tuple of texts with
`content_sized=True`: the documents are drawn onto one surface separated by EOS
patches, overflow is clipped at the window edge, and `sep_patches` reports the
exact separator offsets — divide by `pixels_per_patch` for per-document block
boundaries (segment ids, attention masks, position ids).

```python
from pixar_render import PixarProcessor

processor = PixarProcessor(pixels_per_patch=16, max_seq_length=180)
processor.renderer.content_sized = True

enc = processor.render(
    [("First document.", "Second document.", "Third one.")],
    truncate=True, add_eos=True, padding_side="right",
    channels=1,                      # [B, 1, H, W] uint8, one copy per item
)
boundaries = [s // processor.renderer.pixels_per_patch for s in enc.sep_patches[0]]
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
print(sliced_encoding.attention_mask.sum(dim=1))
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

### Compacting Trailing White Space

```python
from pixar_render import PixarProcessor

processor = PixarProcessor()
texts = ["short", "a much longer sentence"]
encoding = processor.render(texts, padding_side="left")

# Shift text right so at most 5 white pixels remain before the right edge;
# leading all-white patches are masked out of attention_mask.
compact_encoding = processor.align_text_to_right_edge(encoding, max_dist_to_edge=5)

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
    max_seq_length=1024,
    fallback_fonts_dir="./my_fonts",   # optional, see note below
)

# Save configuration
processor.save("./config")
# Creates: ./config/pixar_processor_conf.json
# If fallback_fonts_dir is set, all fallback fonts AND the primary font are
# copied into ./config/fonts, so the folder is fully self-contained and
# reproduces identical rendering on any machine.

# Later (or on another machine), restore the same processor
loaded_processor = PixarProcessor.load("./config")
```

### Using with PyTorch Models

```python
import torch
from pixar_render import PixarProcessor

processor = PixarProcessor()

# Render text -> raw uint8 pixels in [0, 255] on the CPU
texts = ["Training sample 1", "Training sample 2"]
encoding = processor.render(texts)

# render() does rendering + batch assembly ONLY. All numeric transforms are
# device-agnostic tools — move the batch to the GPU first, then apply them
# there (uint8 transfers 4x less data than float32, and the math is free on GPU):
pixel_values = encoding.pixel_values.to('cuda:0', non_blocking=True)
pixel_values = PixarProcessor.normalize(pixel_values)          # float32 [0, 1]
# pixel_values = PixarProcessor.binarize(pixel_values)         # optional: pure black/white
output = your_vision_model(
    pixel_values=pixel_values,
    attention_mask=encoding.attention_mask.to('cuda:0'),
)
```

> **Note (v0.2.0+, breaking changes):** `render()` returns raw `uint8` pixel values in
> `[0, 255]` on the CPU. It does not normalise, binarize, repeat channels or move
> tensors to a device — use the GPU-side tools `normalize` / `binarize` /
> `expand_channels` instead. To reproduce the pre-0.2.0 output:
> `encoding.pixel_values.float() / 255`. **Since v0.4.0**, grayscale processors
> (`rgb=False`) return `[batch, 1, height, width]` — call `expand_channels()` on the
> GPU if your model expects 3 channels (it is a zero-copy view) — and the unused
> `device` / `pad_size` constructor arguments are removed (configs saved by older
> versions still load; stale keys are ignored with a warning).

### Binary (Black/White) Pixels

```python
from pixar_render import PixarProcessor

processor = PixarProcessor()
encoding = processor.render("Binary rendered text")

# Threshold to pure black/white with the device-agnostic tool (run it on the
# GPU in training code). uint8 in -> uint8 {0, 255} out; float in -> {0., 1.} out.
bw = PixarProcessor.binarize(encoding.pixel_values)

images = processor.convert_to_pil(PixarProcessor.binarize(encoding))
images[0].save("binary_output.png")
```

> `PixarProcessor(binary=True)` still works but is deprecated: it binarizes inside
> `render()` on the CPU. Prefer calling `binarize()` after moving the batch to the GPU.

### Grayscale Mode

```python
from pixar_render import PixarProcessor

processor = PixarProcessor(rgb=False)          # faster rendering, 1/3 the data
encoding = processor.render("Grayscale text")
print(encoding.pixel_values.shape)             # [1, 1, height, width] (v0.4.0)

# If the model expects 3 channels, expand on the GPU — it's a zero-copy view:
pv = encoding.pixel_values.to('cuda:0')
pv = PixarProcessor.expand_channels(pv)        # [1, 3, height, width] view
pv = PixarProcessor.normalize(pv)
```

## API Reference

### PixarProcessor

**`__init__`** parameters:
- `font_file` (str): Primary font, a path or bare file name. A bare name is looked up in `fallback_fonts_dir` first, then in the package's `resources/fonts` (default: 'GoNotoCurrent.ttf')
- `font_size` (int): Font size in points; pixel em-size is `dpi / 72 * font_size` (default: 8)
- `binary` (bool): Deprecated — binarizes inside `render()` on the CPU. Prefer the `binarize()` tool on the GPU (default: False)
- `rgb` (bool): True renders RGB `[B, 3, H, W]`; False renders grayscale `[B, 1, H, W]` — use `expand_channels()` for 3-channel models (default: True)
- `dpi` (int): Dots per inch (default: 180)
- `pixels_per_patch` (int): Pixels per patch (default: 24)
- `max_seq_length` (int): Maximum sequence length (default: 529)
- `fallback_fonts_dir` (str | None): Directory for fallback fonts
- `patch_len` (int): Patch length (default: 1)
- `contour_r` (float): Red component of contour (default: 0.0)
- `contour_g` (float): Green component of contour (default: 0.0)
- `contour_b` (float): Blue component of contour (default: 0.0)
- `contour_alpha` (float): Contour transparency (default: 0.7)
- `contour_width` (int): Contour line width (default: 1)

**Rendering:**
- `render(text, padding_side, truncate, add_eos)`: Render text to a raw uint8 PixarEncoding (also callable as `processor(text)`)

**GPU-side tools** (static, device-agnostic — run them after moving the batch to the GPU; accept a tensor or a PixarEncoding):
- `normalize(pixels)`: uint8 [0, 255] -> float32 [0, 1]
- `binarize(pixels, threshold=0.5)`: threshold to pure black/white
- `expand_channels(pixels, num_channels=3)`: grayscale [B, 1, H, W] -> [B, 3, H, W] zero-copy view

**Visualisation:**
- `convert_to_pil(encoding, square, contour)`: Convert to PIL Images
- `save_as_images(encoding, dir_path, square, contour)`: Save images to directory

**Patch-level editing:**
- `slice(encoding, start, end)`: Extract patch range
- `insert(encoding, start, end, inserted)`: Insert encoding into another
- `append(encoding, inserted)`: Concatenate two encodings
- `align_text_to_right_edge(encoding, max_dist_to_edge)`: Compact trailing white space (in-place variant: `align_text_to_right_edge_`)

**Persistence:**
- `save(dir_path)`: Save configuration to JSON; bundles primary + fallback fonts when `fallback_fonts_dir` is set
- `load(dir_path)`: Restore a processor from a saved directory (classmethod)

### PixarEncoding

Dataclass containing:
- `pixel_values` (torch.Tensor): Rendered pixel values, uint8 in [0, 255] on CPU, [batch, channels, height, width] (channels: 3 RGB / 1 grayscale)
- `attention_mask` (torch.Tensor): Attention mask [batch, seq_length]
- `sep_patches` (List[List[int]]): Separator (EOS) patch indices per sample

**Methods:**
- `to(device)`: Move tensors to device
- `clone()`: Create a deep copy
- `enc[i]` / `enc[a:b]`: Select a sub-batch (keeps the batch dimension)

## Requirements

- Python >= 3.11
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
