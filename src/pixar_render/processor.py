"""Processor for Pixar-Render: text -> pixel-tensor encodings for vision language models.

The pipeline is intentionally split in two:

1. ``PixarProcessor.render()`` runs on the CPU (e.g. inside DataLoader workers) and
   does *only* rendering and batch assembly. It returns raw ``uint8`` pixels in
   ``[0, 255]`` with no normalisation, no device placement and no numeric
   post-processing, so the CPU side stays as fast as possible.
2. Numeric transforms (``normalize``, ``binarize``, ``expand_channels``) are
   device-agnostic tools: move the batch to the GPU first, then apply them there,
   where they are effectively free.

Typical training loop::

    processor = PixarProcessor.load("./render_config")
    enc = processor.render(["some text", "另一段文本"])     # CPU, uint8 [B, C, H, W]
    pv = enc.pixel_values.to("cuda", non_blocking=True)     # 1/4 the bytes of float32
    pv = PixarProcessor.normalize(pv)                       # float [0, 1], on GPU
    out = model(pixel_values=pv, attention_mask=enc.attention_mask.to("cuda"))
"""
from typing import Union, List, Tuple, Callable, ParamSpec, TypeVar, Literal
import json
import shutil
import warnings
from pathlib import Path
from math import ceil, sqrt
from copy import deepcopy
from dataclasses import dataclass
import functools
import inspect

import numpy as np
import torch
from torchvision.transforms import ToPILImage
from PIL import Image

from .pangocairo_render import PangoCairoTextRenderer

P = ParamSpec("P")
R = TypeVar("R")

def cache(func: Callable[P, R]) -> Callable[P, R]:
    cached = functools.cache(func)          # do the actual caching
    functools.update_wrapper(cached, func)  # keep name, doc, __wrapped__
    # restore typing metadata many IDEs rely on
    cached.__annotations__ = getattr(func, "__annotations__", {})
    try:
        cached.__signature__ = inspect.signature(func)  # type: ignore # show true signature
    except Exception:
        pass
    return cached  # type: ignore[return-value]  # (rarely needed)

@cache
def square_number(num: int) -> tuple[int, int]:
    upper = int(sqrt(num)) + 1
    n1, n2 = -99999, 99999
    for i in range(1, upper):
        j = num // i
        if i * j == num:
            if j - i < n2 - n1:
                n1, n2 = i, j

    return n1, n2

@cache
def contour_map(pixel_per_patch: int, patch_len: int, image_width: int, width: int) -> torch.Tensor:
    contour = torch.zeros((pixel_per_patch, image_width), dtype=torch.float32)
    for i in range(width):
        contour[i, :] = 1.0
        contour[-i - 1, :] = 1.0

    for i in range(0, image_width, pixel_per_patch * patch_len):
        for w in range(width):
            contour[:, i + w] = 1.0
            contour[:, i - w] = 1.0

    return contour

@cache
def contour_image(
    pixel_per_patch: int,
    patch_len: int,
    image_width: int,
    width: int,
    R: float,
    G: float,
    B: float
) -> torch.Tensor:
    c_map = contour_map(pixel_per_patch, patch_len, image_width, width)
    image = torch.zeros((3, c_map.shape[0], c_map.shape[1]), dtype=torch.float32)
    image[0, :, :] = c_map
    image[1, :, :] = c_map
    image[2, :, :] = c_map
    image[0, :, :] *= R
    image[1, :, :] *= G
    image[2, :, :] *= B
    return image

def cal_sep_patches(sep_patches: List[int], patch_len: int, pixel_per_patch: int) -> List[int]:
    sep_idxes = []
    for n in sep_patches:
        idx = n / patch_len / pixel_per_patch
        if float(int(idx)) == idx:
            sep_idxes.append(int(idx))
    return sep_idxes

def create_attention_mask(
    dims: Tuple[int, int],
    padding_side: Literal['right', 'left'],
    seq_lens: List[int]
) -> torch.Tensor:
    """
    Creates an attention mask tensor.

    Args:
        dims (Tuple[int, int]): The dimensions of the attention mask (batch_size, seq_len).
        padding_side (str): The side to pad on, either 'left' or 'right'.
        seq_lens (List[int]): A list containing the number of
            non-padding tokens for each item in the batch.

    Returns:
        torch.Tensor: The attention mask.
    """
    batch_size, seq_len = dims
    attention_mask = torch.zeros(dims, dtype=torch.long)

    if padding_side not in ['left', 'right']:
        raise ValueError("padding_side must be 'left' or 'right'")

    for i in range(batch_size):
        n = seq_lens[i]
        if n > seq_len:
            raise ValueError(
                f"Number of non-padding tokens ({n}) for item {i} is greater than sequence length ({seq_len})"
            )
        if padding_side == 'right':
            attention_mask[i, :n] = 1
        else: # padding_side == 'left'
            attention_mask[i, -n:] = 1

    return attention_mask


@dataclass
class PixarEncoding:
    """A batch of rendered text, ready to feed a vision model.

    Attributes:
        pixel_values: ``uint8`` tensor in ``[0, 255]``, shape ``[batch, channels,
            height, width]`` (``channels`` is 3 for RGB processors, 1 for
            grayscale ``rgb=False`` processors). Produced on the CPU; normalise
            and move to your device yourself (see ``PixarProcessor.normalize``).
        attention_mask: ``int64`` tensor of 0/1, shape ``[batch, num_patches]``.
            1 marks patches that contain text (or the EOS separator).
        sep_patches: For each batch item, the patch indices of black separator
            (EOS) patches, e.g. ``[[5], [7]]``.

    Example::

        enc = processor.render(["hello", "world!"])
        enc = enc.to("cuda")                      # move both tensors
        short = enc[0]                            # first item, still batched [1, ...]
    """

    pixel_values: torch.Tensor
    attention_mask: torch.Tensor
    sep_patches: List[List[int]]

    def to(self, device: Union[str, int]) -> 'PixarEncoding':
        """Return a copy with ``pixel_values`` and ``attention_mask`` moved to ``device``."""
        return PixarEncoding(
            pixel_values=self.pixel_values.to(device),
            attention_mask=self.attention_mask.to(device),
            sep_patches=deepcopy(self.sep_patches),
        )

    def clone(self) -> 'PixarEncoding':
        """Return a deep copy (tensors cloned, ``sep_patches`` deep-copied)."""
        return PixarEncoding(
            pixel_values=self.pixel_values.clone(),
            attention_mask=self.attention_mask.clone(),
            sep_patches=deepcopy(self.sep_patches),
        )

    def __getitem__(self, index: slice | int) -> 'PixarEncoding':
        """Select a sub-batch. ``enc[0]`` keeps the batch dim (shape ``[1, ...]``)."""
        if isinstance(index, int):
            index = slice(index, index+1)
        return PixarEncoding(
            pixel_values=self.pixel_values[index],
            attention_mask=self.attention_mask[index],
            sep_patches=self.sep_patches[index],
        )


PixelsOrEncoding = Union[torch.Tensor, PixarEncoding]


class PixarProcessor:
    """Renders text into pixel-tensor batches (the PIXAR project's "tokenizer").

    Design: ``render()`` is CPU-side and does *only* rendering + batch assembly,
    returning raw ``uint8`` pixels. All numeric transforms are separate,
    device-agnostic tools — call them after moving the batch to the GPU:

    - ``normalize(x)``       uint8 [0, 255] -> float32 [0, 1]
    - ``binarize(x)``        threshold to pure black/white
    - ``expand_channels(x)`` grayscale [B, 1, H, W] -> [B, 3, H, W] zero-copy view

    Inspection / manipulation tools (CPU-oriented, not for the hot path):

    - ``convert_to_pil(enc)`` / ``save_as_images(enc, dir)``  visualisation
    - ``slice(enc, a, b)`` / ``insert(...)`` / ``append(...)`` patch-level editing
    - ``align_text_to_right_edge(enc, n)``  compact left-padding whitespace
    - ``save(dir)`` / ``PixarProcessor.load(dir)``  config + bundled-fonts snapshot

    Example::

        p = PixarProcessor(font_size=8, dpi=45, pixels_per_patch=8)
        enc = p.render(["Hello", "世界"])          # uint8 [2, 3, 8, W], CPU
        pv = PixarProcessor.normalize(enc.pixel_values.to("cuda"))
    """

    CONF_FILENAME = "pixar_processor_conf.json"

    def __init__(
        self,
        font_file: str = 'GoNotoCurrent.ttf',
        font_size: int = 8,
        binary: bool = False,
        rgb: bool = True,
        dpi: int = 180,
        pad_size: int = 3,
        pixels_per_patch: int = 24,
        max_seq_length: int = 529,
        add_eos: bool = True,
        padding_side: Literal['left', 'right'] = 'right',
        truncate: bool = True,
        fallback_fonts_dir: str | None = None,
        patch_len: int = 1,
        contour_r: float = 0.0,
        contour_g: float = 0.0,
        contour_b: float = 0.0,
        contour_alpha: float = 0.7,
        contour_width: int = 1,
        device: Union[str, int] = 'cpu'
    ):
        """Create a text-to-pixels processor.

        Args:
            font_file: Primary font, either a path or a bare file name. A bare
                name is looked up first in ``fallback_fonts_dir`` (if set), then
                in the package's built-in ``resources/fonts``.
            font_size: Font size in points. The pixel em-size is
                ``dpi / 72 * font_size`` and must fit inside ``pixels_per_patch``.
            binary: Deprecated. Binarizes inside ``render()`` (costs CPU time).
                Prefer ``processor.binarize(...)`` on the GPU after moving the
                batch. Kept so that old saved configs keep their semantics.
            rgb: If True (default), render in RGB and return ``[B, 3, H, W]``.
                If False, render grayscale and return ``[B, 1, H, W]`` — use
                ``expand_channels()`` if your model wants 3 channels (it is a
                zero-copy view, do it on the GPU).
            dpi: Dots per inch; scales the font (em px = dpi / 72 * font_size).
            pad_size: Reserved; padding around the rendered text (currently unused).
            pixels_per_patch: Height of the rendered strip and side length of one
                square patch, in pixels.
            max_seq_length: Maximum number of patches per sequence (the rendering
                surface is ``max_seq_length * pixels_per_patch`` px wide).
            add_eos: Whether to draw a black EOS separator patch after the text.
            padding_side: Which side short sequences are padded on in a batch.
            truncate: If True, crop the batch width to the longest sequence.
            fallback_fonts_dir: Directory of extra ``.ttf``/``.otf`` fonts used
                when the primary font lacks a glyph. Setting this (even to an
                empty dir) *disables system fonts entirely*, making rendering
                reproducible across machines. ``None`` falls back to whatever
                fonts the host system has — not reproducible.
            patch_len: Number of ``pixels_per_patch`` columns that form one
                logical patch (token). ``max_seq_length`` must be divisible by it.
            contour_r / contour_g / contour_b: Contour colour used by
                ``convert_to_pil(..., contour=True)``.
            contour_alpha: Contour opacity for visualisation.
            contour_width: Contour line width in pixels.
            device: Deprecated and ignored. ``render()`` always returns CPU
                tensors; move them yourself (``enc.to(device)``) or in the
                training framework.
        """
        self.font_file = font_file
        self.font_size = font_size
        self.rgb = rgb
        self.binary = binary
        self.dpi = dpi
        self.pad_size = pad_size
        self.pixels_per_patch = pixels_per_patch
        self.max_seq_length = max_seq_length
        self.add_eos = add_eos
        self.padding_side = padding_side
        self.truncate = truncate
        self.fallback_fonts_dir = fallback_fonts_dir
        self.patch_len = patch_len
        self.contour_r = contour_r
        self.contour_g = contour_g
        self.contour_b = contour_b
        self.contour_alpha = contour_alpha
        self.contour_width = contour_width
        self.device = device

        if binary:
            warnings.warn(
                "PixarProcessor(binary=True) is deprecated: it binarizes inside render() "
                "on the CPU. Prefer binary=False and calling processor.binarize(batch) "
                "after moving the batch to the GPU.",
                DeprecationWarning,
                stacklevel=2,
            )

        assert max_seq_length % patch_len == 0, \
            f"max_seq_length must be divisible by patch_len, but got {max_seq_length} and {patch_len}"

        self.renderer = PangoCairoTextRenderer(
            font_file,
            font_size,
            rgb,
            dpi,
            pad_size,
            pixels_per_patch,
            max_seq_length,
            fallback_fonts_dir,
            patch_len
        )

        self._to_pil = ToPILImage(mode="RGB")
        self._block_width = self.patch_len * self.pixels_per_patch

    # ------------------------------------------------------------------
    # Device-agnostic numeric tools (run them on the GPU in training code)
    # ------------------------------------------------------------------

    @staticmethod
    def normalize(pixels: PixelsOrEncoding) -> PixelsOrEncoding:
        """Convert raw ``uint8`` pixels in ``[0, 255]`` to ``float32`` in ``[0, 1]``.

        Device-agnostic: runs wherever the tensor lives, so call it *after*
        moving the batch to the GPU — that is both faster and transfers 4x less
        data over PCIe than moving float32.

        Args:
            pixels: A pixel tensor, or a ``PixarEncoding`` (its ``pixel_values``
                are transformed; ``attention_mask``/``sep_patches`` are shared).

        Returns:
            The same kind of object that was passed in (tensor -> tensor,
            encoding -> encoding).

        Example::

            pv = enc.pixel_values.to("cuda", non_blocking=True)
            pv = PixarProcessor.normalize(pv)      # float32 [0, 1] on the GPU
        """
        if isinstance(pixels, PixarEncoding):
            return PixarEncoding(
                pixel_values=PixarProcessor.normalize(pixels.pixel_values),  # type: ignore[arg-type]
                attention_mask=pixels.attention_mask,
                sep_patches=pixels.sep_patches,
            )
        return pixels.to(torch.float32) / 255

    @staticmethod
    def binarize(pixels: PixelsOrEncoding, threshold: float = 0.5) -> PixelsOrEncoding:
        """Threshold pixels to pure black/white (replaces the ``binary=True`` mode).

        Averages the channel dimension and thresholds it. The input dtype domain
        is preserved: ``uint8`` input yields ``uint8`` {0, 255}; float input
        yields float {0.0, 1.0}. Device-agnostic — prefer calling it on the GPU.

        Args:
            pixels: ``[B, C, H, W]`` pixel tensor (uint8 or float), or a
                ``PixarEncoding``.
            threshold: Brightness cut in the ``[0, 1]`` domain; pixels brighter
                than this become white. Defaults to 0.5.

        Returns:
            Black/white pixels with the same shape and object kind as the input.
            The channel dim is an expanded zero-copy view; call ``.contiguous()``
            if you need to write into it.

        Example::

            pv = enc.pixel_values.to("cuda")
            pv = PixarProcessor.normalize(PixarProcessor.binarize(pv))
        """
        if isinstance(pixels, PixarEncoding):
            return PixarEncoding(
                pixel_values=PixarProcessor.binarize(pixels.pixel_values, threshold),  # type: ignore[arg-type]
                attention_mask=pixels.attention_mask,
                sep_patches=pixels.sep_patches,
            )
        channels = pixels.shape[1]
        if pixels.dtype == torch.uint8:
            gray = pixels.float().mean(dim=1, keepdim=True)
            out = (gray > threshold * 255).to(torch.uint8) * 255
        else:
            gray = pixels.mean(dim=1, keepdim=True)
            out = (gray > threshold).to(pixels.dtype)
        return out.expand(-1, channels, -1, -1)

    @staticmethod
    def expand_channels(pixels: PixelsOrEncoding, num_channels: int = 3) -> PixelsOrEncoding:
        """Expand grayscale ``[B, 1, H, W]`` pixels to ``[B, 3, H, W]`` for Conv2d.

        Returns a zero-copy *view* (``Tensor.expand``): no memory is allocated on
        any device, so always transfer the 1-channel tensor and expand on the
        GPU rather than repeating channels on the CPU.

        Args:
            pixels: ``[B, 1, H, W]`` tensor (any dtype), or a ``PixarEncoding``
                from a grayscale (``rgb=False``) processor.
            num_channels: Target channel count. Defaults to 3.

        Returns:
            A ``[B, num_channels, H, W]`` view (or an encoding wrapping one).
            Call ``.contiguous()`` on it if you need to write into it.

        Example::

            enc = gray_processor.render(texts)              # [B, 1, H, W] uint8
            pv = enc.pixel_values.to("cuda")                # transfer 1/3 the bytes
            pv = PixarProcessor.expand_channels(pv)         # [B, 3, H, W] view
        """
        if isinstance(pixels, PixarEncoding):
            return PixarEncoding(
                pixel_values=PixarProcessor.expand_channels(pixels.pixel_values, num_channels),  # type: ignore[arg-type]
                attention_mask=pixels.attention_mask,
                sep_patches=pixels.sep_patches,
            )
        return pixels.expand(-1, num_channels, -1, -1)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def __call__(
        self,
        text: Union[str, Tuple[str, ...], List[Union[str, Tuple[str, ...]]]],
        padding_side: Literal['left', 'right'] | None = None,
        truncate: bool | None = None,
        add_eos: bool | None = None
    ) -> PixarEncoding:
        """Alias for :meth:`render` — ``processor(texts)`` == ``processor.render(texts)``."""
        return self.render(text, padding_side, truncate, add_eos)

    def render(
        self,
        text: Union[str, Tuple[str, ...], List[Union[str, Tuple[str, ...]]]],
        padding_side: Literal['left', 'right'] | None = None,
        truncate: bool | None = None,
        add_eos: bool | None = None
    ) -> PixarEncoding:
        """Render text into a raw ``uint8`` pixel batch (CPU-only, no post-processing).

        This is the hot path: it renders with PangoCairo, assembles the batch
        (truncation, optional EOS removal, padding side, attention mask) and
        returns raw pixels. It deliberately does **not** normalise, binarize,
        repeat channels or move to a device — do those on the GPU with the
        :meth:`normalize` / :meth:`binarize` / :meth:`expand_channels` tools.

        Args:
            text: One of
                - ``str`` — a single text;
                - ``tuple`` of 2 strings — a text pair (rendered with a black
                  separator patch between them);
                - ``tuple`` of 3+ strings — a text list, one separator each;
                - ``list`` of the above — a batch (one image per element).
            padding_side: Where to pad shorter sequences ('left' or 'right').
                Defaults to the constructor setting.
            truncate: Crop the batch width to the longest sequence. Defaults to
                the constructor setting.
            add_eos: Keep the black EOS separator patch after each text.
                Defaults to the constructor setting.

        Returns:
            A :class:`PixarEncoding` on the CPU with ``pixel_values`` of dtype
            ``uint8`` in ``[0, 255]`` and shape ``[B, 3, H, W]`` (RGB) or
            ``[B, 1, H, W]`` (grayscale, ``rgb=False``).

        Example::

            enc = processor.render(["short", "a much longer sentence"])
            pv = enc.pixel_values.to("cuda", non_blocking=True)
            pv = PixarProcessor.normalize(pv)   # float [0, 1] on the GPU
        """
        if padding_side is None:
            padding_side = self.padding_side # type: ignore
        if truncate is None:
            truncate = self.truncate
        if add_eos is None:
            add_eos = self.add_eos

        if isinstance(text, list):
            rendered = [self.renderer(t) for t in text]
        else:
            rendered = [self.renderer(text)]

        num_text_patches = [
            ceil((p.num_text_patches + 1) / self.patch_len) for p in rendered
        ]
        max_num_patches = max(num_text_patches)

        # The renderer always draws onto a full `max_seq_length`-wide surface, but
        # when truncating only the first `max_num_patches` patches ever survive
        # (often <5% of the width for short inputs). Converting the whole surface
        # and discarding the rest afterwards used to dominate runtime, so slice
        # down to the needed width *before* building the batch.
        if truncate:
            conv_width = max_num_patches * self._block_width
        else:
            conv_width = rendered[0].pixel_values.shape[1]

        # `p.pixel_values` is a non-contiguous numpy view (negative-strided BGR->RGB),
        # so build one contiguous batch and hand it to torch via from_numpy (a view,
        # no extra copy). The channel layout is arranged here in numpy so from_numpy
        # returns a ready [B, C, H, W] tensor with no torch-side copy.
        #
        # Pixels stay raw uint8 [0, 255] on the CPU: normalisation, channel
        # expansion, binarization and device placement belong to the caller (they
        # are cheap on the GPU, expensive here).
        if self.rgb:
            batch = np.stack(
                [np.ascontiguousarray(p.pixel_values[:, :conv_width, :]) for p in rendered]
            )
            # [B, H, W, C] -> [B, C, H, W] to fit the Conv2d operator
            batch = np.ascontiguousarray(batch.transpose(0, 3, 1, 2))
        else:
            batch = np.stack(
                [np.ascontiguousarray(p.pixel_values[:, :conv_width]) for p in rendered]
            )
            # [B, H, W] -> [B, 1, H, W]; expand to 3 channels on the GPU with
            # expand_channels() if the model needs it (zero-copy there).
            batch = batch[:, None, :, :]
        pixel_values = torch.from_numpy(batch)

        # dimension of pixel_values: [batch_size, channels, height, width], uint8 [0, 255]
        if self.binary:
            # Deprecated path, kept so old saved configs render the same. contiguous()
            # materialises binarize's expanded channel view so the EOS-removal and
            # left-padding steps below can write into the tensor.
            pixel_values = self.binarize(pixel_values).contiguous()  # type: ignore[union-attr]

        sep_patches = [
            self._cal_sep_patches(p.sep_patches) for p in rendered
        ]
        sep_patches = [list(set(sep)) for sep in sep_patches]
        for sep in sep_patches:
            sep.sort()

        # remove EOS patches if no need
        if not add_eos:
            for idx in range(pixel_values.shape[0]):
                if sep_patches[idx][-1] == num_text_patches[idx] - 1:
                    num_text_patches[idx] -= 1
                for sep_idx in sep_patches[idx]:
                    begin_idx = sep_idx * self._block_width
                    end_idx = begin_idx + self._block_width
                    pixel_values[idx, :, :, begin_idx:end_idx] = 255  # white (uint8)
                sep_patches[idx] = []

        # truncate if needed
        max_num_patches = max(num_text_patches)
        if truncate:
            pixel_width = max_num_patches * self._block_width
            pixel_values = pixel_values[:, :, :, :pixel_width]

        # change padding_side if needed
        if padding_side == 'left':
            for idx, n in enumerate(num_text_patches):
                if n == max_num_patches:
                    continue
                text_pixel_width = n * self._block_width
                tmp = torch.empty_like(pixel_values[idx])
                tmp[:, :, -text_pixel_width:] = pixel_values[idx, :, :, :text_pixel_width]
                tmp[:, :, :-text_pixel_width] = pixel_values[idx, :, :, text_pixel_width:]
                pixel_values[idx] = tmp
                for i in range(len(sep_patches[idx])):
                    sep_patches[idx][i] = sep_patches[idx][i] + max_num_patches - n
        else:
            if padding_side != 'right':
                raise ValueError(f"padding_side must be 'left' or 'right', but got {padding_side}")

        attention_mask = create_attention_mask(
            dims=(pixel_values.shape[0], max_num_patches),
            seq_lens=num_text_patches,
            padding_side=padding_side   # type: ignore
        )

        return PixarEncoding(
            pixel_values=pixel_values.contiguous(),
            attention_mask=attention_mask,
            sep_patches=sep_patches
        )

    def _cal_sep_patches(self, sep_patches: List[int]) -> List[int]:
        sep_idxes = []
        for n in sep_patches:
            idx = n / self.patch_len / self.pixels_per_patch
            if float(int(idx)) == idx:
                sep_idxes.append(int(idx))
        return sep_idxes

    # ------------------------------------------------------------------
    # Visualisation tools (CPU-side, not for the training hot path)
    # ------------------------------------------------------------------

    def _squarelize(self, pixel_values: torch.Tensor) -> torch.Tensor:
        np = pixel_values.shape[-1] // self.pixels_per_patch
        nrows, _ = square_number(np)

        rows = torch.tensor_split(pixel_values, nrows, dim=-1)
        square = torch.cat(rows, dim=-2).contiguous()

        return square

    def _add_contour(self, pixel_values: torch.Tensor) -> torch.Tensor:
        contour_img = contour_image(
            self.pixels_per_patch,
            self.patch_len,
            pixel_values.shape[-1],
            self.contour_width,
            self.contour_r, self.contour_g, self.contour_b
        )
        contour_m = contour_map(self.pixels_per_patch, self.patch_len, pixel_values.shape[-1], width=self.contour_width)
        reverse_m = 1 - contour_m

        pixel_values = pixel_values * reverse_m + contour_img * contour_m * self.contour_alpha + pixel_values *\
            contour_m * (1 - self.contour_alpha)

        return pixel_values

    @torch.no_grad()
    def convert_to_pil(
        self,
        pixar_encoding: PixarEncoding,
        square: bool = True,
        contour: bool = False
    ) -> List[Image.Image]:
        """Turn an encoding into PIL images for eyeballing the rendered text.

        Accepts both raw ``uint8`` encodings (the normal ``render()`` output,
        RGB or grayscale) and float ``[0, 1]`` pixel values.

        Args:
            pixar_encoding: The encoding to visualise.
            square: Reshape the 1-pixel-row strip into a roughly square image
                (rows of patches). Defaults to True.
            contour: Overlay patch-boundary contour lines (colour/opacity come
                from the constructor's ``contour_*`` settings). Defaults to False.

        Returns:
            One ``PIL.Image`` (mode RGB) per batch item.

        Example::

            imgs = processor.convert_to_pil(processor.render("check me"))
            imgs[0].save("sample.png")
        """
        pixel_values = pixar_encoding.pixel_values
        # The contour blending works in the float [0, 1] domain, so normalise here.
        if pixel_values.dtype == torch.uint8:
            pixel_values = pixel_values.float() / 255
        # Grayscale [B, 1, H, W] -> 3 channels for PIL RGB output
        if pixel_values.shape[1] == 1:
            pixel_values = pixel_values.expand(-1, 3, -1, -1)
        if contour:
            pixel_values = self._add_contour(pixel_values)
        if square:
            pixel_values = self._squarelize(pixel_values)
        pixel_values = (pixel_values * 255).to(torch.uint8)
        images = [self._to_pil(p) for p in pixel_values]
        return images

    def save_as_images(self, pixar_encoding: PixarEncoding, dir_path: str, square: bool = True, contour: bool = False):
        """Save each batch item as ``<dir_path>/<index>.png`` (see :meth:`convert_to_pil`).

        Args:
            pixar_encoding: The encoding whose images should be written.
            dir_path: Output directory; created if missing.
            square: Reshape strips into square images. Defaults to True.
            contour: Overlay patch contour lines. Defaults to False.
        """
        images = self.convert_to_pil(pixar_encoding, square, contour)
        path_dir = Path(dir_path)
        if not path_dir.exists():
            path_dir.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(images):
            img.save(Path(dir_path) / f"{i}.png")

    # ------------------------------------------------------------------
    # Config persistence
    # ------------------------------------------------------------------

    def _config(self) -> dict:
        """Return all constructor settings of this processor as a plain dict."""
        return {
            "font_file": self.font_file,
            "font_size": self.font_size,
            "binary": self.binary,
            "rgb": self.rgb,
            "dpi": self.dpi,
            "pad_size": self.pad_size,
            "pixels_per_patch": self.pixels_per_patch,
            "max_seq_length": self.max_seq_length,
            "add_eos": self.add_eos,
            "padding_side": self.padding_side,
            "truncate": self.truncate,
            "fallback_fonts_dir": self.fallback_fonts_dir,
            "patch_len": self.patch_len,
            "contour_r": self.contour_r,
            "contour_g": self.contour_g,
            "contour_b": self.contour_b,
            "contour_alpha": self.contour_alpha,
            "contour_width": self.contour_width,
            "device": self.device,
        }

    def save(self, dir_path: str) -> None:
        """Save all settings (and fonts) so :meth:`load` can rebuild this processor anywhere.

        Writes ``<dir_path>/pixar_processor_conf.json``. If ``fallback_fonts_dir``
        is set, every fallback font *and* the resolved primary font are copied
        into ``<dir_path>/fonts`` and the config is rewritten to reference them
        relatively — the folder is then fully self-contained and reproduces
        identical rendering on any machine (no reliance on system fonts).

        Args:
            dir_path: Directory to save into; created if missing.

        Example::

            p = PixarProcessor(font_file="PixeloidSans-mLxMm.ttf",
                               fallback_fonts_dir="./my_fonts")
            p.save("./render_config")     # config + bundled fonts
        """
        dst = Path(dir_path)
        dst.mkdir(parents=True, exist_ok=True)
        config = self._config()

        if self.fallback_fonts_dir is not None:
            fonts_dir = dst / "fonts"
            fonts_dir.mkdir(parents=True, exist_ok=True)
            # Copy every fallback font (matching the same "*tf" glob used at load time)
            for font in sorted(Path(self.fallback_fonts_dir).glob("*tf")):
                shutil.copy(font, fonts_dir / font.name)
            # Copy the resolved primary font into the bundle too, so it travels with the
            # fallback dir and can be found by name when loading.
            primary = Path(self.renderer.font_file)
            primary_dst = fonts_dir / primary.name
            if not primary_dst.exists():
                shutil.copy(primary, primary_dst)
            config["fallback_fonts_dir"] = "fonts"
            config["font_file"] = primary.name

        with open(dst / self.CONF_FILENAME, "w", encoding="utf-8") as fp:
            json.dump(config, fp, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, dir_path: str) -> "PixarProcessor":
        """Rebuild a processor from a directory previously written by :meth:`save`.

        A bundled (relative) ``fallback_fonts_dir`` is resolved back to an
        absolute path inside ``dir_path``, so the primary and fallback fonts are
        loaded from the bundle rather than from the host system.

        Args:
            dir_path: Directory containing ``pixar_processor_conf.json``.

        Returns:
            A processor with the saved settings.

        Example::

            processor = PixarProcessor.load("./render_config")
        """
        src = Path(dir_path)
        with open(src / cls.CONF_FILENAME, "r", encoding="utf-8") as fp:
            config = json.load(fp)

        fallback = config.get("fallback_fonts_dir")
        if fallback is not None and not Path(fallback).is_absolute():
            config["fallback_fonts_dir"] = str((src / fallback).resolve())

        return cls(**config)

    # ------------------------------------------------------------------
    # Patch-level editing tools
    # ------------------------------------------------------------------

    def slice(self, pixar_encoding: PixarEncoding, start: int, end: int) -> PixarEncoding:
        """Extract patches ``[start, end)`` from every item in the batch.

        Args:
            pixar_encoding: The encoding to slice.
            start: First patch index to keep (inclusive).
            end: End patch index (exclusive).

        Returns:
            A new encoding covering only the selected patch range
            (``sep_patches`` indices are re-based to the new origin).

        Example::

            middle = processor.slice(enc, 5, 15)   # patches 5..14
        """
        block_len = self.pixels_per_patch * self.patch_len
        # N C H W
        pixel_values = pixar_encoding.pixel_values[:, :, :, start * block_len : end * block_len]
        attention_mask = pixar_encoding.attention_mask[:, start:end]
        sep_patches = [
            [s - start for s in seq if s >= start and s < end] for seq in pixar_encoding.sep_patches
        ]

        return PixarEncoding(
            pixel_values=pixel_values.contiguous(),
            attention_mask=attention_mask.contiguous(),
            sep_patches=sep_patches,
        )

    def insert(self, pixar_encoding: PixarEncoding, start: int, end: int, inserted: PixarEncoding) -> PixarEncoding:
        """Overwrite patches ``[start, end)`` of one encoding with another encoding.

        The ``inserted`` encoding must span exactly ``end - start`` patches and
        have the same batch size, dtype and height as the base encoding.

        Args:
            pixar_encoding: The base encoding (not modified; a copy is returned).
            start: First patch index to overwrite (inclusive).
            end: End patch index (exclusive).
            inserted: The encoding whose pixels/mask replace that range.

        Returns:
            A new encoding with the range replaced and ``sep_patches`` merged.

        Example::

            base = processor.render("Hello ___ world")
            fill = processor.render("beautiful")
            combined = processor.insert(base, 6, 10, fill)
        """
        block_len = self.pixels_per_patch * self.patch_len
        # N C H W
        pixel_values = pixar_encoding.pixel_values.clone()
        pixel_values[:, :, :, start*block_len:end*block_len] = inserted.pixel_values
        sep_patches = [[
                s for s in seq1 if s < start or s >= end
            ] + [
                s + start for s in seq2 if s + start >= start and s + start < end
            ] for seq1, seq2 in zip(pixar_encoding.sep_patches, inserted.sep_patches)
        ]
        for seq in sep_patches:
            seq.sort()

        attention_mask = pixar_encoding.attention_mask.clone()
        attention_mask[:, start:end] = inserted.attention_mask

        return PixarEncoding(
            pixel_values=pixel_values.contiguous(),
            attention_mask=attention_mask.contiguous(),
            sep_patches=sep_patches
        )

    def append(self, pixar_encoding: PixarEncoding, inserted: PixarEncoding) -> PixarEncoding:
        """Concatenate a second encoding after the first, patch-wise.

        Args:
            pixar_encoding: The base encoding.
            inserted: The encoding appended on the right (same batch size,
                dtype and height).

        Returns:
            A new encoding whose width/mask are the concatenation of both;
            ``sep_patches`` of the appended part are shifted accordingly.

        Example::

            joined = processor.append(processor.render("Q: hi"), processor.render("A: hello"))
        """
        pixel_values = torch.cat([pixar_encoding.pixel_values, inserted.pixel_values], dim=-1)
        attention_mask = torch.cat([pixar_encoding.attention_mask, inserted.attention_mask], dim=-1)
        seq_patches = deepcopy(pixar_encoding.sep_patches)
        l = pixar_encoding.pixel_values.shape[-1] // self._block_width
        for idx, seq in enumerate(inserted.sep_patches):
            seq_patches[idx].extend([s + l for s in seq])

        return PixarEncoding(pixel_values, attention_mask, seq_patches)

    def _align_text_to_right_edge_at_i(self, i: int, pixar_encoding: PixarEncoding, max_dist_to_edge: int) -> PixarEncoding:
        """In-place helper for :meth:`align_text_to_right_edge` handling batch item ``i``."""
        # C, H, W
        pixel_values = pixar_encoding.pixel_values[i].clone()
        # white value depends on the pixel dtype: 255 for uint8 render() output, 1.0 for float
        white = 255 if pixel_values.dtype == torch.uint8 else 1.0
        right_edge_patch_idx = int(pixar_encoding.attention_mask[i].nonzero().max().item())
        if right_edge_patch_idx in pixar_encoding.sep_patches[i]:
            right_edge_patch_idx -= 1

        right_edge_pixel_idx = (right_edge_patch_idx + 1) * self._block_width - 1
        current_column_idx = right_edge_pixel_idx
        while pixel_values[:, :, current_column_idx].float().mean().item() == white and current_column_idx > 0:
            current_column_idx -= 1

        dist_to_edge = right_edge_pixel_idx - current_column_idx
        if current_column_idx == 0 or dist_to_edge <= max_dist_to_edge:
            return pixar_encoding

        dist_to_move = dist_to_edge if dist_to_edge <= max_dist_to_edge else dist_to_edge - max_dist_to_edge

        num_text_pixel = current_column_idx + 1
        pixar_encoding.pixel_values[i, :, :, dist_to_move:dist_to_move+num_text_pixel] = pixel_values[:, :, :num_text_pixel]
        pixar_encoding.pixel_values[i, :, :, :dist_to_move] = white

        # scan for begining white patches and set their attention mask to 0
        _, _, W = pixar_encoding.pixel_values[i].shape
        num_blocks = W // self._block_width
        for j in range(num_blocks):
            start = j * self._block_width
            end = start + self._block_width
            if pixar_encoding.pixel_values[i, :, :, start:end].float().mean().item() == white:
                pixar_encoding.attention_mask[i, j] = 0
            else:
                break

        return pixar_encoding

    def align_text_to_right_edge_(self, pixar_encoding: PixarEncoding, max_dist_to_edge: int) -> PixarEncoding:
        """In-place variant of :meth:`align_text_to_right_edge` (modifies the input).

        Args:
            pixar_encoding: The encoding to modify in place.
            max_dist_to_edge: Maximum allowed white pixels between the last text
                column and the right edge of the last attended patch.

        Returns:
            The same (modified) encoding, for chaining.
        """
        for i in range(pixar_encoding.pixel_values.shape[0]):
            pixar_encoding = self._align_text_to_right_edge_at_i(i, pixar_encoding, max_dist_to_edge)
        return pixar_encoding

    def align_text_to_right_edge(self, pixar_encoding: PixarEncoding, max_dist_to_edge: int) -> PixarEncoding:
        """Shift each item's text right so trailing whitespace is at most ``max_dist_to_edge`` px.

        Useful with left padding: after shifting, leading all-white patches are
        masked out of ``attention_mask`` so the sequence effectively shortens.

        Args:
            pixar_encoding: The encoding to compact (a clone is modified).
            max_dist_to_edge: Maximum allowed white pixels between the last text
                column and the right edge of the last attended patch.

        Returns:
            A new encoding with right-aligned text and updated attention mask.

        Example::

            enc = processor.render(texts, padding_side="left")
            enc = processor.align_text_to_right_edge(enc, max_dist_to_edge=4)
        """
        pixar_encoding = pixar_encoding.clone()
        for i in range(pixar_encoding.pixel_values.shape[0]):
            pixar_encoding = self._align_text_to_right_edge_at_i(i, pixar_encoding, max_dist_to_edge)
        return pixar_encoding
