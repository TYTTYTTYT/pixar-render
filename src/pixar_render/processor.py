"""
Processor for Pixar Render
"""
from typing import Union, List, Tuple, Callable, ParamSpec, TypeVar, Literal
import json
from pathlib import Path
from math import ceil, sqrt
from copy import deepcopy
from dataclasses import dataclass
import functools
import inspect

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
    pixel_values: torch.Tensor
    attention_mask: torch.Tensor
    sep_patches: List[List[int]]

    def to(self, device: Union[str, int]) -> 'PixarEncoding':
        return PixarEncoding(
            pixel_values=self.pixel_values.to(device),
            attention_mask=self.attention_mask.to(device),
            sep_patches=deepcopy(self.sep_patches),
        )

    def clone(self) -> 'PixarEncoding':
        return PixarEncoding(
            pixel_values=self.pixel_values.clone(),
            attention_mask=self.attention_mask.clone(),
            sep_patches=deepcopy(self.sep_patches),
        )

    def __getitem__(self, index: slice | int) -> 'PixarEncoding':
        if isinstance(index, int):
            index = slice(index, index+1)
        return PixarEncoding(
            pixel_values=self.pixel_values[index],
            attention_mask=self.attention_mask[index],
            sep_patches=self.sep_patches[index],
        )


class PixarProcessor:
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
        """
        Initializes the PixarProcessor.

        Args:
            font_file (str): Name of the font file. If you want to use a custom font,
                you need to put the font file in the `resources/fonts` directory.
            font_size (int): Font size.
            font_color (str): Font color.
            background_color (str): Background color.
            binary (bool): Whether to binarize the output image.
            rgb (bool): Whether to render in RGB.
            dpi (int): Dots per inch.
            pad_size (int): Padding size.
            pixels_per_patch (int): Number of pixels per patch.
            max_seq_length (int): Maximum sequence length.
            fallback_fonts_dir (str | None): Directory for fallback fonts.
            patch_len (int): Patch length.
            contour_r (float): Red component of the contour color.
            contour_g (float): Green component of the contour color.
            contour_b (float): Blue component of the contour color.
            contour_alpha (float): Alpha component of the contour color.
            contour_width (int): Width of the contour.
            device (Union[str, int]): Device to use for processing ('cpu' or GPU index).
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

    def _binary(self, pixel_values: torch.Tensor) -> torch.Tensor:
        val = pixel_values.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        return (val > 0.5).to(torch.float)

    def __call__(
        self, 
        text: Union[str, Tuple[str, ...], List[Union[str, Tuple[str, ...]]]],
        padding_side: Literal['left', 'right'] | None = None,
        truncate: bool | None = None,
        add_eos: bool | None = None
    ) -> PixarEncoding:
        return self.render(text, padding_side, truncate, add_eos)

    def _cal_sep_patches(self, sep_patches: List[int]) -> List[int]:
        sep_idxes = []
        for n in sep_patches:
            idx = n / self.patch_len / self.pixels_per_patch
            if float(int(idx)) == idx:
                sep_idxes.append(int(idx))
        return sep_idxes

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
        """
        Converts a PixarEncoding object to a list of PIL Images.

        Args:
            pixar_encoding (PixarEncoding): The PixarEncoding to convert.
            square (bool): Whether to reshape the image into a square. Defaults to True.
            contour (bool): Whether to add a contour to the image. Defaults to False.

        Returns:
            List[Image.Image]: A list of converted PIL Images.
        """
        pixel_values = pixar_encoding.pixel_values
        if contour:
            pixel_values = self._add_contour(pixel_values)
        if square:
            pixel_values = self._squarelize(pixel_values)
        pixel_values = pixel_values * 255
        pixel_values = pixel_values.to(torch.uint8)
        images = [self._to_pil(p) for p in pixel_values]
        return images

    def save_as_images(self, pixar_encoding: PixarEncoding, dir_path: str, square: bool = True, contour: bool = False):
        """
        Saves the images from a PixarEncoding object to a directory.

        Args:
            pixar_encoding (PixarEncoding): The PixarEncoding object containing the images.
            dir_path (str): The directory path to save the images to.
            square (bool, optional): Whether to save the images as squares. Defaults to True.
            contour (bool, optional): Whether to add a contour to the images before saving. Defaults to False.
        """
        images = self.convert_to_pil(pixar_encoding, square, contour)
        path_dir = Path(dir_path)
        if not path_dir.exists():
            path_dir.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(images):
            img.save(Path(dir_path) / f"{i}.png")

    def render(
        self, 
        text: Union[str, Tuple[str, ...], List[Union[str, Tuple[str, ...]]]],
        padding_side: Literal['left', 'right'] | None = None,
        truncate: bool | None = None,
        add_eos: bool | None = None
    ) -> PixarEncoding:
        """
        Renders the input text into a PixarEncoding.

        Args:
            text (Union[str, Tuple[str, ...], List[Union[str, Tuple[str, ...]]]]): 
                The text to render. It can be a single string, a tuple of strings, 
                or a list of strings/tuples.
            padding_side (Literal['left', 'right'], optional): 
                The side to pad the text on. Defaults to None.
            truncate (bool, optional): 
                Whether to truncate the text if it exceeds the maximum number of patches. Defaults to None.
            add_eos (bool, optional): 
                Whether to add an end-of-sequence token to the text. Defaults to None.

        Returns:
            PixarEncoding: The rendered text as a PixarEncoding object, containing pixel values and patch information.
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

        pixel_values = torch.stack([torch.tensor(p.pixel_values.copy()) for p in rendered], dim=0)
        pixel_values = pixel_values.to(torch.float32).to(self.device) / 255

        # change the channel dimension to the second dimension to fit the Conv2d operator
        if self.rgb:
            pixel_values = pixel_values.permute(0, 3, 1, 2)
        else:
            # we repeat values 3 times to fit the Conv2d operator
            pixel_values = pixel_values.unsqueeze(1)
            pixel_values = pixel_values.repeat(1, 3, 1, 1)

        # dimension of pixel_values: [batch_size, channels, height, width]
        if self.binary:
            val = pixel_values.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
            pixel_values = (val > 0.5).to(torch.float)

        num_text_patches = [
            ceil((p.num_text_patches + 1) / self.patch_len) for p in rendered
        ]
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
                    pixel_values[idx, :, :, begin_idx:end_idx] = 1
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

    def slice(self, pixar_encoding: PixarEncoding, start: int, end: int) -> PixarEncoding:
        """
        Slices a PixarEncoding object to extract a sub-sequence of patches.

        Args:
            pixar_encoding (PixarEncoding): The PixarEncoding to slice.
            start (int): The starting patch index (inclusive).
            end (int): The ending patch index (exclusive).

        Returns:
            PixarEncoding: A new PixarEncoding object representing the sliced portion.
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
        """
        Inserts one PixarEncoding into another within a specified patch range.

        Args:
            pixar_encoding (PixarEncoding): The base PixarEncoding object to be modified.
            start (int): The starting patch index (inclusive) where the insertion begins.
            end (int): The ending patch index (exclusive) where the insertion ends.
            inserted (PixarEncoding): The PixarEncoding object to insert into the base encoding.

        Returns:
            PixarEncoding: A new PixarEncoding object with the `inserted` encoding placed
                           within the specified range of the original encoding.
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
        """
        Appends one PixarEncoding to the end of another.
        Args:
            pixar_encoding (PixarEncoding): The base PixarEncoding object to be modified.
            inserted (PixarEncoding): The PixarEncoding object to append to the base encoding.
        """
        pixel_values = torch.cat([pixar_encoding.pixel_values, inserted.pixel_values], dim=-1)
        attention_mask = torch.cat([pixar_encoding.attention_mask, inserted.attention_mask], dim=-1)
        seq_patches = deepcopy(pixar_encoding.sep_patches)
        l = pixar_encoding.pixel_values.shape[-1] // self._block_width
        for idx, seq in enumerate(inserted.sep_patches):
            seq_patches[idx].extend([s + l for s in seq])

        return PixarEncoding(pixel_values, attention_mask, seq_patches)

    def _align_text_to_right_edge_at_i(self, i: int, pixar_encoding: PixarEncoding, max_dist_to_edge: int) -> PixarEncoding:
        """
        Aligns the text in the PixarEncoding object to the right edge at the specified batch index.

        Args:
            i (int): The batch index where the text alignment should occur.
            pixar_encoding (PixarEncoding): The PixarEncoding object to be modified.
            max_dist_to_edge (int): The maximum number of white pixels from the right edge.

        Returns:
            PixarEncoding: A new PixarEncoding object with the text aligned to the right edge at index `i`.
        """
        # C, H, W
        pixel_values = pixar_encoding.pixel_values[i].clone()
        right_edge_patch_idx = int(pixar_encoding.attention_mask[i].nonzero().max().item())
        if right_edge_patch_idx in pixar_encoding.sep_patches[i]:
            right_edge_patch_idx -= 1

        right_edge_pixel_idx = (right_edge_patch_idx + 1) * self._block_width - 1
        current_column_idx = right_edge_pixel_idx
        while pixel_values[:, :, current_column_idx].mean().item() == 1.0 and current_column_idx > 0:
            current_column_idx -= 1

        dist_to_edge = right_edge_pixel_idx - current_column_idx
        if current_column_idx == 0 or dist_to_edge <= max_dist_to_edge:
            return pixar_encoding

        dist_to_move = dist_to_edge if dist_to_edge <= max_dist_to_edge else dist_to_edge - max_dist_to_edge

        num_text_pixel = current_column_idx + 1
        pixar_encoding.pixel_values[i, :, :, dist_to_move:dist_to_move+num_text_pixel] = pixel_values[:, :, :num_text_pixel]
        pixar_encoding.pixel_values[i, :, :, :dist_to_move] = 1.0

        # scan for begining white patches and set their attention mask to 0
        _, _, W = pixar_encoding.pixel_values[i].shape
        num_blocks = W // self._block_width
        for j in range(num_blocks):
            start = j * self._block_width
            end = start + self._block_width
            if pixar_encoding.pixel_values[i, :, :, start:end].mean().item() == 1.0:
                pixar_encoding.attention_mask[i, j] = 0
            else:
                break

        return pixar_encoding

    def align_text_to_right_edge_(self, pixar_encoding: PixarEncoding, max_dist_to_edge: int) -> PixarEncoding:
        """
        Aligns the text in the PixarEncoding object to the right edge for each batch.

        Args:
            pixar_encoding (PixarEncoding): The PixarEncoding object to be modified.
            max_white_space (int): The maximum number of white pixels from the right edge.

        Returns:
            PixarEncoding: A new PixarEncoding object with the text aligned to the right edge for each batch.
        """
        for i in range(pixar_encoding.pixel_values.shape[0]):
            pixar_encoding = self._align_text_to_right_edge_at_i(i, pixar_encoding, max_dist_to_edge)
        return pixar_encoding

    def align_text_to_right_edge(self, pixar_encoding: PixarEncoding, max_dist_to_edge: int) -> PixarEncoding:
        """
        Aligns the text in the PixarEncoding object to the right edge for each batch.

        Args:
            pixar_encoding (PixarEncoding): The PixarEncoding object to be modified.
            max_white_space (int): The maximum number of white pixels from the right edge.

        Returns:
            PixarEncoding: A new PixarEncoding object with the text aligned to the right edge for each batch.
        """
        pixar_encoding = pixar_encoding.clone()
        for i in range(pixar_encoding.pixel_values.shape[0]):
            pixar_encoding = self._align_text_to_right_edge_at_i(i, pixar_encoding, max_dist_to_edge)
        return pixar_encoding
