from typing import Union, List, Tuple, Callable, ParamSpec, TypeVar
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


@dataclass
class PixarEncoding:
    pixel_values: torch.Tensor
    num_text_patches: List[int]
    sep_patches: List[List[int]]
    text: List[Union[str, Tuple[str, ...]]]


class PixarProcessor:
    def __init__(
        self, 
        font_file: str = 'GoNotoCurrent.ttf',
        font_size: int = 8,
        font_color: str = "black",
        background_color: str = "white",
        binary: bool = False,
        rgb: bool = True,
        dpi: int = 180,
        pad_size: int = 3,
        pixels_per_patch: int = 24,
        max_seq_length: int = 529,
        fallback_fonts_dir: str | None = None,
        patch_len: int = 1,
        contour_r: float = 0.0,
        contour_g: float = 0.0,
        contour_b: float = 0.0,
        contour_alpha: float = 0.7,
        contour_width: int = 1,
        device: Union[str, int] = 'cpu'
    ):
        self.font_file = font_file
        self.font_size = font_size
        self.font_color = font_color
        self.background_color = background_color
        self.rgb = rgb
        self.binary = binary
        self.dpi = dpi
        self.pad_size = pad_size
        self.pixels_per_patch = pixels_per_patch
        self.max_seq_length = max_seq_length
        self.fallback_fonts_dir = fallback_fonts_dir
        self.patch_len = patch_len
        self.contour_r = contour_r
        self.contour_g = contour_g
        self.contour_b = contour_b
        self.contour_alpha = contour_alpha
        self.contour_width = contour_width
        self.device = device

        assert max_seq_length % patch_len == 0, f"max_seq_length must be divisible by patch_len, but got {max_seq_length} and {patch_len}"

        self.renderer = PangoCairoTextRenderer(
            font_file,
            font_size,
            font_color,
            background_color,
            rgb,
            dpi,
            pad_size,
            pixels_per_patch,
            max_seq_length,
            fallback_fonts_dir,
            patch_len
        )

        self._to_pil = ToPILImage(mode="RGB")

    def _binary(self, pixel_values: torch.Tensor) -> torch.Tensor:
        val = pixel_values.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        return (val > 0.5).to(torch.int64)

    def __call__(self, text: Union[str, Tuple[str, ...], List[Union[str, Tuple[str, ...]]]]):
        return self.render(text)

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
    def convert_to_pil(self, pixar_encoding: PixarEncoding, square: bool = True, contour: bool = False) -> List[Image.Image]:
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
        images = self.convert_to_pil(pixar_encoding, square, contour)
        path_dir = Path(dir_path)
        if not path_dir.exists():
            path_dir.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(images):
            img.save(Path(dir_path) / f"{i}.png")

    @torch.no_grad()
    def render(self, text: Union[str, Tuple[str, ...], List[Union[str, Tuple[str, ...]]]]) -> PixarEncoding:
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

        if self.binary:
            val = pixel_values.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
            pixel_values = (val > 0.5).to(torch.int64)

        num_text_patches = [
            ceil((p.num_text_patches + 1) / self.patch_len) for p in rendered
        ]
        sep_patches = [
            self._cal_sep_patches(p.sep_patches) for p in rendered
        ]
        sep_patches = [list(set(sep)) for sep in sep_patches]
        for sep in sep_patches:
            sep.sort()

        if not isinstance(text, list):
            text = [text]

        return PixarEncoding(
            pixel_values=pixel_values.contiguous(), 
            num_text_patches=num_text_patches, 
            text=deepcopy(text), 
            sep_patches=sep_patches
        )

    def save_conf(self, dir_path: str):
        conf_file = Path(dir_path) / "pixar_processor_conf.json"
        conf = {
            "font_file": self.font_file,
            "font_size": self.font_size,
            "font_color": self.font_color,
            "background_color": self.background_color,
            "rgb": self.rgb,
            "binary": self.binary,
            "dpi": self.dpi,
            "pad_size": self.pad_size,
            "pixels_per_patch": self.pixels_per_patch,
            "max_seq_length": self.max_seq_length,
            "fallback_fonts_dir": self.fallback_fonts_dir,
            "patch_len": self.patch_len,
            "contour_r": self.contour_r,
            "contour_g": self.contour_g,
            "contour_b": self.contour_b,
            "contour_alpha": self.contour_alpha,
            'contour_width': self.contour_width,
            "device": self.device,
        }
        if not conf_file.parent.exists():
            conf_file.parent.mkdir(parents=True)
        with open(conf_file, 'w') as f:
            json.dump(conf, f, indent=4, ensure_ascii=False)

    @classmethod
    def load_conf(cls, dir_path: str) -> 'PixarProcessor':
        conf_file = Path(dir_path) / "pixar_processor_conf.json"
        with open(conf_file, 'r') as f:
            conf = json.load(f)
        return cls(**conf)

    def slice(self, pixar_encoding: PixarEncoding, start: int, end: int) -> PixarEncoding:
        block_len = self.pixels_per_patch * self.patch_len
        # N C H W
        pixel_values = pixar_encoding.pixel_values[:, :, :, start*block_len:end*block_len]
        num_text_patches = [
            min(n - start, end - start) for n in pixar_encoding.num_text_patches
        ]
        text = deepcopy(pixar_encoding.text)
        sep_patches = [[
                s - start for s in seq if s >= start and s < end
            ] for seq in pixar_encoding.sep_patches
        ]

        return PixarEncoding(
            pixel_values=pixel_values.contiguous(), 
            num_text_patches=num_text_patches, 
            text=deepcopy(text), 
            sep_patches=sep_patches
        )

    def insert(self, pixar_encoding: PixarEncoding, start: int, end: int, inserted: PixarEncoding) -> PixarEncoding:
        block_len = self.pixels_per_patch * self.patch_len
        # N C H W
        pixel_values = pixar_encoding.pixel_values.clone()
        pixel_values[:, :, :, start*block_len:end*block_len] = inserted.pixel_values
        num_text_patches = [
            max(n1, start + n2) for n1, n2 in zip(pixar_encoding.num_text_patches, inserted.num_text_patches)
        ]
        text = deepcopy(pixar_encoding.text)
        sep_patches = [[
                s for s in seq1 if s < start or s >= end
            ] + [
                s + start for s in seq2 if s + start >= start and s + start < end
            ] for seq1, seq2 in zip(pixar_encoding.sep_patches, inserted.sep_patches)
        ]
        for seq in sep_patches:
            seq.sort()

        return PixarEncoding(
            pixel_values=pixel_values.contiguous(), 
            num_text_patches=num_text_patches, 
            text=deepcopy(text), 
            sep_patches=sep_patches
        )

    def reduce_white_space(self, pixar_encoding: PixarEncoding) -> PixarEncoding:
        # TODO:
