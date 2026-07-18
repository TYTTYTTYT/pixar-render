"""Tests for content-sized rendering and packed windows (0.5.0).

Run:  python tests/test_packed_window.py

Covers:
  1. channels=1 batch equals the default path's first channel (bit-identical)
  2. channels=1 is opt-in: default output shape/dtype unchanged
  3. _render_text_fast pixels match the canvas-sized path over its width
  4. packed window: documents laid out in order, separated by EOS patches
  5. packed window: sep_patches give exact per-document block boundaries
  6. packed window: overflow is clipped at the window edge, never wrapped
  7. content_sized is off by default (old behaviour preserved)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pixar_render import PixarProcessor  # noqa: E402
from pixar_render.pangocairo_render import PangoCairoTextRenderer  # noqa: E402

TEXTS = [
    'The quick brown fox jumps over the lazy dog.',
    'Bonjour le monde, ceci est un test de rendu.',
    'Kurz.',
]


def make_processor(**kw):
    p = PixarProcessor(
        font_size=8, pixels_per_patch=16, max_seq_length=64, patch_len=1, **kw
    )
    p.binary = False
    p.rgb = False
    p.renderer.rgb = False
    return p


def test_channels1_matches_default():
    p = make_processor()
    a = p.render(TEXTS, truncate=True, add_eos=True, padding_side='right')
    b = p.render(TEXTS, truncate=True, add_eos=True, padding_side='right', channels=1)
    assert b.pixel_values.shape[1] == 1, b.pixel_values.shape
    ref = a.pixel_values[:, :1] if a.pixel_values.ndim == 4 else a.pixel_values
    assert np.array_equal(np.asarray(ref), np.asarray(b.pixel_values)), \
        'channels=1 fast path must be bit-identical to the default path'
    print('PASS 1: channels=1 bit-identical')


def test_channels_default_unchanged():
    p = make_processor()
    a = p.render(TEXTS, truncate=True, add_eos=True, padding_side='right')
    assert str(a.pixel_values.dtype).endswith('uint8'), a.pixel_values.dtype
    assert a.pixel_values.shape[0] == len(TEXTS)
    print(f'PASS 2: default path shape {tuple(a.pixel_values.shape)} unchanged')


def test_render_text_fast_matches():
    slow = PangoCairoTextRenderer(font_size=8, pixels_per_patch=16,
                                  max_seq_length=64, patch_len=1)
    fast = PangoCairoTextRenderer(font_size=8, pixels_per_patch=16,
                                  max_seq_length=64, patch_len=1,
                                  content_sized=True)
    for text in TEXTS:
        a = slow(text)
        b = fast(text)
        w = b.pixel_values.shape[1]
        assert np.array_equal(np.asarray(a.pixel_values)[:, :w],
                              np.asarray(b.pixel_values)), \
            f'content-sized render differs for {text!r}'
        assert a.num_text_patches == b.num_text_patches
    print('PASS 3: content-sized single render bit-identical over its width')


def test_packed_window_layout():
    r = PangoCairoTextRenderer(font_size=8, pixels_per_patch=16,
                               max_seq_length=64, patch_len=1,
                               content_sized=True)
    enc = r(tuple(TEXTS))
    pp = r.pixels_per_patch
    assert len(enc.sep_patches) == len(TEXTS), \
        f'expected one EOS per doc, got {len(enc.sep_patches)}'
    assert enc.sep_patches == sorted(enc.sep_patches), 'EOS offsets must ascend'
    for sp in enc.sep_patches:
        assert np.all(np.asarray(enc.pixel_values)[:, sp:sp + pp] == 0), \
            'EOS patch must be solid'
    print(f'PASS 4: packed window, {len(enc.sep_patches)} EOS separators in order')


def test_packed_window_boundaries_are_exact():
    r = PangoCairoTextRenderer(font_size=8, pixels_per_patch=16,
                               max_seq_length=64, patch_len=1,
                               content_sized=True)
    enc = r(tuple(TEXTS))
    pp = r.pixels_per_patch
    blocks = [sp // pp for sp in enc.sep_patches]
    assert len(set(blocks)) == len(blocks), 'two docs share a boundary block'
    assert max(blocks) < r.max_seq_length
    # content ends at the last EOS; everything past it is white padding
    end = enc.sep_patches[-1] + pp
    assert np.all(np.asarray(enc.pixel_values)[:, end:] == 255), \
        'window tail must be white padding'
    print(f'PASS 5: exact boundaries at blocks {blocks}')


def test_packed_window_clips_overflow():
    r = PangoCairoTextRenderer(font_size=8, pixels_per_patch=16,
                               max_seq_length=48, patch_len=1,   # fits ~2 docs
                               content_sized=True)
    many = tuple(TEXTS * 6)                                       # cannot all fit
    enc = r(many)
    W = r.max_pixels_len
    assert np.asarray(enc.pixel_values).shape[1] == W, 'window must stay fixed width'
    assert 0 < len(enc.sep_patches) < len(many), \
        f'expected a partial fit, got {len(enc.sep_patches)}/{len(many)}'
    assert all(sp + r.pixels_per_patch <= W for sp in enc.sep_patches)
    # the clipped document keeps its pixels but gets no EOS
    assert enc.num_text_patches <= r.max_seq_length
    print(f'PASS 6: overflow clipped ({len(enc.sep_patches)}/{len(many)} docs fit)')


def test_content_sized_off_by_default():
    r = PangoCairoTextRenderer(font_size=8, pixels_per_patch=16,
                               max_seq_length=64, patch_len=1)
    assert r.content_sized is False
    enc = r(TEXTS[0])
    assert np.asarray(enc.pixel_values).shape[1] == r.max_pixels_len, \
        'default path must still render the full canvas width'
    print('PASS 7: content_sized defaults off')


if __name__ == '__main__':
    test_channels1_matches_default()
    test_channels_default_unchanged()
    test_render_text_fast_matches()
    test_packed_window_layout()
    test_packed_window_boundaries_are_exact()
    test_packed_window_clips_overflow()
    test_content_sized_off_by_default()
    print('ALL PASS')
