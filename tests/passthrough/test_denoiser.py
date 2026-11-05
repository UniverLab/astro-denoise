import numpy as np
import pytest

from astro_denoise_passthrough import PassthroughDenoiser


def test_preserves_shape():
    rng = np.random.default_rng(0)
    patch = rng.normal(100.0, 2.0, (64, 64))
    out = PassthroughDenoiser().denoise(patch)
    assert out.shape == patch.shape


def test_rejects_non_2d():
    with pytest.raises(ValueError):
        PassthroughDenoiser().denoise(np.zeros(10))
