"""passthrough denoiser implementing the astro-denoise contract.

Replace the body of :meth:`denoise` with your algorithm. It receives a
2-D float array (one patch / region) and must return an array of the
same shape. Config keys arrive through :class:`PassthroughDenoiser`'s
constructor (forwarded from the CLI ``--config`` JSON).
"""

from __future__ import annotations

import numpy as np

from .interface import DenoiseMethod


class PassthroughDenoiser(DenoiseMethod):
    """TODO: describe the method and its parameters."""

    def __init__(self, **config):
        self.config = config

    def name(self) -> str:
        return "passthrough"

    def denoise(self, patch: np.ndarray) -> np.ndarray:
        patch = np.asarray(patch, dtype=np.float64)
        if patch.ndim != 2:
            raise ValueError(f"expected a 2-D patch, got shape {patch.shape}")
        # TODO: implement the actual denoising. The pass-through below
        # keeps the module runnable end-to-end from the first minute.
        return patch.copy()
