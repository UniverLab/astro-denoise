"""Visual before/after previews of denoising runs.

Renders a three-panel PNG (input, denoised, residual) with the
astronomical z-scale + asinh stretch shared between input and output,
so the visual comparison is faithful: what disappears in the residual
panel is exactly what the method removed.
"""

from __future__ import annotations

import matplotlib
import numpy as np
from astropy.io import fits
from astropy.visualization import AsinhStretch, ImageNormalize, ZScaleInterval

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _first_image(path):
    with fits.open(path) as hdul:
        for hdu in hdul:
            if hdu.data is not None and np.asarray(hdu.data).ndim == 2:
                return np.asarray(hdu.data, dtype=np.float64)
    raise ValueError(f"{path}: no 2-D image HDU found")


def render_preview(before_path, after_path, out_path, title=None):
    """Write a before/after/residual PNG; returns the output path str."""
    before = _first_image(before_path)
    after = _first_image(after_path)
    residual = before - after

    norm = ImageNormalize(before, interval=ZScaleInterval(), stretch=AsinhStretch())
    res_norm = ImageNormalize(residual, interval=ZScaleInterval())

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), constrained_layout=True)
    panels = (
        (before, norm, "input"),
        (after, norm, "denoised"),
        (residual, res_norm, "residual (input - denoised)"),
    )
    for ax, (data, panel_norm, label) in zip(axes, panels, strict=True):
        ax.imshow(data, origin="lower", cmap="gray", norm=panel_norm)
        ax.set_title(label, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    if title:
        fig.suptitle(title, fontsize=11)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return str(out_path)
