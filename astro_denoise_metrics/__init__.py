"""astro-denoise-metrics: scientific evaluation metrics for astronomical image denoising."""

__version__ = "0.1.0"

from .metrics import (
    completeness,
    purity,
    completeness_vs_mag,
    photometric_accuracy,
    psnr,
    ssim,
    sky_sigma,
)
from .detect import detect_sources

__all__ = [
    "completeness",
    "purity",
    "completeness_vs_mag",
    "photometric_accuracy",
    "psnr",
    "ssim",
    "sky_sigma",
    "detect_sources",
    "__version__",
]
