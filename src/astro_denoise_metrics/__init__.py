"""astro-denoise-metrics: scientific evaluation metrics for astronomical image denoising."""

__version__ = "0.2.0"

from .detect import detect_sources
from .metrics import (
    completeness,
    completeness_vs_mag,
    photometric_accuracy,
    psnr,
    purity,
    sky_sigma,
    ssim,
)
from .truth import load_dc2_truth, project_truth

__all__ = [
    "completeness",
    "purity",
    "completeness_vs_mag",
    "photometric_accuracy",
    "psnr",
    "ssim",
    "sky_sigma",
    "detect_sources",
    "load_dc2_truth",
    "project_truth",
    "__version__",
]
