"""astro-denoise-metrics: scientific evaluation metrics for astronomical image denoising."""

__version__ = "0.3.0"

from .detect import detect_sources
from .metrics import (
    completeness,
    completeness_vs_mag,
    detection_rate,
    flux_snr,
    fourier_sharpness,
    mae,
    mse,
    photometric_accuracy,
    psnr,
    purity,
    rmse,
    sky_sigma,
    ssim,
)
from .truth import load_dc2_truth, project_truth

__all__ = [
    "completeness",
    "purity",
    "completeness_vs_mag",
    "photometric_accuracy",
    "detection_rate",
    "flux_snr",
    "fourier_sharpness",
    "psnr",
    "ssim",
    "mse",
    "rmse",
    "mae",
    "sky_sigma",
    "detect_sources",
    "load_dc2_truth",
    "project_truth",
    "__version__",
]
