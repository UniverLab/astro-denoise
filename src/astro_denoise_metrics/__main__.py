"""CLI: evaluate a denoised image against a truth catalog.

Usage:
    python -m astro_denoise_metrics --input denoised.fits --truth truth.fits \
        --output results.json [--original clean.fits] [--radius 2.0]

The truth file is a FITS binary table with columns x, y and flux (and
optionally mag). Results are written as structured JSON and a summary
table is printed to the console.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table
from rich.console import Console
from rich.table import Table as RichTable

from . import __version__
from .detect import detect_sources, source_mask
from .metrics import (
    completeness,
    completeness_vs_mag,
    photometric_accuracy,
    psnr,
    purity,
    sky_sigma,
)

COLUMN_ALIASES = {
    "x": ("x", "X", "x_pix", "xcentroid", "X_IMAGE"),
    "y": ("y", "Y", "y_pix", "ycentroid", "Y_IMAGE"),
    "flux": ("flux", "FLUX", "flux_true", "FLUX_AUTO"),
    "mag": ("mag", "MAG", "mag_true", "MAG_AUTO"),
}


def load_image(path):
    """Return the first 2-D image HDU in a FITS file."""
    with fits.open(path) as hdul:
        for hdu in hdul:
            if hdu.data is not None and np.asarray(hdu.data).ndim == 2:
                return np.asarray(hdu.data, dtype=np.float64)
    raise ValueError(f"{path}: no 2-D image HDU found")


def load_truth(path):
    """Load a truth catalog table and normalize its column names."""
    table = Table.read(path)
    out = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in table.colnames:
                out[canonical] = np.asarray(table[alias], dtype=np.float64)
                break
    if "x" not in out or "y" not in out:
        raise ValueError(f"{path}: truth catalog needs x/y columns; found {table.colnames}")
    return out


def _nan_to_none(obj):
    """Make results JSON-safe: NaN/inf become null."""
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def default_mag_bins(truth):
    if "mag" in truth:
        mags = truth["mag"]
    elif "flux" in truth:
        flux = truth["flux"]
        with np.errstate(divide="ignore", invalid="ignore"):
            mags = -2.5 * np.log10(np.where(flux > 0, flux, np.nan))
    else:
        return None
    mags = mags[np.isfinite(mags)]
    if len(mags) == 0:
        return None
    return np.linspace(np.floor(mags.min()), np.ceil(mags.max()), 11)


def evaluate(args):
    denoised = load_image(args.input)
    truth = load_truth(args.truth)
    catalog = detect_sources(denoised, thresh=args.thresh, minarea=args.minarea)

    results = {
        "module": "astro-denoise-metrics",
        "version": __version__,
        "input": str(args.input),
        "truth": str(args.truth),
        "parameters": {
            "radius": args.radius,
            "thresh": args.thresh,
            "minarea": args.minarea,
        },
        "n_detected": int(len(catalog)),
        "n_truth": int(len(truth["x"])),
        "completeness": completeness(catalog, truth, args.radius),
        "purity": purity(catalog, truth, args.radius),
        "sky_sigma": sky_sigma(denoised, mask=source_mask(denoised, thresh=args.thresh)),
    }

    if "flux" in truth:
        results["photometric_accuracy"] = photometric_accuracy(catalog, truth, radius=args.radius)
        bins = default_mag_bins(truth)
        if bins is not None:
            results["completeness_vs_mag"] = completeness_vs_mag(
                catalog, truth, bins, radius=args.radius
            )

    if args.original:
        original = load_image(args.original)
        results["psnr"] = psnr(original, denoised)
        results["ssim"] = ssim_safe(original, denoised)

    return results


def ssim_safe(original, denoised):
    from .metrics import ssim

    return ssim(original, denoised)


def print_summary(console, results):
    table = RichTable(title="astro-denoise-metrics results")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for key in ("n_detected", "n_truth", "completeness", "purity", "sky_sigma", "psnr", "ssim"):
        if key in results and results[key] is not None:
            value = results[key]
            table.add_row(key, f"{value:.4f}" if isinstance(value, float) else str(value))
    phot = results.get("photometric_accuracy")
    if phot:
        table.add_row("flux_bias", f"{phot['flux_bias']:.4f}")
        table.add_row("flux_scatter", f"{phot['flux_scatter']:.4f}")
    console.print(table)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m astro_denoise_metrics",
        description="Scientific metrics for denoised astronomical images.",
    )
    parser.add_argument("--input", required=True, type=Path, help="Denoised FITS image to evaluate")
    parser.add_argument(
        "--truth",
        required=True,
        type=Path,
        help="FITS truth catalog (binary table with x, y, flux)",
    )
    parser.add_argument("--output", required=True, type=Path, help="Path of the JSON results file")
    parser.add_argument(
        "--original", type=Path, default=None, help="Noise-free reference image for PSNR/SSIM"
    )
    parser.add_argument(
        "--radius", type=float, default=2.0, help="Matching radius in pixels (default: 2.0)"
    )
    parser.add_argument(
        "--thresh", type=float, default=1.5, help="Detection threshold in background-RMS units"
    )
    parser.add_argument(
        "--minarea", type=int, default=5, help="Minimum connected pixels per detection"
    )
    return parser


def main(argv=None):
    console = Console(stderr=True)
    args = build_parser().parse_args(argv)
    try:
        results = _nan_to_none(evaluate(args))
    except Exception as exc:  # surface a structured error for orchestrators
        print(
            json.dumps(
                {
                    "module": "astro-denoise-metrics",
                    "version": __version__,
                    "status": "error",
                    "error": str(exc),
                }
            )
        )
        console.print(f"[red]error:[/red] {exc}")
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print_summary(console, results)
    print(
        json.dumps(
            {
                "module": "astro-denoise-metrics",
                "version": __version__,
                "status": "ok",
                "output": str(args.output),
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
