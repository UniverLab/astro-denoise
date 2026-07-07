"""CLI: evaluate a denoised image against a truth catalog.

Usage:
    python -m astro_denoise_metrics --input denoised.fits --truth truth.fits \
        --output results.json [--original clean.fits] [--radius 2.0]

The truth file is either a FITS binary table with pixel columns x, y and
flux (optionally mag), or a DESC DC2 ``truth_match`` Parquet catalog
(``ra``/``dec``/``flux_<band>``) — the latter requires ``--band`` and is
projected onto the input image's WCS. Results are written as structured
JSON and a summary table is printed to the console.
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
from .detect import background_rms, detect_sources, source_mask
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
)

COLUMN_ALIASES = {
    "x": ("x", "X", "x_pix", "xcentroid", "X_IMAGE"),
    "y": ("y", "Y", "y_pix", "ycentroid", "Y_IMAGE"),
    "flux": ("flux", "FLUX", "flux_true", "FLUX_AUTO"),
    "mag": ("mag", "MAG", "mag_true", "MAG_AUTO"),
}


def load_image_and_header(path):
    """Return (data, header) for the first 2-D image HDU in a FITS file."""
    with fits.open(path) as hdul:
        for hdu in hdul:
            if hdu.data is not None and np.asarray(hdu.data).ndim == 2:
                return np.asarray(hdu.data, dtype=np.float64), hdu.header.copy()
    raise ValueError(f"{path}: no 2-D image HDU found")


def load_image(path):
    """Return the first 2-D image HDU in a FITS file."""
    return load_image_and_header(path)[0]


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


def resolve_truth(args, denoised, header):
    """Load the truth catalog in pixel space, projecting DC2 Parquet via WCS."""
    if args.truth.suffix.lower() == ".parquet":
        if not args.band:
            raise ValueError("--band is required when --truth is a DC2 Parquet catalog")
        from astropy.wcs import WCS

        from .truth import load_dc2_truth, project_truth

        sky = load_dc2_truth(args.truth, args.band)
        return project_truth(sky, WCS(header), denoised.shape, mag_limit=args.mag_limit)
    return load_truth(args.truth)


def evaluate(args):
    denoised, header = load_image_and_header(args.input)
    truth = resolve_truth(args, denoised, header)
    thresh_rms = background_rms(load_image(args.thresh_image)) if args.thresh_image else None
    catalog = detect_sources(denoised, thresh=args.thresh, minarea=args.minarea, err=thresh_rms)

    results = {
        "module": "astro-denoise-metrics",
        "version": __version__,
        "input": str(args.input),
        "truth": str(args.truth),
        "parameters": {
            "radius": args.radius,
            "thresh": args.thresh,
            "minarea": args.minarea,
            "band": args.band,
            "mag_limit": args.mag_limit,
            "thresh_image": str(args.thresh_image) if args.thresh_image else None,
            "thresh_rms": thresh_rms,
        },
        "n_detected": int(len(catalog)),
        "n_truth": int(len(truth["x"])),
        "completeness": completeness(catalog, truth, args.radius),
        "purity": purity(catalog, truth, args.radius),
        "sky_sigma": sky_sigma(
            denoised, mask=source_mask(denoised, thresh=args.thresh, err=thresh_rms)
        ),
        "sharpness_sf": fourier_sharpness(denoised),
    }
    results["flux_snr"] = flux_snr(catalog, results["sky_sigma"])

    if args.thresh_sweep:
        thresholds = [float(t) for t in args.thresh_sweep.split(",")]
        results["detection_rate"] = detection_rate(
            denoised,
            thresholds,
            minarea=args.minarea,
            truth=truth,
            radius=args.radius,
            err=thresh_rms,
        )

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
        results["mse"] = mse(original, denoised)
        results["rmse"] = rmse(original, denoised)
        results["mae"] = mae(original, denoised)
        original_sf = fourier_sharpness(original)
        results["sharpness_sf_original"] = original_sf
        results["sharpness_ratio"] = (
            results["sharpness_sf"] / original_sf if original_sf > 0 else float("nan")
        )

    return results


def ssim_safe(original, denoised):
    from .metrics import ssim

    return ssim(original, denoised)


def print_summary(console, results):
    table = RichTable(title="astro-denoise-metrics results")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for key in (
        "n_detected",
        "n_truth",
        "completeness",
        "purity",
        "sky_sigma",
        "sharpness_sf",
        "flux_snr",
        "psnr",
        "ssim",
        "mse",
        "rmse",
        "mae",
    ):
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
    parser.add_argument("--input", type=Path, help="Denoised FITS image to evaluate")
    parser.add_argument(
        "--truth",
        type=Path,
        help="Truth catalog: FITS table (x, y, flux) or DC2 Parquet (needs --band)",
    )
    parser.add_argument(
        "--band",
        default=None,
        help="DC2 band (u/g/r/i/z/y) selecting flux_<band> in a Parquet truth catalog",
    )
    parser.add_argument(
        "--mag-limit",
        type=float,
        default=None,
        help="Drop Parquet truth sources fainter than this AB magnitude",
    )
    parser.add_argument("--output", type=Path, help="Path of the JSON results file")
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Describe every reported metric (what it measures, how to read it) and exit",
    )
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
        "--thresh-image",
        type=Path,
        default=None,
        help="Measure the background RMS that scales --thresh on this FITS image "
        "(e.g. the noisy input) instead of on --input, so the detection depth "
        "is comparable across denoising methods",
    )
    parser.add_argument(
        "--minarea", type=int, default=5, help="Minimum connected pixels per detection"
    )
    parser.add_argument(
        "--thresh-sweep",
        default=None,
        help="Comma-separated thresholds for a detection-rate sweep (e.g. '1,1.5,2,3,5')",
    )
    return parser


def print_explanations(console):
    from .explain import explain_lines

    table = RichTable(title="astro-denoise-metrics — qué mide cada métrica", show_lines=True)
    table.add_column("Métrica", style="bold", no_wrap=True)
    table.add_column("Qué mide", max_width=44)
    table.add_column("Cómo interpretarla", max_width=44)
    table.add_column("Referencia", max_width=28)
    for row in explain_lines():
        table.add_row(*row)
    console.print(table)


def main(argv=None):
    console = Console(stderr=True)
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.explain:
        print_explanations(Console())
        return 0
    if not args.input or not args.truth or not args.output:
        parser.error("--input, --truth and --output are required (or use --explain)")
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
