"""CLI: orchestrate denoising benchmarks over DC2 data.

Usage:
    # classic patch-pair mode
    python -m astro_denoise_bench --algo bm3d --data-dir /path/to/data \
        --output report.json [--config config.json] [--python /venv/bin/python]

    # curate the heterogeneous DC2 region block
    python -m astro_denoise_bench --curate --coadd-root data/coadd-... \
        --truth-root data/truth_match --curated-out data/curated

    # benchmark over the curated block with the real metrics engine
    python -m astro_denoise_bench --algo bm3d --manifest data/curated/manifest.json \
        --metrics-python ../astro-denoise-metrics/.venv/bin/python \
        --python ../astro-denoise-bm3d/.venv/bin/python --previews

    python -m astro_denoise_bench --list    # show discovered method modules
    python -m astro_denoise_bench --tui     # interactive dashboard
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from . import __version__
from .bench import discover_methods, run_benchmark, run_curated_benchmark
from .dc2 import curate, load_manifest


def print_report(console, report):
    table = Table(title=f"astro-denoise-bench — {report['algorithm']}")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("patches", str(report.get("n_patches", report.get("n_regions"))))
    for key, value in report["summary"].items():
        if key == "per_band":
            continue
        table.add_row(key, str(value))
    console.print(table)
    per_band = report["summary"].get("per_band")
    if per_band:
        band_table = Table(title="per band")
        band_table.add_column("band")
        for col in ("n_ok", "mean_completeness", "mean_purity", "mean_sky_sigma"):
            band_table.add_column(col, justify="right")
        for band, block in per_band.items():
            band_table.add_row(
                band,
                str(block.get("n_ok")),
                str(block.get("mean_completeness")),
                str(block.get("mean_purity")),
                str(block.get("mean_sky_sigma")),
            )
        console.print(band_table)


def emit(payload):
    payload.setdefault("module", "astro-denoise-bench")
    payload.setdefault("version", __version__)
    print(json.dumps(payload))


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m astro_denoise_bench",
        description="Orchestrate denoising benchmarks over FITS patch datasets.",
    )
    parser.add_argument("--algo", help="Algorithm name (e.g. bm3d, unet)")
    parser.add_argument(
        "--data-dir", type=Path, help="Directory with *_noisy.fits / *_clean.fits patches"
    )
    parser.add_argument(
        "--manifest", type=Path, help="Curated-block manifest.json (alternative to --data-dir)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("report.json"),
        help="Path of the JSON report (default: report.json)",
    )
    parser.add_argument(
        "--config", type=Path, default=None, help="Config JSON forwarded to the method module"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None, help="Where to write denoised patches"
    )
    parser.add_argument(
        "--python",
        default=None,
        help="Interpreter used to invoke the method module "
        "(e.g. the module's own .venv/bin/python)",
    )
    parser.add_argument(
        "--metrics-python",
        default=None,
        help="Interpreter of the astro-denoise-metrics venv (manifest mode)",
    )
    parser.add_argument(
        "--bands", default=None, help="Comma-separated band filter for manifest mode (e.g. 'z,r')"
    )
    parser.add_argument(
        "--mag-limit", type=float, default=26.0, help="Truth AB magnitude cut (manifest mode)"
    )
    parser.add_argument(
        "--thresh", type=float, default=1.5, help="Detection threshold forwarded to metrics"
    )
    parser.add_argument(
        "--previews", action="store_true", help="Render before/after/residual PNGs per region"
    )
    parser.add_argument(
        "--list", action="store_true", help="List discovered astro-denoise method modules"
    )
    parser.add_argument("--tui", action="store_true", help="Launch the interactive dashboard")

    curate_group = parser.add_argument_group("dataset curation (--curate)")
    curate_group.add_argument(
        "--curate", action="store_true", help="Build the curated DC2 region block"
    )
    curate_group.add_argument("--coadd-root", type=Path, help="DC2 coadd root (has deepCoadd/)")
    curate_group.add_argument("--truth-root", type=Path, help="Directory with truth_tract*.parquet")
    curate_group.add_argument("--curated-out", type=Path, help="Output directory for the block")
    curate_group.add_argument("--size", type=int, default=256, help="Cutout size in pixels")
    curate_group.add_argument(
        "--regions-per-image", type=int, default=3, help="Cutouts sampled per coadd image"
    )
    return parser


def run_curate_mode(console, args, parser):
    if not args.coadd_root or not args.truth_root or not args.curated_out:
        parser.error("--curate needs --coadd-root, --truth-root and --curated-out")
    bands = args.bands.split(",") if args.bands else None
    manifest = curate(
        args.coadd_root,
        args.truth_root,
        args.curated_out,
        bands=bands,
        size=args.size,
        regions_per_image=args.regions_per_image,
    )
    console.print(
        f"[green]curated[/green] {manifest['n_regions']} regions "
        f"({', '.join(manifest['bands'])}) -> {args.curated_out}/manifest.json"
    )
    emit(
        {
            "status": "ok",
            "manifest": str(Path(args.curated_out) / "manifest.json"),
            "n_regions": manifest["n_regions"],
            "bands": manifest["bands"],
        }
    )
    return 0


def main(argv=None):
    console = Console(stderr=True)
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.tui:
        from .tui import run_tui

        return run_tui(manifest_path=args.manifest)

    if args.list:
        methods = discover_methods()
        if methods:
            for name, ep in sorted(methods.items()):
                console.print(f"[bold]{name}[/bold] -> {ep.value}")
        else:
            console.print("no astro_denoise.methods entry points installed")
        emit({"status": "ok", "methods": sorted(methods)})
        return 0

    if args.curate:
        try:
            return run_curate_mode(console, args, parser)
        except Exception as exc:
            console.print(f"[red]error:[/red] {exc}")
            emit({"status": "error", "error": str(exc)})
            return 1

    if not args.algo or not (args.data_dir or args.manifest):
        parser.error("--algo with --data-dir or --manifest is required (or --list/--curate/--tui)")

    def progress(index, total, name):
        console.print(f"[{index}/{total}] {name}")

    try:
        if args.manifest:
            manifest = load_manifest(args.manifest)
            report = run_curated_benchmark(
                manifest,
                args.algo,
                out_dir=args.out_dir or args.manifest.parent,
                method_python=args.python,
                metrics_python=args.metrics_python,
                config=args.config,
                bands=args.bands.split(",") if args.bands else None,
                mag_limit=args.mag_limit,
                thresh=args.thresh,
                previews=args.previews,
                progress=progress,
            )
        else:
            report = run_benchmark(
                args.algo,
                args.data_dir,
                output_dir=args.out_dir,
                config=args.config,
                python=args.python,
                progress=progress,
            )
    except Exception as exc:
        console.print(f"[red]error:[/red] {exc}")
        emit({"status": "error", "error": str(exc)})
        return 1

    report["version"] = __version__
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print_report(console, report)
    emit({"status": "ok", "report": str(args.output), "summary": report["summary"]})
    return 0


if __name__ == "__main__":
    sys.exit(main())
