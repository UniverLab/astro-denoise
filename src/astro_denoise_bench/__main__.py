"""CLI: run a denoising benchmark over a dataset of FITS patches.

Usage:
    python -m astro_denoise_bench --algo bm3d --data-dir /path/to/data \
        --output report.json [--config config.json] [--python /venv/bin/python]

    python -m astro_denoise_bench --list    # show discovered method modules
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from . import __version__
from .bench import discover_methods, run_benchmark


def print_report(console, report):
    table = Table(title=f"astro-denoise-bench — {report['algorithm']}")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("patches", str(report["n_patches"]))
    for key, value in report["summary"].items():
        table.add_row(key, str(value))
    console.print(table)


def main(argv=None):
    console = Console(stderr=True)
    parser = argparse.ArgumentParser(
        prog="python -m astro_denoise_bench",
        description="Orchestrate denoising benchmarks over FITS patch datasets.",
    )
    parser.add_argument("--algo", help="Algorithm name (e.g. bm3d, unet)")
    parser.add_argument(
        "--data-dir", type=Path, help="Directory with *_noisy.fits / *_clean.fits patches"
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
        "--list", action="store_true", help="List discovered astro-denoise method modules"
    )
    args = parser.parse_args(argv)

    if args.list:
        methods = discover_methods()
        if methods:
            for name, ep in sorted(methods.items()):
                console.print(f"[bold]{name}[/bold] -> {ep.value}")
        else:
            console.print("no astro_denoise.methods entry points installed")
        print(
            json.dumps(
                {
                    "module": "astro-denoise-bench",
                    "version": __version__,
                    "status": "ok",
                    "methods": sorted(methods),
                }
            )
        )
        return 0

    if not args.algo or not args.data_dir:
        parser.error("--algo and --data-dir are required (or use --list)")

    def progress(index, total, name):
        console.print(f"[{index}/{total}] {name}")

    try:
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
        print(
            json.dumps(
                {
                    "module": "astro-denoise-bench",
                    "version": __version__,
                    "status": "error",
                    "error": str(exc),
                }
            )
        )
        return 1

    report["version"] = __version__
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print_report(console, report)
    print(
        json.dumps(
            {
                "module": "astro-denoise-bench",
                "version": __version__,
                "status": "ok",
                "report": str(args.output),
                "summary": report["summary"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
