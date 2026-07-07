"""CLI: denoise a FITS image following the astro-denoise bench protocol.

Usage:
    python -m astro_denoise_passthrough --input input.fits --output output.fits \
        [--config config.json]

stdout emits a single JSON status line consumed by the orchestrator:
    {"algorithm": "passthrough", "version": "0.1.0", "status": "ok"}
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from astropy.io import fits
from rich.console import Console

from . import __version__
from .denoiser import PassthroughDenoiser


def load_config(path):
    if path is None:
        return {}
    return json.loads(Path(path).read_text())


def find_image_hdu(hdul):
    for index, hdu in enumerate(hdul):
        if hdu.data is not None and np.asarray(hdu.data).ndim == 2:
            return index
    raise ValueError("no 2-D image HDU found in input FITS")


def main(argv=None):
    console = Console(stderr=True)
    parser = argparse.ArgumentParser(
        prog="python -m astro_denoise_passthrough",
        description="passthrough denoising method for the astro-denoise benchmark",
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        denoiser = PassthroughDenoiser(**config)

        with fits.open(args.input) as hdul:
            image_hdu = find_image_hdu(hdul)
            image = np.asarray(hdul[image_hdu].data, dtype=np.float64)
            header = hdul[image_hdu].header.copy()

        start = time.perf_counter()
        denoised = denoiser.denoise(image)
        elapsed = time.perf_counter() - start

        header["HISTORY"] = f"astro-denoise-passthrough {__version__}"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fits.PrimaryHDU(data=denoised.astype(np.float32), header=header).writeto(
            args.output, overwrite=True
        )
        console.print(f"[green]ok[/green] {args.input.name} -> {args.output.name}")
        print(
            json.dumps(
                {
                    "algorithm": "passthrough",
                    "version": __version__,
                    "status": "ok",
                    "elapsed_s": round(elapsed, 4),
                }
            )
        )
        return 0
    except Exception as exc:
        console.print(f"[red]error:[/red] {exc}")
        print(
            json.dumps(
                {
                    "algorithm": "passthrough",
                    "version": __version__,
                    "status": "error",
                    "error": str(exc),
                }
            )
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
