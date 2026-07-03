"""DC2 dataset curation: heterogeneous sky-region cutout blocks.

Builds the curated evaluation dataset of the platform: WCS-correct
cutouts sampled from every available DC2 coadd (tract x patch x band),
each paired with its variance plane and truth-catalog reference. The
resulting ``manifest.json`` is the reproducible contract consumed by
the benchmark runner: same manifest, same regions, same metrics.

Coadd layout expected (DESC DC2 ``deepCoadd``)::

    <coadd_root>/deepCoadd/<band>/<tract>/<patch>.fits

where each FITS stores the image in HDU 1 and the variance plane in
HDU 3. Truth catalogs are per-tract Parquet files named
``truth_tract<tract>.parquet`` under ``truth_root``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS

IMAGE_HDU = 1
VARIANCE_HDU = 3


def discover_coadds(coadd_root, bands=None):
    """List available coadd images as dicts of band/tract/patch/path."""
    root = Path(coadd_root) / "deepCoadd"
    found = []
    if not root.is_dir():
        return found
    for band_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if bands and band_dir.name not in bands:
            continue
        for tract_dir in sorted(p for p in band_dir.iterdir() if p.is_dir()):
            for path in sorted(tract_dir.glob("*.fits")):
                if path.stem.endswith("_nImage"):
                    continue
                found.append(
                    {
                        "band": band_dir.name,
                        "tract": int(tract_dir.name),
                        "patch": path.stem,
                        "path": str(path),
                    }
                )
    return found


def grid_positions(shape, size, n):
    """Deterministic, spread-out cutout centers inside ``shape``.

    Centers are placed along the two diagonals at evenly spaced
    fractions, keeping a half-cutout margin from the borders, so the
    same call always yields the same heterogeneous sampling.
    """
    height, width = shape
    margin_y, margin_x = size / 2, size / 2
    positions = []
    for k in range(n):
        frac = (k + 1) / (n + 1)
        fy = frac if k % 2 == 0 else 1 - frac
        x = margin_x + frac * (width - 2 * margin_x)
        y = margin_y + fy * (height - 2 * margin_y)
        positions.append((float(x), float(y)))
    return positions


def _region_id(entry, index):
    patch = entry["patch"].replace(",", "")
    return f"t{entry['tract']}-p{patch}-{entry['band']}-r{index}"


def cut_region(coadd_path, center, size):
    """Return (image cutout, variance cutout or None, WCS header)."""
    with fits.open(coadd_path) as hdul:
        image = np.asarray(hdul[IMAGE_HDU].data, dtype=np.float32)
        wcs = WCS(hdul[IMAGE_HDU].header)
        cut = Cutout2D(image, position=center, size=size, wcs=wcs)
        variance = None
        if len(hdul) > VARIANCE_HDU and hdul[VARIANCE_HDU].data is not None:
            var_plane = np.asarray(hdul[VARIANCE_HDU].data, dtype=np.float32)
            if var_plane.shape == image.shape:
                variance = var_plane[cut.slices_original]
    return cut.data, variance, cut.wcs.to_header()


def curate(
    coadd_root,
    truth_root,
    out_dir,
    bands=None,
    size=256,
    regions_per_image=3,
):
    """Build the curated heterogeneous region block and its manifest.

    Writes one FITS per region (image HDU with WCS + VARIANCE extension
    when the coadd provides one) under ``<out_dir>/regions/`` and a
    ``manifest.json`` describing every region. Returns the manifest.
    """
    out_dir = Path(out_dir)
    regions_dir = out_dir / "regions"
    regions_dir.mkdir(parents=True, exist_ok=True)
    truth_root = Path(truth_root)

    coadds = discover_coadds(coadd_root, bands=bands)
    if not coadds:
        raise FileNotFoundError(f"no coadd images found under {coadd_root}")

    regions = []
    for entry in coadds:
        with fits.open(entry["path"]) as hdul:
            shape = hdul[IMAGE_HDU].data.shape
        for index, center in enumerate(grid_positions(shape, size, regions_per_image)):
            region_id = _region_id(entry, index)
            image, variance, header = cut_region(entry["path"], center, size)
            hdus = [fits.PrimaryHDU(data=image, header=header)]
            if variance is not None:
                hdus.append(fits.ImageHDU(data=variance, name="VARIANCE"))
            rel_path = f"regions/{region_id}.fits"
            fits.HDUList(hdus).writeto(out_dir / rel_path, overwrite=True)

            truth = truth_root / f"truth_tract{entry['tract']}.parquet"
            regions.append(
                {
                    "id": region_id,
                    "tract": entry["tract"],
                    "patch": entry["patch"],
                    "band": entry["band"],
                    "center_x": center[0],
                    "center_y": center[1],
                    "size": size,
                    "path": rel_path,
                    "has_variance": variance is not None,
                    "truth": str(truth) if truth.exists() else None,
                }
            )

    manifest = {
        "module": "astro-denoise-bench",
        "kind": "dc2-curated-block",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "coadd_root": str(coadd_root),
        "truth_root": str(truth_root),
        "size": size,
        "regions_per_image": regions_per_image,
        "bands": sorted({r["band"] for r in regions}),
        "tracts": sorted({r["tract"] for r in regions}),
        "n_regions": len(regions),
        "regions": regions,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load_manifest(path):
    """Read a curated-block manifest, resolving region paths to absolute."""
    path = Path(path)
    manifest = json.loads(path.read_text())
    base = path.parent
    for region in manifest["regions"]:
        region["path"] = str(base / region["path"])
    return manifest
