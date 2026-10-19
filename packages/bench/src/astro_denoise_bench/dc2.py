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


def random_positions(shape, size, n, rng):
    """Pseudo-random, spread-out cutout centers inside ``shape``.

    Uniform sampling with a half-cutout margin from the borders. The
    sampling is driven by ``rng`` so a curated block is different on
    every run unless the seed is fixed — the seed is recorded in the
    manifest, which is what makes any given block reproducible.
    """
    height, width = shape
    margin = size / 2
    xs = rng.uniform(margin, width - margin, n)
    ys = rng.uniform(margin, height - margin, n)
    return [(float(x), float(y)) for x, y in zip(xs, ys, strict=True)]


def _region_id(entry, index):
    patch = entry["patch"].replace(",", "")
    return f"t{entry['tract']}-p{patch}-{entry['band']}-r{index}"


def cut_region(coadd_path, center, size):
    """Return (image cutout, variance cutout or None, WCS header, sky bbox).

    The sky bbox is (ra_min, ra_max, dec_min, dec_max) of the cutout
    corners, used to trim the truth catalog to the covered area.
    """
    with fits.open(coadd_path) as hdul:
        image = np.asarray(hdul[IMAGE_HDU].data, dtype=np.float32)
        wcs = WCS(hdul[IMAGE_HDU].header)
        cut = Cutout2D(image, position=center, size=size, wcs=wcs)
        variance = None
        if len(hdul) > VARIANCE_HDU and hdul[VARIANCE_HDU].data is not None:
            var_plane = np.asarray(hdul[VARIANCE_HDU].data, dtype=np.float32)
            if var_plane.shape == image.shape:
                variance = var_plane[cut.slices_original]

    ny, nx = cut.data.shape
    corners_x = np.array([0.0, nx - 1.0, 0.0, nx - 1.0])
    corners_y = np.array([0.0, 0.0, ny - 1.0, ny - 1.0])
    ra, dec = cut.wcs.pixel_to_world_values(corners_x, corners_y)
    bbox = (float(np.min(ra)), float(np.max(ra)), float(np.min(dec)), float(np.max(dec)))
    return cut.data, variance, cut.wcs.to_header(), bbox


#: sky margin (degrees) added around region bboxes when trimming truth
TRUTH_MARGIN_DEG = 0.0015

#: truth columns kept in the distributed block
TRUTH_COLUMNS = ("ra", "dec", "is_unique_truth_entry")
TRUTH_FLUX_PREFIX = "flux_"


def trim_truth(parquet_in, parquet_out, bboxes, margin=TRUTH_MARGIN_DEG):
    """Write a trimmed copy of a DC2 truth catalog covering ``bboxes``.

    Keeps only the rows inside any region bbox (+``margin`` degrees) and
    the columns the metrics engine needs (ra/dec/flux_*/uniqueness), so
    the curated block stays small enough to distribute. Assumes DC2 RA
    ranges do not wrap through 0 degrees.
    """
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    schema_names = set(pq.read_schema(parquet_in).names)
    columns = [c for c in TRUTH_COLUMNS if c in schema_names]
    columns += sorted(c for c in schema_names if c.startswith(TRUTH_FLUX_PREFIX))
    table = pq.read_table(parquet_in, columns=columns)

    mask = None
    for ra_min, ra_max, dec_min, dec_max in bboxes:
        box = pc.and_(
            pc.and_(
                pc.greater_equal(table["ra"], ra_min - margin),
                pc.less_equal(table["ra"], ra_max + margin),
            ),
            pc.and_(
                pc.greater_equal(table["dec"], dec_min - margin),
                pc.less_equal(table["dec"], dec_max + margin),
            ),
        )
        mask = box if mask is None else pc.or_(mask, box)
    trimmed = table.filter(mask)
    pq.write_table(trimmed, parquet_out)
    return trimmed.num_rows


def curate(
    coadd_root,
    truth_root,
    out_dir,
    bands=None,
    size=256,
    regions_per_image=3,
    seed=None,
):
    """Build the curated heterogeneous region block and its manifest.

    Region positions are sampled pseudo-randomly: with ``seed=None`` a
    fresh seed is drawn, so every curated block covers different sky
    regions; the seed used is recorded in the manifest so any block can
    be regenerated exactly. Writes one FITS per region (image HDU with
    WCS + VARIANCE extension when the coadd provides one) under
    ``<out_dir>/regions/`` and a ``manifest.json``. Returns the manifest.
    """
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % (2**32))
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    regions_dir = out_dir / "regions"
    regions_dir.mkdir(parents=True, exist_ok=True)
    truth_root = Path(truth_root)

    coadds = discover_coadds(coadd_root, bands=bands)
    if not coadds:
        raise FileNotFoundError(f"no coadd images found under {coadd_root}")

    regions = []
    tract_bboxes: dict[int, list] = {}
    for entry in coadds:
        with fits.open(entry["path"]) as hdul:
            shape = hdul[IMAGE_HDU].data.shape
        for index, center in enumerate(random_positions(shape, size, regions_per_image, rng)):
            region_id = _region_id(entry, index)
            image, variance, header, bbox = cut_region(entry["path"], center, size)
            hdus = [fits.PrimaryHDU(data=image, header=header)]
            if variance is not None:
                hdus.append(fits.ImageHDU(data=variance, name="VARIANCE"))
            rel_path = f"regions/{region_id}.fits"
            fits.HDUList(hdus).writeto(out_dir / rel_path, overwrite=True)
            tract_bboxes.setdefault(entry["tract"], []).append(bbox)

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
                    "truth": None,  # filled below once the tract truth is trimmed
                }
            )

    # Self-contained block: trim each tract's truth catalog to the sky
    # area actually covered and ship it inside the block with a relative
    # path, so the manifest stays portable (Hugging Face distribution).
    truth_dir = out_dir / "truth"
    tract_truth = {}
    for tract, bboxes in tract_bboxes.items():
        source = truth_root / f"truth_tract{tract}.parquet"
        if not source.exists():
            continue
        truth_dir.mkdir(parents=True, exist_ok=True)
        rel_truth = f"truth/truth_tract{tract}.parquet"
        trim_truth(source, out_dir / rel_truth, bboxes)
        tract_truth[tract] = rel_truth
    for region in regions:
        region["truth"] = tract_truth.get(region["tract"])

    manifest = {
        "module": "astro-denoise-bench",
        "kind": "dc2-curated-block",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "coadd_root": str(coadd_root),
        "truth_root": str(truth_root),
        "size": size,
        "regions_per_image": regions_per_image,
        "seed": seed,
        "bands": sorted({r["band"] for r in regions}),
        "tracts": sorted({r["tract"] for r in regions}),
        "n_regions": len(regions),
        "regions": regions,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load_manifest(path):
    """Read a curated-block manifest, resolving relative paths to absolute.

    Region and truth paths are stored relative to the manifest so the
    block is portable; absolute truth paths (older manifests) pass
    through untouched.
    """
    path = Path(path)
    manifest = json.loads(path.read_text())
    base = path.parent
    for region in manifest["regions"]:
        region["path"] = str(base / region["path"])
        truth = region.get("truth")
        if truth and not Path(truth).is_absolute():
            region["truth"] = str(base / truth)
    return manifest
