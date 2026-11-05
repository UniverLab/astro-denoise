"""DC2 truth-catalog adapter: sky-coordinate Parquet -> pixel-space truth.

The DESC DC2 ``truth_match`` catalogs ship as per-tract Parquet files with
sky coordinates (``ra``/``dec`` in degrees) and per-band fluxes in nJy.
The metrics layer consumes pixel-space catalogs, so this module projects
truth sources onto a given image's WCS and keeps those inside the image
bounds, optionally cut at a limiting magnitude.
"""

from __future__ import annotations

import numpy as np

DC2_BANDS = ("u", "g", "r", "i", "z", "y")

#: AB magnitude zero point for fluxes expressed in nJy.
AB_ZEROPOINT_NJY = 31.4


def flux_njy_to_mag(flux):
    """AB magnitude from flux in nJy; non-positive fluxes become NaN."""
    flux = np.asarray(flux, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(flux > 0, -2.5 * np.log10(flux) + AB_ZEROPOINT_NJY, np.nan)


def load_dc2_truth(path, band):
    """Read ``ra``/``dec``/``flux_<band>`` arrays from a DC2 Parquet catalog.

    Rows flagged duplicate by ``is_unique_truth_entry`` (when the column
    exists) are dropped, so each physical source appears once.
    """
    import pyarrow.parquet as pq

    if band not in DC2_BANDS:
        raise ValueError(f"unknown DC2 band {band!r} (expected one of {DC2_BANDS})")
    flux_col = f"flux_{band}"
    available = set(pq.read_schema(path).names)
    missing = {"ra", "dec", flux_col} - available
    if missing:
        raise ValueError(f"{path}: missing DC2 truth columns {sorted(missing)}")

    columns = ["ra", "dec", flux_col]
    unique_col = "is_unique_truth_entry"
    if unique_col in available:
        columns.append(unique_col)
    data = pq.read_table(path, columns=columns).to_pydict()

    ra = np.asarray(data["ra"], dtype=np.float64)
    dec = np.asarray(data["dec"], dtype=np.float64)
    flux = np.asarray(data[flux_col], dtype=np.float64)
    if unique_col in data:
        keep = np.asarray(data[unique_col], dtype=bool)
        ra, dec, flux = ra[keep], dec[keep], flux[keep]
    return {"ra": ra, "dec": dec, "flux": flux}


def project_truth(truth, wcs, shape, mag_limit=None, margin=0.5):
    """Project sky truth onto ``wcs`` pixels, keeping sources inside ``shape``.

    Returns the pixel-space catalog dict (``x``, ``y``, ``flux``, ``mag``)
    that the metric functions and the CLI consume. ``mag_limit`` drops
    sources fainter than the given AB magnitude (useful to restrict the
    truth to the detectable range of the image).
    """
    x, y = wcs.world_to_pixel_values(truth["ra"], truth["dec"])
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mag = flux_njy_to_mag(truth["flux"])

    keep = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= -margin)
        & (x <= shape[1] - 1 + margin)
        & (y >= -margin)
        & (y <= shape[0] - 1 + margin)
    )
    if mag_limit is not None:
        keep &= np.isfinite(mag) & (mag <= mag_limit)
    return {
        "x": x[keep],
        "y": y[keep],
        "flux": np.asarray(truth["flux"], dtype=np.float64)[keep],
        "mag": mag[keep],
    }
