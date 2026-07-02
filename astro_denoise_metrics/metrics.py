"""Scientific metrics for evaluating astronomical image denoising.

Catalogs are anything indexable by column name (numpy structured array,
astropy Table, dict of arrays) with at least ``x`` and ``y`` columns in
pixel coordinates. Photometric metrics additionally need a ``flux``
column; magnitude-binned metrics accept a ``mag`` column or derive
instrumental magnitudes as ``-2.5 * log10(flux)``.

Matching between a detection catalog and a truth catalog is one-to-one
nearest-neighbour within a matching radius (pixels), so a single
detection cannot "recover" two truth sources.
"""

from __future__ import annotations

import numpy as np
from astropy.stats import sigma_clipped_stats
from scipy.spatial import cKDTree
from skimage.metrics import peak_signal_noise_ratio as _sk_psnr
from skimage.metrics import structural_similarity as _sk_ssim


# ---------------------------------------------------------------------------
# catalog helpers
# ---------------------------------------------------------------------------

def _column(catalog, name):
    """Extract a column from a Table / structured array / dict as float64."""
    try:
        col = catalog[name]
    except (KeyError, ValueError, IndexError) as exc:
        raise KeyError(f"catalog has no column {name!r}") from exc
    return np.asarray(col, dtype=np.float64)


def _positions(catalog):
    return np.column_stack([_column(catalog, "x"), _column(catalog, "y")])


def _magnitudes(catalog):
    """Return magnitudes, preferring a 'mag' column over flux-derived ones."""
    try:
        return _column(catalog, "mag")
    except KeyError:
        flux = _column(catalog, "flux")
        with np.errstate(divide="ignore", invalid="ignore"):
            return -2.5 * np.log10(np.where(flux > 0, flux, np.nan))


def match_catalogs(catalog, truth, radius):
    """One-to-one nearest-neighbour match of detections against truth.

    Returns ``(det_idx, truth_idx)`` arrays of matched pairs, each index
    used at most once, with pair distance <= radius. Pairs are assigned
    greedily by increasing distance.
    """
    det_xy = _positions(catalog)
    truth_xy = _positions(truth)
    if len(det_xy) == 0 or len(truth_xy) == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)

    tree = cKDTree(truth_xy)
    dist, nearest = tree.query(det_xy, distance_upper_bound=radius)
    candidates = np.flatnonzero(np.isfinite(dist))
    order = candidates[np.argsort(dist[candidates])]

    det_used = np.zeros(len(det_xy), dtype=bool)
    truth_used = np.zeros(len(truth_xy), dtype=bool)
    det_idx, truth_idx = [], []
    for i in order:
        j = nearest[i]
        if det_used[i] or truth_used[j]:
            continue
        det_used[i] = True
        truth_used[j] = True
        det_idx.append(i)
        truth_idx.append(j)
    return np.asarray(det_idx, dtype=int), np.asarray(truth_idx, dtype=int)


# ---------------------------------------------------------------------------
# source-recovery metrics
# ---------------------------------------------------------------------------

def completeness(catalog, truth, radius):
    """Fraction of truth sources recovered by the detection catalog.

    A truth source counts as recovered when it is matched (one-to-one,
    within ``radius`` pixels) to a detection.
    """
    n_truth = len(_column(truth, "x"))
    if n_truth == 0:
        return float("nan")
    _, truth_idx = match_catalogs(catalog, truth, radius)
    return len(truth_idx) / n_truth


def purity(catalog, truth, radius):
    """Fraction of detections that correspond to a real (truth) source."""
    n_det = len(_column(catalog, "x"))
    if n_det == 0:
        return float("nan")
    det_idx, _ = match_catalogs(catalog, truth, radius)
    return len(det_idx) / n_det


def completeness_vs_mag(catalog, truth, mag_bins, radius=2.0):
    """Completeness curve as a function of truth magnitude.

    Parameters
    ----------
    mag_bins : array-like
        Bin edges in magnitude (length nbins + 1).
    radius : float
        Matching radius in pixels.

    Returns
    -------
    dict with keys ``mag_bins`` (edges), ``mag_centers``, ``completeness``
    (per-bin recovered fraction, NaN where a bin has no truth sources),
    ``n_truth`` and ``n_recovered`` per bin.
    """
    mag_bins = np.asarray(mag_bins, dtype=np.float64)
    truth_mag = _magnitudes(truth)
    _, truth_idx = match_catalogs(catalog, truth, radius)
    recovered = np.zeros(len(truth_mag), dtype=bool)
    recovered[truth_idx] = True

    n_truth, _ = np.histogram(truth_mag, bins=mag_bins)
    n_rec, _ = np.histogram(truth_mag[recovered], bins=mag_bins)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(n_truth > 0, n_rec / np.maximum(n_truth, 1), np.nan)

    return {
        "mag_bins": mag_bins.tolist(),
        "mag_centers": (0.5 * (mag_bins[:-1] + mag_bins[1:])).tolist(),
        "completeness": frac.tolist(),
        "n_truth": n_truth.tolist(),
        "n_recovered": n_rec.tolist(),
    }


def photometric_accuracy(catalog, truth, radius=2.0):
    """Flux bias and scatter of matched detections against truth fluxes.

    Uses the relative flux error ``(f_det - f_true) / f_true`` on matched
    pairs. Bias is the median; scatter is the robust (MAD-based) standard
    deviation. Also reports the magnitude offset statistics.
    """
    det_idx, truth_idx = match_catalogs(catalog, truth, radius)
    empty = {
        "n_matched": 0,
        "flux_bias": float("nan"),
        "flux_scatter": float("nan"),
        "mag_bias": float("nan"),
        "mag_scatter": float("nan"),
    }
    if len(det_idx) == 0:
        return empty

    f_det = _column(catalog, "flux")[det_idx]
    f_true = _column(truth, "flux")[truth_idx]
    valid = (f_true > 0) & np.isfinite(f_det) & np.isfinite(f_true)
    if not np.any(valid):
        return empty
    f_det, f_true = f_det[valid], f_true[valid]

    rel_err = (f_det - f_true) / f_true
    mad = np.median(np.abs(rel_err - np.median(rel_err)))

    pos = (f_det > 0)
    if np.any(pos):
        dmag = -2.5 * np.log10(f_det[pos] / f_true[pos])
        mag_bias = float(np.median(dmag))
        mag_mad = np.median(np.abs(dmag - np.median(dmag)))
        mag_scatter = float(1.4826 * mag_mad)
    else:
        mag_bias = mag_scatter = float("nan")

    return {
        "n_matched": int(valid.sum()),
        "flux_bias": float(np.median(rel_err)),
        "flux_scatter": float(1.4826 * mad),
        "mag_bias": mag_bias,
        "mag_scatter": mag_scatter,
    }


# ---------------------------------------------------------------------------
# image-quality metrics
# ---------------------------------------------------------------------------

def _data_range(reference):
    rng = float(np.nanmax(reference) - np.nanmin(reference))
    return rng if rng > 0 else 1.0


def psnr(original, denoised):
    """Peak signal-to-noise ratio (dB) of ``denoised`` against ``original``."""
    original = np.asarray(original, dtype=np.float64)
    denoised = np.asarray(denoised, dtype=np.float64)
    return float(_sk_psnr(original, denoised, data_range=_data_range(original)))


def ssim(original, denoised):
    """Structural similarity index of ``denoised`` against ``original``."""
    original = np.asarray(original, dtype=np.float64)
    denoised = np.asarray(denoised, dtype=np.float64)
    return float(_sk_ssim(original, denoised, data_range=_data_range(original)))


def sky_sigma(image, mask=None):
    """Residual background noise: sigma-clipped std of sky pixels.

    Parameters
    ----------
    image : ndarray
        Image to measure.
    mask : ndarray of bool, optional
        True where pixels must be *excluded* (sources, bad pixels).
    """
    image = np.asarray(image, dtype=np.float64)
    _, _, std = sigma_clipped_stats(image, mask=mask, sigma=3.0, maxiters=5)
    return float(std)
