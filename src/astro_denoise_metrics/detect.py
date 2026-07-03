"""Source detection used to build catalogs from denoised images.

Primary backend is SEP (SExtractor in Python); if SEP is unavailable it
falls back to photutils image segmentation. Both return the same
structured array with columns ``x``, ``y``, ``flux``.
"""

from __future__ import annotations

import numpy as np

try:
    import sep as _sep
except ImportError:  # pragma: no cover - depends on environment
    _sep = None


_CATALOG_DTYPE = np.dtype([("x", "f8"), ("y", "f8"), ("flux", "f8"), ("npix", "f8")])


def _as_catalog(x, y, flux, npix):
    cat = np.empty(len(x), dtype=_CATALOG_DTYPE)
    cat["x"], cat["y"], cat["flux"], cat["npix"] = x, y, flux, npix
    return cat


def _detect_sep(image, thresh, minarea):
    data = np.ascontiguousarray(image, dtype=np.float32)
    bkg = _sep.Background(data)
    data_sub = data - bkg.back()
    objects = _sep.extract(data_sub, thresh, err=bkg.globalrms, minarea=minarea)
    return _as_catalog(objects["x"], objects["y"], objects["flux"], objects["npix"])


def _detect_photutils(image, thresh, minarea):
    from astropy.stats import sigma_clipped_stats
    from photutils.segmentation import SourceCatalog, detect_sources

    data = np.asarray(image, dtype=np.float64)
    _, median, std = sigma_clipped_stats(data, sigma=3.0)
    segm = detect_sources(data - median, thresh * std, npixels=minarea)
    if segm is None:
        return _as_catalog([], [], [], [])
    cat = SourceCatalog(data - median, segm)
    return _as_catalog(
        np.asarray(cat.xcentroid),
        np.asarray(cat.ycentroid),
        np.asarray(cat.segment_flux),
        np.asarray(cat.area.value, dtype=np.float64),
    )


def detect_sources(image, thresh=1.5, minarea=5):
    """Detect sources and return a catalog with ``x``, ``y``, ``flux``.

    Parameters
    ----------
    image : 2-D ndarray
        Background-unsubtracted image.
    thresh : float
        Detection threshold in units of the background RMS.
    minarea : int
        Minimum number of connected pixels per source.
    """
    if _sep is not None:
        return _detect_sep(image, thresh, minarea)
    return _detect_photutils(image, thresh, minarea)


def source_mask(image, thresh=1.5, minarea=5, dilate=3):
    """Boolean mask that is True on detected source pixels.

    Useful as the exclusion mask for :func:`~astro_denoise_metrics.sky_sigma`.
    """
    from scipy.ndimage import binary_dilation

    data = np.asarray(image, dtype=np.float64)
    if _sep is not None:
        d32 = np.ascontiguousarray(data, dtype=np.float32)
        bkg = _sep.Background(d32)
        _, segmap = _sep.extract(
            d32 - bkg.back(),
            thresh,
            err=bkg.globalrms,
            minarea=minarea,
            segmentation_map=True,
        )
        mask = segmap > 0
    else:
        from astropy.stats import sigma_clipped_stats
        from photutils.segmentation import detect_sources as _pu_detect

        _, median, std = sigma_clipped_stats(data, sigma=3.0)
        segm = _pu_detect(data - median, thresh * std, npixels=minarea)
        mask = np.zeros(data.shape, dtype=bool) if segm is None else segm.data > 0
    if dilate > 0:
        mask = binary_dilation(mask, iterations=dilate)
    return mask
