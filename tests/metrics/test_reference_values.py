"""Reference-value tests for every metric the benchmark reports.

Each expected value comes from the published definition of the metric or
from a closed-form result computed here, never from the output of the code
under test. A silent change in what a metric means therefore fails a test.

Sources are named beside the value they justify:

* source matching, completeness, purity: one-to-one nearest-neighbour
  matching, expectations read off hand-built catalogues (construction);
* photometric accuracy: relative flux error and the magnitude offset
  ``-2.5 log10(f)``, both closed form;
* PSNR: ``10 log10(range^2 / MSE)``, closed form (Gonzalez & Woods, 3rd
  ed., 2008, ch. 3);
* SSIM: Wang et al. (2004), "Image quality assessment: from error
  visibility to structural similarity", IEEE TIP 13(4), 600-612;
* sky noise: sigma of a seeded Gaussian, recovered with Astropy's
  sigma-clipped statistics.
"""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, uniform_filter
from skimage.metrics import structural_similarity as skimage_ssim

from astro_denoise.metrics import (
    completeness,
    completeness_vs_mag,
    photometric_accuracy,
    psnr,
    purity,
    sky_sigma,
    ssim,
)
from astro_denoise.metrics.metrics import match_catalogs


def cat(x, y, flux=None, mag=None):
    """Catalog as a dict of arrays; photometric columns are optional."""
    catalog = {"x": np.asarray(x, dtype=float), "y": np.asarray(y, dtype=float)}
    if flux is not None:
        catalog["flux"] = np.asarray(flux, dtype=float)
    if mag is not None:
        catalog["mag"] = np.asarray(mag, dtype=float)
    return catalog


# A[i, j] = (8 i + j) % 7 spans 0..6, so data_range = 6. Two pixels move.
PAIR_A = np.fromfunction(lambda i, j: (8 * i + j) % 7, (8, 8), dtype=float)
PAIR_B = PAIR_A.copy()
PAIR_B[3, 3] += 2.0
PAIR_B[6, 1] -= 1.5
PAIR_RANGE = 6.0


def _ssim_uniform(x, y, win=7, data_range=PAIR_RANGE):
    """SSIM with a uniform window, independent of the library under test.

    Mirrors the scikit-image default for this image size: 7x7 uniform
    window, sample covariance ``N / (N - 1)``, constants
    ``C1 = (K1 L)^2``, ``C2 = (K2 L)^2`` with K1 = 0.01, K2 = 0.03
    (Wang et al. 2004), and the border cropped by ``pad = (win - 1) // 2``.
    """
    npix = win * win
    cov_norm = npix / (npix - 1)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ux = uniform_filter(x, size=win)
    uy = uniform_filter(y, size=win)
    uxx = cov_norm * (uniform_filter(x * x, size=win) - ux * ux)
    uyy = cov_norm * (uniform_filter(y * y, size=win) - uy * uy)
    uxy = cov_norm * (uniform_filter(x * y, size=win) - ux * uy)
    numerator = (2 * ux * uy + c1) * (2 * uxy + c2)
    denominator = (ux * ux + uy * uy + c1) * (uxx + uyy + c2)
    pad = (win - 1) // 2
    return float((numerator / denominator)[pad:-pad, pad:-pad].mean())


def _ssim_wang(x, y, sigma=1.5, truncate=10 / 3, data_range=PAIR_RANGE):
    """SSIM with the window and constants of Wang et al. (2004).

    Sigma = 1.5 (``truncate = 10 / 3`` gives an 11x11 window), Gaussian
    weights normalised to unit sum so the weighted covariance is the
    population one, K1 = 0.01, K2 = 0.03. The paper leaves border handling
    open; this implementation pads by reflection and averages the full map.
    """
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    def smooth(image):
        return gaussian_filter(image, sigma=sigma, truncate=truncate, mode="reflect")

    ux = smooth(x)
    uy = smooth(y)
    uxx = smooth(x * x) - ux * ux
    uyy = smooth(y * y) - uy * uy
    uxy = smooth(x * y) - ux * uy
    numerator = (2 * ux * uy + c1) * (2 * uxy + c2)
    denominator = (ux * ux + uy * uy + c1) * (uxx + uyy + c2)
    return float((numerator / denominator).mean())


class TestMatchConstruction:
    def test_detection_within_radius_of_two_truth(self):
        # Nearest-neighbour by construction: det at (10, 10) is 1 px from
        # truth 0 and 1.5 px from truth 1, so only the closer one may match.
        det = cat([10.0], [10.0])
        truth = cat([9.0, 11.5], [10.0, 10.0])
        det_idx, truth_idx = match_catalogs(det, truth, radius=2.0)
        np.testing.assert_array_equal(det_idx, [0])
        np.testing.assert_array_equal(truth_idx, [0])

    def test_exactly_at_radius(self):
        # The locked scipy 1.18.0 treats distance_upper_bound as strict: a
        # neighbour at exactly that distance comes back as inf. Its
        # cKDTree.query docstring only promises "within this distance", so
        # this pins the behaviour of the pinned scipy rather than a wider
        # contract: a source exactly at the radius is not matched, one just
        # inside it is.
        det = cat([10.0], [10.0])
        at_radius = cat([12.0], [10.0])
        det_idx, truth_idx = match_catalogs(det, at_radius, radius=2.0)
        assert det_idx.size == 0
        assert truth_idx.size == 0

        just_inside = cat([11.999], [10.0])
        det_idx, truth_idx = match_catalogs(det, just_inside, radius=2.0)
        assert det_idx.size == 1
        assert truth_idx.size == 1

    def test_empty_detection_catalog(self):
        det_idx, truth_idx = match_catalogs(cat([], []), cat([1.0], [1.0]), radius=2.0)
        assert det_idx.size == 0
        assert truth_idx.size == 0

    def test_empty_truth_catalog(self):
        det_idx, truth_idx = match_catalogs(cat([1.0], [1.0]), cat([], []), radius=2.0)
        assert det_idx.size == 0
        assert truth_idx.size == 0


class TestCompletenessPurityRatios:
    def test_plain_ratios(self):
        # 5 truth sources, 3 recovered, 4 detections -> 3/5 and 3/4.
        truth = cat([10, 20, 30, 40, 50], [10, 20, 30, 40, 50])
        det = cat([10.0, 20.0, 30.0, 99.0], [10.0, 20.0, 30.0, 99.0])
        assert completeness(det, truth, radius=1.0) == pytest.approx(3 / 5)
        assert purity(det, truth, radius=1.0) == pytest.approx(3 / 4)

    def test_vs_mag_counts_per_bin(self):
        # Mags -10, -8, -7, -6 fall in [-11, -5); mag -2 in [-5, 0).
        # Detections recover the first, second and fourth truth source.
        mags = np.array([-10.0, -8.0, -7.0, -6.0, -2.0])
        fluxes = 10 ** (-0.4 * mags)
        xy = np.arange(10.0, 60.0, 10.0)
        truth = cat(xy, xy, flux=fluxes, mag=mags)
        det = cat([10.0, 20.0, 40.0], [10.0, 20.0, 40.0])

        bins = [-11.0, -5.0, 0.0]
        curve = completeness_vs_mag(det, truth, mag_bins=bins, radius=1.0)

        # By construction: [-11, -5) holds four truth sources of which three
        # are recovered, and [-5, 0) holds one of which none is.
        np.testing.assert_array_equal(curve["n_truth"], [4, 1])
        np.testing.assert_array_equal(curve["n_recovered"], [3, 0])
        np.testing.assert_array_equal(curve["completeness"], [3 / 4, 0.0])

        # Cross-check the same counts from the matching primitive and the
        # binned histogram, independently of the curve's own bookkeeping.
        _, truth_idx = match_catalogs(det, truth, radius=1.0)
        recovered = np.zeros(mags.size, dtype=bool)
        recovered[truth_idx] = True
        n_truth, _ = np.histogram(mags, bins=bins)
        n_recovered, _ = np.histogram(mags[recovered], bins=bins)

        np.testing.assert_array_equal(curve["n_truth"], n_truth)
        np.testing.assert_array_equal(curve["n_recovered"], n_recovered)
        np.testing.assert_array_equal(curve["completeness"], n_recovered / n_truth)


class TestPhotometricReference:
    def test_constant_factor_bias(self):
        # Closed form: relative flux error is factor - 1 and the magnitude
        # offset is -2.5 log10(factor), with zero scatter.
        truth_flux = np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0])
        pos = np.arange(6.0) * 10.0
        truth = cat(pos, pos, flux=truth_flux)
        det = cat(pos, pos, flux=truth_flux * 1.25)
        result = photometric_accuracy(det, truth, radius=1.0)
        assert result["n_matched"] == 6
        assert result["flux_bias"] == pytest.approx(0.25)
        assert result["mag_bias"] == pytest.approx(-2.5 * np.log10(1.25))
        assert result["flux_scatter"] == pytest.approx(0.0, abs=1e-12)

    def test_gaussian_scatter(self):
        # Relative flux errors drawn N(0, 0.05): the robust scatter
        # 1.4826 * MAD estimates the Gaussian sigma (Hampel 1974; the
        # Gaussian-consistency constant is 1.4826 = 1/Phi^-1(0.75)).
        rng = np.random.default_rng(42)
        sigma = 0.05
        n_src = 1000
        truth_flux = np.full(n_src, 1000.0)
        rel_err = rng.normal(0.0, sigma, n_src)
        pos = np.arange(n_src, dtype=float)
        truth = cat(pos, pos, flux=truth_flux)
        det = cat(pos, pos, flux=truth_flux * (1.0 + rel_err))
        result = photometric_accuracy(det, truth, radius=1.0)
        assert result["n_matched"] == n_src
        assert abs(result["flux_bias"]) < 0.008
        assert result["flux_scatter"] == pytest.approx(sigma, rel=0.15)


class TestPsnrClosedForm:
    def test_equals_10log10_range_squared_over_mse(self):
        # Closed form (Gonzalez & Woods 2008, ch. 3): range = 4, constant
        # error 0.1 gives MSE = 0.01, so PSNR = 10 log10(16 / 0.01) dB.
        original = np.linspace(0.0, 4.0, 64).reshape(8, 8)
        denoised = original + 0.1
        data_range = original.max() - original.min()
        mse = np.mean((original - denoised) ** 2)
        expected = 10.0 * np.log10(data_range**2 / mse)
        assert expected == pytest.approx(32.04119982655924)
        assert psnr(original, denoised) == pytest.approx(expected, abs=1e-12)


class TestSsimReference:
    def test_identity_is_one(self):
        # Wang et al. (2004): identical images give SSIM = 1 everywhere.
        assert ssim(PAIR_A, PAIR_A) == pytest.approx(1.0)

    def test_default_window_is_pinned(self):
        # scikit-image defaults for an 8x8 image: 7x7 uniform window,
        # sample covariance, K1 = 0.01, K2 = 0.03. Value from the
        # independent uniform-window implementation, equal to the library.
        library = skimage_ssim(PAIR_A, PAIR_B, data_range=PAIR_RANGE)
        assert library == pytest.approx(0.9851537881821366, abs=1e-9)
        assert ssim(PAIR_A, PAIR_B) == pytest.approx(library, abs=1e-12)
        assert ssim(PAIR_A, PAIR_B) == pytest.approx(_ssim_uniform(PAIR_A, PAIR_B), abs=1e-9)

    def test_wang2004_differs_bounded(self):
        # Wang et al. (2004) uses an 11x11 Gaussian window with sigma 1.5;
        # on this pair it must differ from the library default, and the
        # difference must stay small (the two are related definitions).
        expected_wang = 0.98872677215362
        wang = _ssim_wang(PAIR_A, PAIR_B)
        assert wang == pytest.approx(expected_wang, abs=1e-9)
        library = ssim(PAIR_A, PAIR_B)
        assert 1e-4 < abs(library - wang) < 0.02


class TestSkyNoiseReference:
    def test_recovers_sigma(self):
        # Pure Gaussian sky N(0, 3): sigma-clipped std recovers sigma.
        rng = np.random.default_rng(7)
        image = rng.normal(0.0, 3.0, size=(256, 256))
        assert sky_sigma(image) == pytest.approx(3.0, rel=0.05)

    def test_sources_masked_do_not_move_sigma(self):
        # Masking the source footprints leaves the same sky pixels, so the
        # measured sigma is unchanged by the presence of sources.
        rng = np.random.default_rng(7)
        sky = rng.normal(0.0, 3.0, size=(256, 256))
        image = sky.copy()
        mask = np.zeros(sky.shape, dtype=bool)
        for cx, cy in [(40, 40), (90, 90)]:
            image[cy - 4 : cy + 5, cx - 4 : cx + 5] += 200.0
            mask[cy - 4 : cy + 5, cx - 4 : cx + 5] = True
        assert sky_sigma(image, mask=mask) == pytest.approx(sky_sigma(sky), rel=0.03)
