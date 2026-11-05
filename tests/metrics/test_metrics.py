"""Unit tests for the metric functions on synthetic catalogs and images."""

import numpy as np
import pytest

from astro_denoise_metrics import (
    completeness,
    completeness_vs_mag,
    photometric_accuracy,
    psnr,
    purity,
    sky_sigma,
    ssim,
)
from astro_denoise_metrics.metrics import match_catalogs


def catalog(x, y, flux=None):
    cat = {"x": np.asarray(x, dtype=float), "y": np.asarray(y, dtype=float)}
    if flux is not None:
        cat["flux"] = np.asarray(flux, dtype=float)
    return cat


class TestMatchCatalogs:
    def test_exact_match(self):
        truth = catalog([10, 20, 30], [10, 20, 30])
        det_idx, truth_idx = match_catalogs(truth, truth, radius=1.0)
        assert len(det_idx) == 3
        np.testing.assert_array_equal(det_idx, truth_idx)

    def test_one_to_one(self):
        # Two detections near a single truth source: only one may match.
        det = catalog([10.0, 10.5], [10.0, 10.0])
        truth = catalog([10.1], [10.0])
        det_idx, truth_idx = match_catalogs(det, truth, radius=2.0)
        assert len(det_idx) == 1
        assert det_idx[0] == 0  # closest detection wins

    def test_radius_cut(self):
        det = catalog([10], [10])
        truth = catalog([15], [10])
        det_idx, _ = match_catalogs(det, truth, radius=2.0)
        assert len(det_idx) == 0

    def test_empty_catalogs(self):
        det_idx, truth_idx = match_catalogs(catalog([], []), catalog([1], [1]), 2.0)
        assert len(det_idx) == 0 and len(truth_idx) == 0


class TestRecoveryMetrics:
    def test_completeness_and_purity(self):
        truth = catalog([10, 20, 30, 40], [10, 20, 30, 40])
        # 3 recovered truth sources + 1 spurious detection.
        det = catalog([10.2, 20.1, 30.0, 90.0], [10.0, 20.0, 30.1, 90.0])
        assert completeness(det, truth, radius=1.0) == pytest.approx(0.75)
        assert purity(det, truth, radius=1.0) == pytest.approx(0.75)

    def test_empty_truth_is_nan(self):
        assert np.isnan(completeness(catalog([1], [1]), catalog([], []), 1.0))
        assert np.isnan(purity(catalog([], []), catalog([1], [1]), 1.0))

    def test_completeness_vs_mag(self):
        # Bright source recovered, faint one missed.
        truth = catalog([10, 20], [10, 20], flux=[10000.0, 10.0])
        det = catalog([10.1], [10.0], flux=[10000.0])
        curve = completeness_vs_mag(det, truth, mag_bins=[-11, -5, 0], radius=1.0)
        assert curve["n_truth"] == [1, 1]
        assert curve["completeness"][0] == pytest.approx(1.0)
        assert curve["completeness"][1] == pytest.approx(0.0)


class TestPhotometricAccuracy:
    def test_known_bias(self):
        truth = catalog([10, 20, 30], [10, 20, 30], flux=[100.0, 200.0, 300.0])
        det = catalog([10, 20, 30], [10, 20, 30], flux=[110.0, 220.0, 330.0])
        result = photometric_accuracy(det, truth, radius=1.0)
        assert result["n_matched"] == 3
        assert result["flux_bias"] == pytest.approx(0.10)
        assert result["flux_scatter"] == pytest.approx(0.0, abs=1e-12)
        assert result["mag_bias"] == pytest.approx(-2.5 * np.log10(1.1))

    def test_no_matches(self):
        result = photometric_accuracy(
            catalog([0], [0], flux=[1.0]), catalog([99], [99], flux=[1.0]), 1.0
        )
        assert result["n_matched"] == 0
        assert np.isnan(result["flux_bias"])


class TestImageQuality:
    def test_psnr_identity_is_inf(self):
        image = np.random.default_rng(0).normal(size=(32, 32))
        assert np.isinf(psnr(image, image))

    def test_psnr_decreases_with_noise(self):
        rng = np.random.default_rng(1)
        image = rng.normal(100, 10, size=(64, 64))
        small = psnr(image, image + rng.normal(0, 0.1, image.shape))
        large = psnr(image, image + rng.normal(0, 1.0, image.shape))
        assert small > large

    def test_ssim_identity_is_one(self):
        image = np.random.default_rng(2).normal(100, 10, size=(64, 64))
        assert ssim(image, image) == pytest.approx(1.0)

    def test_sky_sigma_recovers_noise_level(self):
        rng = np.random.default_rng(3)
        image = rng.normal(0.0, 5.0, size=(256, 256))
        assert sky_sigma(image) == pytest.approx(5.0, rel=0.05)

    def test_sky_sigma_mask_excludes_sources(self):
        rng = np.random.default_rng(4)
        image = rng.normal(0.0, 2.0, size=(128, 128))
        mask = np.zeros(image.shape, dtype=bool)
        image[60:68, 60:68] += 1000.0  # bright source
        mask[60:68, 60:68] = True
        assert sky_sigma(image, mask=mask) == pytest.approx(2.0, rel=0.05)
