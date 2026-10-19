"""Tests for the literature-aligned metrics added in v0.3.

Covers the classical pixel metrics (MSE/RMSE/MAE), the Fourier
sharpness S_F, the aperture FLUX-SNR and the detection-rate sweep.
"""

import numpy as np
import pytest

from astro_denoise_metrics import (
    detect_sources,
    detection_rate,
    flux_snr,
    fourier_sharpness,
    mae,
    mse,
    rmse,
    sky_sigma,
)


def gaussian_source(shape, x, y, flux, fwhm=3.0):
    sigma = fwhm / 2.3548
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    return flux * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))


class TestClassicalPixelMetrics:
    def test_identity_is_zero(self):
        image = np.random.default_rng(0).normal(size=(32, 32))
        assert mse(image, image) == 0.0
        assert rmse(image, image) == 0.0
        assert mae(image, image) == 0.0

    def test_constant_offset(self):
        image = np.zeros((16, 16))
        shifted = image + 2.0
        assert mse(image, shifted) == pytest.approx(4.0)
        assert rmse(image, shifted) == pytest.approx(2.0)
        assert mae(image, shifted) == pytest.approx(2.0)

    def test_rmse_is_sqrt_mse(self):
        rng = np.random.default_rng(1)
        a, b = rng.normal(size=(32, 32)), rng.normal(size=(32, 32))
        assert rmse(a, b) == pytest.approx(np.sqrt(mse(a, b)))


class TestFourierSharpness:
    def test_smoothing_reduces_sharpness(self):
        from scipy.ndimage import gaussian_filter

        rng = np.random.default_rng(2)
        image = rng.normal(0, 1, (128, 128))
        smoothed = gaussian_filter(image, sigma=2.0)
        assert fourier_sharpness(smoothed) < fourier_sharpness(image)

    def test_white_noise_has_high_sharpness(self):
        rng = np.random.default_rng(3)
        noise = rng.normal(0, 1, (128, 128))
        # white noise spreads power evenly; most of it sits at high radius
        assert fourier_sharpness(noise) > 0.5

    def test_flat_image_is_zero(self):
        assert fourier_sharpness(np.full((64, 64), 7.0)) == 0.0

    def test_bounded_between_zero_and_one(self):
        rng = np.random.default_rng(4)
        s = fourier_sharpness(rng.normal(size=(64, 64)))
        assert 0.0 <= s <= 1.0


class TestFluxSnr:
    def test_bright_sources_have_high_snr(self):
        rng = np.random.default_rng(5)
        shape = (128, 128)
        image = np.full(shape, 100.0) + rng.normal(0, 2.0, shape)
        for x, y in [(30, 40), (64, 64), (100, 90)]:
            image += gaussian_source(shape, x, y, flux=8000.0)

        catalog = detect_sources(image, thresh=1.5)
        snr = flux_snr(catalog, sky_sigma(image))
        assert np.isfinite(snr)
        assert snr > 10.0

    def test_empty_catalog_is_nan(self):
        catalog = np.empty(0, dtype=[("x", "f8"), ("y", "f8"), ("flux", "f8"), ("npix", "f8")])
        assert np.isnan(flux_snr(catalog, 1.0))

    def test_catalog_without_npix_is_nan(self):
        catalog = {"x": np.array([1.0]), "y": np.array([1.0]), "flux": np.array([10.0])}
        assert np.isnan(flux_snr(catalog, 1.0))


class TestDetectionRate:
    def make_image(self):
        rng = np.random.default_rng(6)
        shape = (128, 128)
        image = np.full(shape, 100.0) + rng.normal(0, 2.0, shape)
        xs, ys = [30.0, 64.0, 100.0], [40.0, 64.0, 85.0]
        for x, y in zip(xs, ys, strict=True):
            image += gaussian_source(shape, x, y, flux=5000.0)
        truth = {"x": np.array(xs), "y": np.array(ys)}
        return image, truth

    def test_detections_decrease_with_threshold(self):
        image, _ = self.make_image()
        result = detection_rate(image, thresholds=[1.0, 3.0, 10.0])
        n = result["n_detected"]
        assert n[0] >= n[1] >= n[2]

    def test_completeness_and_purity_reported_with_truth(self):
        image, truth = self.make_image()
        result = detection_rate(image, thresholds=[1.5, 5.0], truth=truth)
        assert len(result["completeness"]) == 2
        assert len(result["purity"]) == 2
        assert result["completeness"][0] == pytest.approx(1.0)

    def test_detect_sources_exposes_npix(self):
        image, _ = self.make_image()
        catalog = detect_sources(image, thresh=1.5)
        assert "npix" in catalog.dtype.names
        assert (np.asarray(catalog["npix"]) >= 5).all()
