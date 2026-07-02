"""Unit tests for dataset collection, fallback metrics and summaries."""

import numpy as np
import pytest
from astropy.io import fits

from astro_denoise_bench.bench import (
    _fallback_psnr,
    _fallback_sky_sigma,
    collect_patches,
    load_image,
    summarize,
)


def write_fits(path, data=None):
    data = data if data is not None else np.zeros((16, 16), dtype=np.float32)
    fits.PrimaryHDU(data).writeto(path)


class TestCollectPatches:
    def test_pairs_noisy_with_clean(self, tmp_path):
        write_fits(tmp_path / "a_noisy.fits")
        write_fits(tmp_path / "a_clean.fits")
        write_fits(tmp_path / "b_noisy.fits")  # unpaired
        pairs = collect_patches(tmp_path)
        assert [(n.name, c.name if c else None) for n, c in pairs] == [
            ("a_noisy.fits", "a_clean.fits"),
            ("b_noisy.fits", None),
        ]

    def test_plain_fits_fallback(self, tmp_path):
        write_fits(tmp_path / "img1.fits")
        write_fits(tmp_path / "img2.fits")
        pairs = collect_patches(tmp_path)
        assert [(n.name, c) for n, c in pairs] == [
            ("img1.fits", None),
            ("img2.fits", None),
        ]

    def test_empty_dir(self, tmp_path):
        assert collect_patches(tmp_path) == []


class TestLoadImage:
    def test_reads_first_2d_hdu(self, tmp_path):
        data = np.arange(16, dtype=np.float32).reshape(4, 4)
        write_fits(tmp_path / "img.fits", data)
        np.testing.assert_array_equal(load_image(tmp_path / "img.fits"), data)

    def test_no_image_hdu_raises(self, tmp_path):
        fits.PrimaryHDU().writeto(tmp_path / "empty.fits")
        with pytest.raises(ValueError):
            load_image(tmp_path / "empty.fits")


class TestFallbackMetrics:
    def test_psnr_identity_is_inf(self):
        image = np.random.default_rng(0).normal(size=(32, 32))
        assert np.isinf(_fallback_psnr(image, image))

    def test_psnr_orders_noise_levels(self):
        rng = np.random.default_rng(1)
        image = rng.normal(100, 10, size=(64, 64))
        small = _fallback_psnr(image, image + rng.normal(0, 0.1, image.shape))
        large = _fallback_psnr(image, image + rng.normal(0, 1.0, image.shape))
        assert small > large

    def test_sky_sigma_recovers_noise(self):
        rng = np.random.default_rng(2)
        image = rng.normal(0, 3.0, size=(256, 256))
        assert _fallback_sky_sigma(image) == pytest.approx(3.0, rel=0.1)


class TestSummarize:
    def test_aggregates_ok_runs(self):
        runs = [
            {"status": "ok", "wall_time_s": 1.0, "metrics": {"psnr": 30.0, "sky_sigma": 1.0}},
            {"status": "ok", "wall_time_s": 3.0, "metrics": {"psnr": 40.0, "sky_sigma": 3.0}},
            {"status": "error", "wall_time_s": 0.1},
        ]
        summary = summarize(runs)
        assert summary["n_ok"] == 2
        assert summary["n_failed"] == 1
        assert summary["mean_wall_time_s"] == pytest.approx(2.0)
        assert summary["throughput_patches_per_s"] == pytest.approx(0.5)
        assert summary["mean_psnr"] == pytest.approx(35.0)
        assert summary["mean_sky_sigma"] == pytest.approx(2.0)

    def test_all_failed(self):
        summary = summarize([{"status": "error"}])
        assert summary == {"n_ok": 0, "n_failed": 1}

    def test_skips_non_finite_metrics(self):
        runs = [
            {"status": "ok", "wall_time_s": 1.0, "metrics": {"psnr": float("inf")}},
            {"status": "ok", "wall_time_s": 1.0, "metrics": {"psnr": 20.0}},
        ]
        assert summarize(runs)["mean_psnr"] == pytest.approx(20.0)
