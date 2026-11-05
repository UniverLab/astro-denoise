"""Tests for the DC2 truth-catalog adapter (Parquet + WCS projection)."""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from astro_denoise_metrics.truth import (
    AB_ZEROPOINT_NJY,
    flux_njy_to_mag,
    load_dc2_truth,
    project_truth,
)


def make_wcs(shape, pixel_scale_deg=5.5e-5):
    """Simple TAN WCS centered on the image."""
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crpix = [shape[1] / 2, shape[0] / 2]
    wcs.wcs.crval = [56.65, -36.45]
    wcs.wcs.cd = [[-pixel_scale_deg, 0.0], [0.0, pixel_scale_deg]]
    return wcs


def write_parquet(path, ra, dec, flux, band="z", unique=None):
    columns = {
        "ra": pa.array(ra, type=pa.float64()),
        "dec": pa.array(dec, type=pa.float64()),
        f"flux_{band}": pa.array(flux, type=pa.float64()),
    }
    if unique is not None:
        columns["is_unique_truth_entry"] = pa.array(unique, type=pa.bool_())
    pq.write_table(pa.table(columns), path)


class TestFluxToMag:
    def test_zeropoint(self):
        assert flux_njy_to_mag([1.0]) == pytest.approx(AB_ZEROPOINT_NJY)

    def test_brighter_is_smaller_mag(self):
        mags = flux_njy_to_mag([10.0, 1000.0])
        assert mags[1] < mags[0]

    def test_nonpositive_flux_is_nan(self):
        mags = flux_njy_to_mag([0.0, -5.0])
        assert np.isnan(mags).all()


class TestLoadDc2Truth:
    def test_reads_band_flux(self, tmp_path):
        path = tmp_path / "truth.parquet"
        write_parquet(path, ra=[56.65], dec=[-36.45], flux=[100.0], band="z")
        truth = load_dc2_truth(path, "z")
        assert truth["flux"] == pytest.approx([100.0])

    def test_drops_duplicate_entries(self, tmp_path):
        path = tmp_path / "truth.parquet"
        write_parquet(
            path,
            ra=[56.65, 56.66],
            dec=[-36.45, -36.46],
            flux=[100.0, 200.0],
            unique=[True, False],
        )
        truth = load_dc2_truth(path, "z")
        assert len(truth["ra"]) == 1
        assert truth["flux"] == pytest.approx([100.0])

    def test_unknown_band_raises(self, tmp_path):
        with pytest.raises(ValueError, match="unknown DC2 band"):
            load_dc2_truth(tmp_path / "truth.parquet", "q")

    def test_missing_columns_raises(self, tmp_path):
        path = tmp_path / "truth.parquet"
        write_parquet(path, ra=[56.65], dec=[-36.45], flux=[100.0], band="r")
        with pytest.raises(ValueError, match="missing DC2 truth columns"):
            load_dc2_truth(path, "z")


class TestProjectTruth:
    def test_roundtrip_recovers_pixels(self):
        shape = (128, 128)
        wcs = make_wcs(shape)
        px = np.array([10.0, 64.0, 100.0])
        py = np.array([20.0, 64.0, 90.0])
        ra, dec = wcs.pixel_to_world_values(px, py)
        truth = {"ra": np.asarray(ra), "dec": np.asarray(dec), "flux": np.full(3, 500.0)}

        projected = project_truth(truth, wcs, shape)
        assert projected["x"] == pytest.approx(px, abs=1e-6)
        assert projected["y"] == pytest.approx(py, abs=1e-6)

    def test_filters_out_of_bounds(self):
        shape = (64, 64)
        wcs = make_wcs(shape)
        ra, dec = wcs.pixel_to_world_values([32.0, 500.0], [32.0, 500.0])
        truth = {"ra": np.asarray(ra), "dec": np.asarray(dec), "flux": np.full(2, 500.0)}

        projected = project_truth(truth, wcs, shape)
        assert len(projected["x"]) == 1

    def test_mag_limit_drops_faint_sources(self):
        shape = (64, 64)
        wcs = make_wcs(shape)
        ra, dec = wcs.pixel_to_world_values([20.0, 40.0], [20.0, 40.0])
        # 1e4 nJy -> mag 21.4 (kept); 10 nJy -> mag 28.9 (dropped at 26)
        truth = {
            "ra": np.asarray(ra),
            "dec": np.asarray(dec),
            "flux": np.array([1e4, 10.0]),
        }

        projected = project_truth(truth, wcs, shape, mag_limit=26.0)
        assert len(projected["x"]) == 1
        assert projected["mag"][0] == pytest.approx(21.4, abs=0.01)


class TestCliParquet:
    def gaussian(self, shape, x, y, flux, fwhm=3.0):
        sigma = fwhm / 2.3548
        yy, xx = np.mgrid[: shape[0], : shape[1]]
        return flux * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))

    def test_cli_end_to_end_with_parquet_truth(self, tmp_path):
        from astro_denoise_metrics.__main__ import main

        rng = np.random.default_rng(7)
        shape = (128, 128)
        wcs = make_wcs(shape)
        px = np.array([30.0, 64.0, 100.0])
        py = np.array([40.0, 64.0, 85.0])

        image = np.full(shape, 100.0)
        for x, y in zip(px, py, strict=True):
            image += self.gaussian(shape, x, y, flux=5000.0)
        image += rng.normal(0, 2.0, shape)

        header = wcs.to_header()
        image_path = tmp_path / "denoised.fits"
        fits.PrimaryHDU(data=image.astype(np.float32), header=header).writeto(image_path)

        ra, dec = wcs.pixel_to_world_values(px, py)
        truth_path = tmp_path / "truth.parquet"
        write_parquet(truth_path, ra=list(ra), dec=list(dec), flux=[1e4] * 3, band="z")

        out_path = tmp_path / "results.json"
        rc = main(
            [
                "--input",
                str(image_path),
                "--truth",
                str(truth_path),
                "--band",
                "z",
                "--output",
                str(out_path),
            ]
        )
        assert rc == 0
        import json

        results = json.loads(out_path.read_text())
        assert results["n_truth"] == 3
        assert results["completeness"] == pytest.approx(1.0)

    def test_cli_parquet_without_band_errors(self, tmp_path):
        from astro_denoise_metrics.__main__ import main

        wcs = make_wcs((32, 32))
        image_path = tmp_path / "denoised.fits"
        fits.PrimaryHDU(data=np.zeros((32, 32), dtype=np.float32), header=wcs.to_header()).writeto(
            image_path
        )
        truth_path = tmp_path / "truth.parquet"
        write_parquet(truth_path, ra=[56.65], dec=[-36.45], flux=[100.0])

        rc = main(
            [
                "--input",
                str(image_path),
                "--truth",
                str(truth_path),
                "--output",
                str(tmp_path / "results.json"),
            ]
        )
        assert rc == 1
