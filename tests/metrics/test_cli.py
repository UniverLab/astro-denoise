"""End-to-end test of the CLI protocol on a synthetic image + truth catalog."""

import json

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from astro_denoise_metrics.__main__ import main
from astro_denoise_metrics.detect import background_rms, detect_sources


def gaussian_source(shape, x, y, flux, fwhm=3.0):
    sigma = fwhm / 2.3548
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    profile = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))
    return flux * profile / (2 * np.pi * sigma**2)


@pytest.fixture
def synthetic_scene(tmp_path):
    """FITS image with 8 bright sources plus its truth catalog."""
    rng = np.random.default_rng(42)
    size, noise_sigma = 128, 2.0
    xs = rng.uniform(15, size - 15, 8)
    ys = rng.uniform(15, size - 15, 8)
    fluxes = rng.uniform(2000, 8000, 8)

    clean = np.full((size, size), 100.0)
    for x, y, f in zip(xs, ys, fluxes, strict=True):
        clean += gaussian_source((size, size), x, y, f)
    noisy = clean + rng.normal(0, noise_sigma, clean.shape)

    image_path = tmp_path / "denoised.fits"
    clean_path = tmp_path / "clean.fits"
    truth_path = tmp_path / "truth.fits"
    fits.PrimaryHDU(noisy.astype(np.float32)).writeto(image_path)
    fits.PrimaryHDU(clean.astype(np.float32)).writeto(clean_path)
    Table({"x": xs, "y": ys, "flux": fluxes}).write(truth_path)
    return image_path, clean_path, truth_path


def test_detect_sources_finds_bright_sources(synthetic_scene):
    image_path, _, truth_path = synthetic_scene
    with fits.open(image_path) as hdul:
        image = np.asarray(hdul[0].data, dtype=np.float64)
    cat = detect_sources(image, thresh=3.0, minarea=5)
    assert len(cat) >= 6  # bright synthetic sources are hard to miss
    assert set(cat.dtype.names) == {"x", "y", "flux", "npix"}


def test_reference_err_makes_threshold_comparable(synthetic_scene):
    """A background-flattening 'denoiser' must not detect deeper for free.

    The over-smoothed image keeps faint correlated wiggles that its own
    (tiny) RMS promotes to detections; thresholding against the noisy
    input's RMS keeps only the real sources.
    """
    from scipy.ndimage import gaussian_filter

    image_path, clean_path, _ = synthetic_scene
    with fits.open(image_path) as hdul:
        noisy = np.asarray(hdul[0].data, dtype=np.float64)
    with fits.open(clean_path) as hdul:
        clean = np.asarray(hdul[0].data, dtype=np.float64)
    rng = np.random.default_rng(7)
    smoothed = clean + gaussian_filter(rng.normal(0, 1.0, clean.shape), 2.0) * 0.3

    ref_rms = background_rms(noisy)
    assert 1.0 < ref_rms < 3.0  # scene noise sigma is 2.0

    self_cat = detect_sources(smoothed, thresh=1.5)
    ref_cat = detect_sources(smoothed, thresh=1.5, err=ref_rms)
    assert len(ref_cat) >= 6  # real sources survive the comparable threshold
    assert len(ref_cat) < len(self_cat)  # spurious deep detections are gone


def test_cli_writes_structured_results(synthetic_scene, tmp_path, capsys):
    image_path, clean_path, truth_path = synthetic_scene
    output = tmp_path / "results.json"
    rc = main(
        [
            "--input",
            str(image_path),
            "--truth",
            str(truth_path),
            "--original",
            str(clean_path),
            "--output",
            str(output),
            "--thresh",
            "3.0",
        ]
    )
    assert rc == 0

    status = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert status["status"] == "ok"

    results = json.loads(output.read_text())
    assert results["n_truth"] == 8
    assert results["completeness"] >= 0.75
    assert 0.0 < results["purity"] <= 1.0
    assert results["psnr"] > 20.0
    assert 0.0 < results["ssim"] <= 1.0
    assert results["photometric_accuracy"]["n_matched"] > 0
    assert "completeness_vs_mag" in results


def test_cli_thresh_image_records_reference_rms(synthetic_scene, tmp_path, capsys):
    image_path, _, truth_path = synthetic_scene
    output = tmp_path / "results.json"
    rc = main(
        [
            "--input",
            str(image_path),
            "--truth",
            str(truth_path),
            "--output",
            str(output),
            "--thresh",
            "3.0",
            "--thresh-image",
            str(image_path),
            "--thresh-sweep",
            "1.5,3.0",
        ]
    )
    assert rc == 0
    results = json.loads(output.read_text())
    assert results["parameters"]["thresh_image"] == str(image_path)
    assert 1.0 < results["parameters"]["thresh_rms"] < 3.0  # scene noise sigma is 2.0
    assert results["completeness"] >= 0.75


def test_cli_reports_error_on_missing_input(tmp_path, capsys):
    rc = main(
        [
            "--input",
            str(tmp_path / "missing.fits"),
            "--truth",
            str(tmp_path / "missing_truth.fits"),
            "--output",
            str(tmp_path / "out.json"),
        ]
    )
    assert rc == 1
    status = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert status["status"] == "error"


def test_cli_explain_lists_all_metrics(capsys):
    from astro_denoise_metrics.__main__ import main
    from astro_denoise_metrics.explain import METRIC_INFO

    assert main(["--explain"]) == 0
    out = capsys.readouterr().out
    for key in METRIC_INFO:
        assert key.split(" / ")[0] in out


def test_cli_missing_args_without_explain_errors():
    import pytest

    from astro_denoise_metrics.__main__ import main

    with pytest.raises(SystemExit):
        main([])
