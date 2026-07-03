"""Tests for DC2 curation, the curated benchmark runner and TUI helpers."""

import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from astro_denoise_bench.bench import run_curated_benchmark, summarize_curated
from astro_denoise_bench.dc2 import curate, discover_coadds, grid_positions, load_manifest
from astro_denoise_bench.tui import comparison_rows, dataset_rows, method_rows


def make_wcs(shape):
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crpix = [shape[1] / 2, shape[0] / 2]
    wcs.wcs.crval = [56.65, -36.45]
    wcs.wcs.cd = [[-5.5e-5, 0.0], [0.0, 5.5e-5]]
    return wcs


@pytest.fixture
def coadd_tree(tmp_path):
    """Miniature DC2 deepCoadd tree: 2 bands x 1 tract, with variance HDU."""
    rng = np.random.default_rng(0)
    shape = (200, 200)
    for band in ("z", "r"):
        target = tmp_path / "coadd" / "deepCoadd" / band / "3828"
        target.mkdir(parents=True)
        image = rng.normal(100.0, 2.0, shape).astype(np.float32)
        variance = np.full(shape, 4.0, dtype=np.float32)
        hdus = fits.HDUList(
            [
                fits.PrimaryHDU(),
                fits.ImageHDU(data=image, header=make_wcs(shape).to_header()),
                fits.ImageHDU(data=np.zeros(shape, dtype=np.int32)),
                fits.ImageHDU(data=variance),
            ]
        )
        hdus.writeto(target / "2,2.fits")
        # nImage companions must be ignored by discovery
        fits.PrimaryHDU(data=np.ones(shape, dtype=np.int32)).writeto(target / "2,2_nImage.fits")
    truth_root = tmp_path / "truth"
    truth_root.mkdir()
    (truth_root / "truth_tract3828.parquet").write_bytes(b"placeholder")
    return tmp_path / "coadd", truth_root


class TestDiscoverAndGrid:
    def test_discovers_images_not_nimage(self, coadd_tree):
        coadd_root, _ = coadd_tree
        found = discover_coadds(coadd_root)
        assert len(found) == 2
        assert {f["band"] for f in found} == {"z", "r"}
        assert all("nImage" not in f["path"] for f in found)

    def test_band_filter(self, coadd_tree):
        coadd_root, _ = coadd_tree
        assert len(discover_coadds(coadd_root, bands=["z"])) == 1

    def test_grid_positions_deterministic_and_inside(self):
        a = grid_positions((200, 200), 64, 3)
        b = grid_positions((200, 200), 64, 3)
        assert a == b
        for x, y in a:
            assert 32 <= x <= 168
            assert 32 <= y <= 168


class TestCurate:
    def test_builds_manifest_and_regions(self, coadd_tree, tmp_path):
        coadd_root, truth_root = coadd_tree
        out = tmp_path / "curated"
        manifest = curate(coadd_root, truth_root, out, size=64, regions_per_image=2)

        assert manifest["n_regions"] == 4  # 2 bands x 2 regions
        assert manifest["bands"] == ["r", "z"]
        loaded = load_manifest(out / "manifest.json")
        for region in loaded["regions"]:
            with fits.open(region["path"]) as hdul:
                assert hdul[0].data.shape == (64, 64)
                assert region["has_variance"]
                assert hdul["VARIANCE"].data.shape == (64, 64)
            assert region["truth"].endswith("truth_tract3828.parquet")

    def test_cutout_wcs_matches_parent(self, coadd_tree, tmp_path):
        coadd_root, truth_root = coadd_tree
        out = tmp_path / "curated"
        manifest = curate(coadd_root, truth_root, out, size=64, regions_per_image=1)
        region = load_manifest(out / "manifest.json")["regions"][0]

        parent = discover_coadds(coadd_root, bands=[region["band"]])[0]
        with fits.open(parent["path"]) as hdul:
            parent_wcs = WCS(hdul[1].header)
        with fits.open(region["path"]) as hdul:
            cut_wcs = WCS(hdul[0].header)
        # the cutout center pixel must map to the same sky position
        world_cut = cut_wcs.pixel_to_world_values(31.5, 31.5)
        world_parent = parent_wcs.pixel_to_world_values(
            region["center_x"] - 0.5, region["center_y"] - 0.5
        )
        assert float(world_cut[0]) == pytest.approx(float(world_parent[0]), abs=1e-6)
        assert manifest["n_regions"] == 2


FAKE_METHOD = """
import json, sys
from pathlib import Path
from astropy.io import fits
import numpy as np

def main():
    args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
    with fits.open(args["--input"]) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float32)
        header = hdul[0].header
    fits.PrimaryHDU(data=data, header=header).writeto(args["--output"], overwrite=True)
    print(json.dumps({"algorithm": "fake", "version": "0", "status": "ok"}))

main()
"""

FAKE_METRICS = """
import json, sys

def main():
    args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
    results = {
        "module": "astro-denoise-metrics", "version": "0",
        "completeness": 0.8, "purity": 0.9, "sky_sigma": 1.5,
        "sharpness_sf": 0.4, "flux_snr": 12.0,
        "n_detected": 10, "n_truth": 12,
        "band": args.get("--band"),
    }
    from pathlib import Path
    Path(args["--output"]).write_text(json.dumps(results))
    print(json.dumps({"module": "astro-denoise-metrics", "status": "ok"}))

main()
"""


@pytest.fixture
def fake_modules(tmp_path, monkeypatch):
    """Materialize fake method + metrics modules on a temp PYTHONPATH."""
    root = tmp_path / "modules"
    for name, body in (
        ("astro_denoise_fake", FAKE_METHOD),
        ("astro_denoise_metrics", FAKE_METRICS),
    ):
        pkg = root / name
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text("")
        (pkg / "__main__.py").write_text(textwrap.dedent(body))
    monkeypatch.setenv("PYTHONPATH", str(root))
    return root


class TestCuratedBenchmark:
    def test_runs_regions_and_groups_by_band(self, coadd_tree, tmp_path, fake_modules):
        coadd_root, truth_root = coadd_tree
        out = tmp_path / "curated"
        curate(coadd_root, truth_root, out, size=64, regions_per_image=2)
        manifest = load_manifest(out / "manifest.json")

        report = run_curated_benchmark(
            manifest,
            "fake",
            out_dir=out,
            method_python=sys.executable,
            metrics_python=sys.executable,
            previews=True,
        )
        assert report["n_regions"] == 4
        assert report["summary"]["n_failed"] == 0
        assert report["summary"]["mean_completeness"] == pytest.approx(0.8)
        assert set(report["summary"]["per_band"]) == {"r", "z"}
        for run in report["runs"]:
            assert run["metrics"]["purity"] == pytest.approx(0.9)
            assert Path(run["preview"]).exists()

    def test_band_filter_limits_regions(self, coadd_tree, tmp_path, fake_modules):
        coadd_root, truth_root = coadd_tree
        out = tmp_path / "curated"
        curate(coadd_root, truth_root, out, size=64, regions_per_image=1)
        manifest = load_manifest(out / "manifest.json")

        report = run_curated_benchmark(
            manifest,
            "fake",
            out_dir=out,
            method_python=sys.executable,
            metrics_python=sys.executable,
            bands=["z"],
        )
        assert report["n_regions"] == 1
        assert report["bands"] == ["z"]

    def test_summarize_curated_handles_errors(self):
        runs = [
            {"status": "ok", "band": "z", "wall_time_s": 1.0, "metrics": {"completeness": 0.5}},
            {"status": "error", "band": "z", "wall_time_s": 0.1},
        ]
        summary = summarize_curated(runs)
        assert summary["n_ok"] == 1
        assert summary["n_failed"] == 1
        assert summary["per_band"]["z"]["mean_completeness"] == pytest.approx(0.5)


class TestTuiHelpers:
    def test_method_rows_reports_on_disk_repos(self, tmp_path):
        repo = tmp_path / "astro-denoise-wavelet"
        (repo / "src" / "astro_denoise_wavelet").mkdir(parents=True)
        (repo / ".venv").mkdir()
        rows = method_rows(code_root=tmp_path)
        names = [r[0] for r in rows]
        assert "wavelet" in names
        row = rows[names.index("wavelet")]
        assert row[1] == "on disk"
        assert row[3] == "venv ready"

    def test_dataset_rows(self, coadd_tree, tmp_path):
        coadd_root, truth_root = coadd_tree
        out = tmp_path / "curated"
        curate(coadd_root, truth_root, out, size=64, regions_per_image=1)
        manifest = load_manifest(out / "manifest.json")
        rows = dataset_rows(manifest)
        assert len(rows) == 2
        assert rows[0][4] == "yes"  # truth available

    def test_comparison_rows_reads_curated_reports(self, tmp_path):
        report = {
            "kind": "curated-benchmark",
            "algorithm": "bm3d",
            "summary": {
                "n_ok": 4,
                "mean_completeness": 0.5,
                "mean_purity": 0.6,
                "mean_sky_sigma": 0.01,
                "mean_sharpness_sf": 0.001,
                "mean_flux_snr": 20.0,
                "mean_wall_time_s": 1.0,
                "per_band": {
                    "z": {"n_ok": 2, "mean_completeness": 0.55},
                },
            },
        }
        path = tmp_path / "report_bm3d.json"
        path.write_text(json.dumps(report))
        rows = comparison_rows([path])
        assert len(rows) == 2  # all + z
        assert rows[0][0] == "bm3d"
        assert rows[1][1] == "z"
