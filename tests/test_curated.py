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
from astro_denoise_bench.dc2 import curate, discover_coadds, load_manifest, random_positions
from astro_denoise_bench.tui import (
    comparison_rows,
    dataset_rows,
    dataset_summary,
    method_rows,
    report_for_algo,
    report_region_rows,
)


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
    # real parquet: sources covering the image sky area plus far-away ones
    # that trimming must drop (image spans ~0.011 deg around crval)
    import pyarrow as pa
    import pyarrow.parquet as pq

    n_inside = 50
    ra_in = rng.uniform(56.645, 56.655, n_inside)
    dec_in = rng.uniform(-36.455, -36.445, n_inside)
    ra = np.concatenate([ra_in, [57.65, 55.65]])  # 2 far outside
    dec = np.concatenate([dec_in, [-36.45, -36.45]])
    table = pa.table(
        {
            "ra": pa.array(ra, type=pa.float64()),
            "dec": pa.array(dec, type=pa.float64()),
            "flux_z": pa.array(rng.uniform(10, 1e4, len(ra)), type=pa.float64()),
            "flux_r": pa.array(rng.uniform(10, 1e4, len(ra)), type=pa.float64()),
            "is_unique_truth_entry": pa.array([True] * len(ra), type=pa.bool_()),
            "mag_r": pa.array(rng.uniform(18, 28, len(ra)), type=pa.float64()),
        }
    )
    pq.write_table(table, truth_root / "truth_tract3828.parquet")
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

    def test_random_positions_seeded_and_inside(self):
        a = random_positions((200, 200), 64, 5, np.random.default_rng(42))
        b = random_positions((200, 200), 64, 5, np.random.default_rng(42))
        c = random_positions((200, 200), 64, 5, np.random.default_rng(43))
        assert a == b  # same seed -> same block
        assert a != c  # different seed -> different sky regions
        for x, y in a:
            assert 32 <= x <= 168
            assert 32 <= y <= 168

    def test_curate_records_seed_and_is_reproducible(self, coadd_tree, tmp_path):
        coadd_root, truth_root = coadd_tree
        m1 = curate(coadd_root, truth_root, tmp_path / "c1", size=64, seed=7)
        m2 = curate(coadd_root, truth_root, tmp_path / "c2", size=64, seed=7)
        m3 = curate(coadd_root, truth_root, tmp_path / "c3", size=64)  # fresh seed

        assert m1["seed"] == 7
        centers1 = [(r["center_x"], r["center_y"]) for r in m1["regions"]]
        centers2 = [(r["center_x"], r["center_y"]) for r in m2["regions"]]
        centers3 = [(r["center_x"], r["center_y"]) for r in m3["regions"]]
        assert centers1 == centers2  # same seed -> identical block
        assert centers1 != centers3  # fresh seed -> different block
        assert isinstance(m3["seed"], int)


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

    def test_block_is_self_contained_and_trimmed(self, coadd_tree, tmp_path):
        import pyarrow.parquet as pq

        coadd_root, truth_root = coadd_tree
        out = tmp_path / "curated"
        curate(coadd_root, truth_root, out, size=64, regions_per_image=2)

        raw = json.loads((out / "manifest.json").read_text())
        for region in raw["regions"]:
            # portable: relative paths inside the block, no machine-specific roots
            assert region["truth"] == "truth/truth_tract3828.parquet"
            assert not Path(region["path"]).is_absolute()

        trimmed = pq.read_table(out / "truth" / "truth_tract3828.parquet")
        source = pq.read_table(truth_root / "truth_tract3828.parquet")
        assert trimmed.num_rows < source.num_rows  # far-away sources dropped
        assert trimmed.num_rows > 0
        names = set(trimmed.schema.names)
        assert {"ra", "dec", "flux_z", "flux_r", "is_unique_truth_entry"} <= names
        assert "mag_r" not in names  # non-essential columns stripped

        # resolved truth must sit inside the block, ready for distribution
        loaded = load_manifest(out / "manifest.json")
        assert loaded["regions"][0]["truth"].startswith(str(out))

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

        # grid consistency: cutout pixel (0,0) lands on an integer parent
        # pixel, and offsets propagate 1:1 (Cutout2D snaps to the grid)
        ra0, dec0 = cut_wcs.pixel_to_world_values(0.0, 0.0)
        px0, py0 = parent_wcs.world_to_pixel_values(ra0, dec0)
        assert float(px0) == pytest.approx(round(float(px0)), abs=1e-6)
        ra1, dec1 = cut_wcs.pixel_to_world_values(10.0, 20.0)
        px1, py1 = parent_wcs.world_to_pixel_values(ra1, dec1)
        assert float(px1 - px0) == pytest.approx(10.0, abs=1e-6)
        assert float(py1 - py0) == pytest.approx(20.0, abs=1e-6)
        # and the cutout window sits around the requested center
        assert abs(float(px0) + 32 - region["center_x"]) <= 1.0
        assert abs(float(py0) + 32 - region["center_y"]) <= 1.0
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
        assert rows[0][5] == "yes"  # truth available

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


class TestHfDistribution:
    def test_fetch_validates_manifest_presence(self, tmp_path, monkeypatch):
        import sys
        import types

        def fake_snapshot_download(repo_id, repo_type, revision, token, local_dir):
            Path(local_dir).mkdir(parents=True, exist_ok=True)
            return local_dir

        stub = types.SimpleNamespace(snapshot_download=fake_snapshot_download)
        monkeypatch.setitem(sys.modules, "huggingface_hub", stub)
        from astro_denoise_bench.hf import fetch_curated

        with pytest.raises(FileNotFoundError, match="manifest.json"):
            fetch_curated("user/astro-denoise-curated", tmp_path / "dl")

    def test_fetch_returns_manifest_path(self, tmp_path, monkeypatch):
        import sys
        import types

        def fake_snapshot_download(repo_id, repo_type, revision, token, local_dir):
            Path(local_dir).mkdir(parents=True, exist_ok=True)
            (Path(local_dir) / "manifest.json").write_text("{}")
            return local_dir

        stub = types.SimpleNamespace(snapshot_download=fake_snapshot_download)
        monkeypatch.setitem(sys.modules, "huggingface_hub", stub)
        from astro_denoise_bench.hf import fetch_curated

        manifest = fetch_curated("user/astro-denoise-curated", tmp_path / "dl")
        assert manifest.name == "manifest.json"
        assert manifest.exists()

    def test_push_requires_manifest(self, tmp_path):
        from astro_denoise_bench.hf import push_curated

        with pytest.raises(FileNotFoundError, match="curate first"):
            push_curated("user/astro-denoise-curated", tmp_path)

    def test_push_uploads_block_and_skips_run_outputs(self, tmp_path, monkeypatch):
        import sys
        import types

        calls = {}

        class FakeApi:
            def __init__(self, token=None):
                calls["token"] = token

            def create_repo(self, repo_id, repo_type, private, exist_ok):
                calls["create"] = (repo_id, repo_type, exist_ok)

            def upload_folder(
                self, folder_path, repo_id, repo_type, commit_message, ignore_patterns
            ):
                calls["upload"] = (folder_path, repo_id, ignore_patterns)

        stub = types.SimpleNamespace(HfApi=FakeApi)
        monkeypatch.setitem(sys.modules, "huggingface_hub", stub)
        from astro_denoise_bench.hf import push_curated

        (tmp_path / "manifest.json").write_text("{}")
        url = push_curated("user/astro-denoise-curated", tmp_path)
        assert url.endswith("user/astro-denoise-curated")
        assert calls["create"][1] == "dataset"
        assert "denoised_*" in calls["upload"][2]


class TestTuiDrilldown:
    def make_report(self, tmp_path, algo="bm3d"):
        report = {
            "kind": "curated-benchmark",
            "algorithm": algo,
            "summary": {"n_ok": 1, "per_band": {}},
            "runs": [
                {
                    "region": "t3828-p22-z-r0",
                    "band": "z",
                    "status": "ok",
                    "wall_time_s": 1.2,
                    "peak_rss_mb": 150.0,
                    "metrics": {"completeness": 0.5, "purity": 0.6},
                }
            ],
        }
        path = tmp_path / f"report_{algo}.json"
        path.write_text(json.dumps(report))
        return path

    def test_report_for_algo_finds_matching_report(self, tmp_path):
        p1 = self.make_report(tmp_path, "bm3d")
        p2 = self.make_report(tmp_path, "unet")
        assert report_for_algo([p1, p2], "unet") == p2
        assert report_for_algo([p1, p2], "missing") is None

    def test_report_region_rows(self, tmp_path):
        path = self.make_report(tmp_path)
        rows = report_region_rows(path)
        assert len(rows) == 1
        region, band, status, completeness, *_rest = rows[0]
        assert (region, band, status) == ("t3828-p22-z-r0", "z", "ok")
        assert completeness == "0.5"

    def test_dataset_summary_includes_seed(self):
        manifest = {
            "n_regions": 4,
            "bands": ["r", "z"],
            "tracts": [3828],
            "size": 64,
            "seed": 12345,
            "regions": [{}] * 4,
        }
        line = dataset_summary(manifest)
        assert "seed: 12345" in line
        assert "4 regions" in line
