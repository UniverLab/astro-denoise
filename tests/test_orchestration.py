"""End-to-end orchestration test against a fake method module.

A minimal ``astro_denoise_fake`` package is materialized on a temporary
PYTHONPATH so ``run_benchmark`` exercises the real subprocess CLI
protocol (``python -m astro_denoise_fake --input ... --output ...``)
without requiring any real denoiser venv.
"""

import json

import numpy as np
import pytest
from astropy.io import fits

from astro_denoise_bench.__main__ import main
from astro_denoise_bench.bench import run_benchmark, run_method_subprocess

FAKE_MAIN = """
import argparse, json
from pathlib import Path
import numpy as np
from astropy.io import fits

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--config", type=Path, default=None)
args = parser.parse_args()

with fits.open(args.input) as hdul:
    data = np.asarray(hdul[0].data, dtype=np.float64)
fits.PrimaryHDU(data.astype(np.float32)).writeto(args.output, overwrite=True)
print(json.dumps({"algorithm": "fake", "status": "ok"}))
"""


@pytest.fixture
def fake_method(tmp_path, monkeypatch):
    """Identity 'denoiser' importable as astro_denoise_fake via PYTHONPATH."""
    pkg = tmp_path / "fakelib" / "astro_denoise_fake"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "__main__.py").write_text(FAKE_MAIN)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "fakelib"))
    return "fake"


@pytest.fixture
def dataset(tmp_path):
    rng = np.random.default_rng(0)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for i in range(2):
        clean = np.full((32, 32), 100.0)
        noisy = clean + rng.normal(0, 2.0, clean.shape)
        fits.PrimaryHDU(clean.astype(np.float32)).writeto(data_dir / f"p{i}_clean.fits")
        fits.PrimaryHDU(noisy.astype(np.float32)).writeto(data_dir / f"p{i}_noisy.fits")
    return data_dir


def test_run_method_subprocess_protocol(fake_method, dataset, tmp_path):
    status = run_method_subprocess(fake_method, dataset / "p0_noisy.fits", tmp_path / "out.fits")
    assert status == {"algorithm": "fake", "status": "ok"}
    assert (tmp_path / "out.fits").exists()


def test_run_method_subprocess_missing_module_raises(dataset, tmp_path):
    with pytest.raises(RuntimeError):
        run_method_subprocess("does_not_exist", dataset / "p0_noisy.fits", tmp_path / "out.fits")


def test_run_benchmark_full_report(fake_method, dataset, tmp_path):
    report = run_benchmark(fake_method, dataset, output_dir=tmp_path / "out")
    assert report["algorithm"] == "fake"
    assert report["n_patches"] == 2
    assert report["summary"]["n_ok"] == 2
    assert report["summary"]["n_failed"] == 0
    for run in report["runs"]:
        assert run["status"] == "ok"
        # Identity denoiser against the clean reference still yields metrics.
        assert "sky_sigma" in run["metrics"]
        assert "psnr" in run["metrics"]


def test_run_benchmark_empty_dir_raises(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        run_benchmark("fake", empty)


def test_cli_writes_report_and_status(fake_method, dataset, tmp_path, capsys):
    report_path = tmp_path / "report.json"
    rc = main(
        [
            "--algo",
            fake_method,
            "--data-dir",
            str(dataset),
            "--output",
            str(report_path),
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 0
    status = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert status["status"] == "ok"
    report = json.loads(report_path.read_text())
    assert report["summary"]["n_ok"] == 2


def test_cli_list_emits_json(capsys):
    assert main(["--list"]) == 0
    status = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert status["status"] == "ok"
    assert isinstance(status["methods"], list)
