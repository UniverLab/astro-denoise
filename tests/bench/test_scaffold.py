"""Tests for the zero-friction method scaffold."""

import json
import subprocess
import sys

import pytest

from astro_denoise_bench.scaffold import scaffold_method


class TestScaffoldStructure:
    def test_creates_complete_module(self, tmp_path):
        repo = scaffold_method("wavelet", out_dir=tmp_path)
        for rel in (
            "pyproject.toml",
            "src/astro_denoise_wavelet/__init__.py",
            "src/astro_denoise_wavelet/interface.py",
            "src/astro_denoise_wavelet/denoiser.py",
            "src/astro_denoise_wavelet/__main__.py",
            "tests/test_denoiser.py",
            "tests/test_cli.py",
            "README.md",
            ".github/workflows/ci.yml",
            "LICENSE",
            ".gitignore",
        ):
            assert (repo / rel).exists(), rel
        pyproject = (repo / "pyproject.toml").read_text()
        assert 'wavelet = "astro_denoise_wavelet.denoiser:WaveletDenoiser"' in pyproject
        assert 'where = ["src"]' in pyproject

    def test_rejects_invalid_names(self, tmp_path):
        for bad in ("Wavelet", "my-method", "2fast", ""):
            with pytest.raises(ValueError):
                scaffold_method(bad, out_dir=tmp_path)

    def test_refuses_to_overwrite(self, tmp_path):
        scaffold_method("wavelet", out_dir=tmp_path)
        with pytest.raises(FileExistsError):
            scaffold_method("wavelet", out_dir=tmp_path)


class TestScaffoldRunsOutOfTheBox:
    def test_generated_tests_pass_and_cli_speaks_protocol(self, tmp_path):
        """The whole zero-friction promise: scaffold -> tests green -> CLI ok."""
        repo = scaffold_method("wavelet", out_dir=tmp_path)
        env_path = str(repo / "src")

        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", str(repo / "tests")],
            capture_output=True,
            text=True,
            env={"PYTHONPATH": env_path, "PATH": "/usr/bin:/bin"},
            cwd=repo,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr

    def test_cli_matches_bench_protocol(self, tmp_path):
        import numpy as np
        from astropy.io import fits

        repo = scaffold_method("wavelet", out_dir=tmp_path)
        image = np.random.default_rng(0).normal(100.0, 2.0, (32, 32)).astype(np.float32)
        fits.PrimaryHDU(data=image).writeto(tmp_path / "in.fits")

        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "astro_denoise_wavelet",
                "--input",
                str(tmp_path / "in.fits"),
                "--output",
                str(tmp_path / "out.fits"),
            ],
            capture_output=True,
            text=True,
            env={"PYTHONPATH": str(repo / "src"), "PATH": "/usr/bin:/bin"},
        )
        assert proc.returncode == 0, proc.stderr
        status = json.loads(proc.stdout.strip().splitlines()[-1])
        assert status == {
            "algorithm": "wavelet",
            "version": "0.1.0",
            "status": "ok",
            "elapsed_s": status["elapsed_s"],
        }
        assert (tmp_path / "out.fits").exists()
