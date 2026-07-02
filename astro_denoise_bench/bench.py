"""Benchmark orchestrator for astro-denoise method modules.

Responsibilities:

* **Discovery** — find installed algorithm modules through the
  ``astro_denoise.methods`` entry-point group (packages named
  ``astro-denoise-*`` register their ``DenoiseMethod`` there).
* **Execution** — run one algorithm over every FITS patch in a data
  directory. Methods are invoked as isolated subprocesses through the
  standard CLI protocol (``python -m astro_denoise_<algo> --input ...
  --output ... --config ...``), so each module keeps its own venv; if
  the module is importable in-process (entry point available), it can
  also run in-process.
* **Evaluation** — compute image-quality metrics against the clean
  reference patch when one exists. Uses ``astro_denoise_metrics`` when
  importable, otherwise a built-in numpy PSNR / sky-sigma fallback.
* **Reporting** — consolidate per-patch timing and metrics into a
  comparative JSON report.

Dataset layout: ``<data-dir>/*_noisy.fits`` patches, each optionally
paired with ``<name>_clean.fits``. Plain ``*.fits`` files (no suffix
convention) are treated as noisy patches without reference.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
from importlib.metadata import entry_points
from pathlib import Path

import numpy as np
from astropy.io import fits

ENTRY_POINT_GROUP = "astro_denoise.methods"


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------

def discover_methods():
    """Map algorithm name -> entry point for installed astro-denoise modules."""
    return {ep.name: ep for ep in entry_points(group=ENTRY_POINT_GROUP)}


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------

def collect_patches(data_dir):
    """Return (noisy_path, clean_path_or_None) pairs found in ``data_dir``."""
    data_dir = Path(data_dir)
    pairs = []
    for noisy in sorted(data_dir.glob("*_noisy.fits")):
        clean = noisy.with_name(noisy.name.replace("_noisy.fits", "_clean.fits"))
        pairs.append((noisy, clean if clean.exists() else None))
    if not pairs:
        already = {p for p, _ in pairs}
        for path in sorted(data_dir.glob("*.fits")):
            if path not in already and not path.name.endswith("_clean.fits"):
                pairs.append((path, None))
    return pairs


def load_image(path):
    with fits.open(path) as hdul:
        for hdu in hdul:
            if hdu.data is not None and np.asarray(hdu.data).ndim == 2:
                return np.asarray(hdu.data, dtype=np.float64)
    raise ValueError(f"{path}: no 2-D image HDU found")


# ---------------------------------------------------------------------------
# metrics (astro_denoise_metrics when available, numpy fallback otherwise)
# ---------------------------------------------------------------------------

def _fallback_psnr(original, denoised):
    mse = float(np.mean((original - denoised) ** 2))
    if mse == 0:
        return float("inf")
    rng = float(np.max(original) - np.min(original)) or 1.0
    return 10.0 * np.log10(rng**2 / mse)


def _fallback_sky_sigma(image):
    data = image[np.isfinite(image)]
    for _ in range(5):  # simple 3-sigma clipping
        mean, std = float(np.mean(data)), float(np.std(data))
        kept = data[np.abs(data - mean) < 3.0 * std]
        if len(kept) == len(data):
            break
        data = kept
    return float(np.std(data))


def evaluate_patch(denoised, clean):
    """Metrics for one denoised patch; richer if astro_denoise_metrics exists."""
    try:
        from astro_denoise_metrics import psnr, ssim, sky_sigma

        metrics = {"sky_sigma": sky_sigma(denoised)}
        if clean is not None:
            metrics["psnr"] = psnr(clean, denoised)
            metrics["ssim"] = ssim(clean, denoised)
        return metrics
    except ImportError:
        metrics = {"sky_sigma": _fallback_sky_sigma(denoised)}
        if clean is not None:
            metrics["psnr"] = _fallback_psnr(clean, denoised)
        return metrics


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------

def run_method_subprocess(algo, noisy, output, config=None, python=None,
                          timeout=3600):
    """Invoke a method module via the standard CLI protocol.

    ``python`` selects the interpreter (e.g. the module's own venv);
    defaults to the current one. Returns the parsed stdout JSON.
    """
    module = f"astro_denoise_{algo}"
    cmd = [python or sys.executable, "-m", module,
           "--input", str(noisy), "--output", str(output)]
    if config:
        cmd += ["--config", str(config)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        detail = proc.stdout.strip() or proc.stderr.strip()[-400:]
        raise RuntimeError(f"{module} failed (rc={proc.returncode}): {detail}")
    for line in reversed(proc.stdout.strip().splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise RuntimeError(f"{module} produced no JSON status on stdout")


def run_benchmark(algo, data_dir, output_dir=None, config=None, python=None,
                  progress=None):
    """Run ``algo`` over every patch in ``data_dir`` and gather metrics.

    Returns the full report dict. ``progress`` is an optional callback
    ``(index, total, patch_name)`` used by the CLI for display.
    """
    pairs = collect_patches(data_dir)
    if not pairs:
        raise FileNotFoundError(f"no FITS patches found in {data_dir}")

    output_dir = Path(output_dir or Path(data_dir) / f"denoised_{algo}")
    output_dir.mkdir(parents=True, exist_ok=True)

    discovered = sorted(discover_methods())
    runs = []
    for index, (noisy_path, clean_path) in enumerate(pairs, start=1):
        if progress:
            progress(index, len(pairs), noisy_path.name)
        out_path = output_dir / noisy_path.name.replace("_noisy", "_denoised")
        record = {"patch": noisy_path.name, "output": str(out_path)}
        start = time.perf_counter()
        try:
            status = run_method_subprocess(
                algo, noisy_path, out_path, config=config, python=python
            )
            record["wall_time_s"] = round(time.perf_counter() - start, 4)
            record["status"] = status.get("status", "ok")
            record["method_meta"] = status
            clean = load_image(clean_path) if clean_path else None
            record["metrics"] = evaluate_patch(load_image(out_path), clean)
        except Exception as exc:
            record["wall_time_s"] = round(time.perf_counter() - start, 4)
            record["status"] = "error"
            record["error"] = str(exc)
        runs.append(record)

    return {
        "module": "astro-denoise-bench",
        "algorithm": algo,
        "data_dir": str(data_dir),
        "discovered_methods": discovered,
        "n_patches": len(pairs),
        "runs": runs,
        "summary": summarize(runs),
    }


def summarize(runs):
    """Aggregate per-patch metrics into a comparative summary."""
    ok = [r for r in runs if r.get("status") == "ok"]
    summary = {
        "n_ok": len(ok),
        "n_failed": len(runs) - len(ok),
    }
    if ok:
        times = [r["wall_time_s"] for r in ok]
        summary["mean_wall_time_s"] = round(statistics.fmean(times), 4)
        summary["throughput_patches_per_s"] = (
            round(len(ok) / sum(times), 4) if sum(times) > 0 else None
        )
        for key in ("psnr", "ssim", "sky_sigma"):
            values = [r["metrics"][key] for r in ok
                      if key in r.get("metrics", {})
                      and np.isfinite(r["metrics"][key])]
            if values:
                summary[f"mean_{key}"] = round(statistics.fmean(values), 4)
    return summary
