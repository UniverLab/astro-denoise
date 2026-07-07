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
        from astro_denoise_metrics import psnr, sky_sigma, ssim

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


_GNU_TIME = "/usr/bin/time"


def _run_json_subprocess(cmd, timeout, label):
    """Run a protocol subprocess and parse its stdout JSON status line.

    When GNU time is available the command is wrapped to record the
    child's peak RSS, attached to the status as ``peak_rss_mb``.
    """
    import os
    import tempfile

    mem_path = None
    if os.path.exists(_GNU_TIME):
        fd, mem_path = tempfile.mkstemp(suffix=".rss")
        os.close(fd)
        cmd = [_GNU_TIME, "-f", "%M", "-o", mem_path, *cmd]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        peak_rss_mb = None
        if mem_path is not None:
            raw = Path(mem_path).read_text().strip().splitlines()
            if raw and raw[-1].isdigit():
                peak_rss_mb = round(int(raw[-1]) / 1024, 1)
    finally:
        if mem_path is not None:
            Path(mem_path).unlink(missing_ok=True)

    if proc.returncode != 0:
        detail = proc.stdout.strip() or proc.stderr.strip()[-400:]
        raise RuntimeError(f"{label} failed (rc={proc.returncode}): {detail}")
    for line in reversed(proc.stdout.strip().splitlines()):
        try:
            status = json.loads(line)
        except json.JSONDecodeError:
            continue
        if peak_rss_mb is not None:
            status["peak_rss_mb"] = peak_rss_mb
        return status
    raise RuntimeError(f"{label} produced no JSON status on stdout")


def run_method_subprocess(algo, noisy, output, config=None, python=None, timeout=3600):
    """Invoke a method module via the standard CLI protocol.

    ``python`` selects the interpreter (e.g. the module's own venv);
    defaults to the current one. Returns the parsed stdout JSON.
    """
    module = f"astro_denoise_{algo}"
    cmd = [python or sys.executable, "-m", module, "--input", str(noisy), "--output", str(output)]
    if config:
        cmd += ["--config", str(config)]
    return _run_json_subprocess(cmd, timeout, module)


def run_metrics_subprocess(
    denoised,
    truth,
    output,
    python=None,
    band=None,
    mag_limit=None,
    thresh=1.5,
    radius=2.0,
    thresh_sweep=None,
    thresh_image=None,
    timeout=3600,
):
    """Invoke the astro-denoise-metrics CLI on a denoised image.

    ``truth`` may be a FITS table or a DC2 Parquet catalog (then
    ``band`` selects the flux column). ``thresh_image`` is the image
    whose background RMS scales ``thresh`` (pass the noisy input so the
    detection depth is comparable across methods). Returns the full
    results dict read back from the metrics JSON output file, with
    ``peak_rss_mb`` attached when measurable.
    """
    cmd = [
        python or sys.executable,
        "-m",
        "astro_denoise_metrics",
        "--input",
        str(denoised),
        "--truth",
        str(truth),
        "--output",
        str(output),
        "--thresh",
        str(thresh),
        "--radius",
        str(radius),
    ]
    if band:
        cmd += ["--band", band]
    if mag_limit is not None:
        cmd += ["--mag-limit", str(mag_limit)]
    if thresh_sweep:
        cmd += ["--thresh-sweep", thresh_sweep]
    if thresh_image is not None:
        cmd += ["--thresh-image", str(thresh_image)]
    status = _run_json_subprocess(cmd, timeout, "astro_denoise_metrics")
    results = json.loads(Path(output).read_text())
    if "peak_rss_mb" in status:
        results["peak_rss_mb"] = status["peak_rss_mb"]
    return results


def run_benchmark(algo, data_dir, output_dir=None, config=None, python=None, progress=None):
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
            status = run_method_subprocess(algo, noisy_path, out_path, config=config, python=python)
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
            values = [
                r["metrics"][key]
                for r in ok
                if key in r.get("metrics", {}) and np.isfinite(r["metrics"][key])
            ]
            if values:
                summary[f"mean_{key}"] = round(statistics.fmean(values), 4)
    return summary


# ---------------------------------------------------------------------------
# curated-manifest benchmark (real DC2 regions + metrics engine)
# ---------------------------------------------------------------------------

CURATED_METRIC_KEYS = (
    "completeness",
    "purity",
    "sky_sigma",
    "sharpness_sf",
    "flux_snr",
)


def discover_method_config(manifest_path, algo):
    """Per-method config shipped with a curated block, if any.

    A curated block may carry ``<algo>_config.json`` next to its manifest
    (e.g. ``bm3d_config.json`` with ``variance_hdu`` pointing at the
    VARIANCE HDU the curator writes into every region), so the method
    runs with the block's calibration by default. An explicit
    ``--config`` always takes precedence.
    """
    candidate = Path(manifest_path).parent / f"{algo}_config.json"
    return candidate if candidate.is_file() else None


def _stats_of(records, getter):
    """Mean, sample std and count of the finite values (std is None if n < 2)."""
    values = [v for v in (getter(r) for r in records) if v is not None and np.isfinite(v)]
    if not values:
        return None, None, 0
    mean = round(statistics.fmean(values), 4)
    std = round(statistics.stdev(values), 4) if len(values) > 1 else None
    return mean, std, len(values)


def summarize_curated(runs):
    """Aggregate per-region results overall and per band.

    Each ``mean_<key>`` comes with ``std_<key>`` (sample std across regions,
    None when fewer than two finite values) and ``n_<key>`` so that
    method-vs-method differences can be judged against region-to-region
    dispersion instead of bare means.
    """

    def put(out, key, records, getter):
        out[f"mean_{key}"], out[f"std_{key}"], out[f"n_{key}"] = _stats_of(records, getter)

    def block(records):
        ok = [r for r in records if r.get("status") == "ok"]
        out = {"n_ok": len(ok), "n_failed": len(records) - len(ok)}
        put(out, "wall_time_s", ok, lambda r: r.get("wall_time_s"))
        put(out, "peak_rss_mb", ok, lambda r: r.get("peak_rss_mb"))
        for key in CURATED_METRIC_KEYS:
            put(out, key, ok, lambda r, k=key: r.get("metrics", {}).get(k))
        return out

    summary = block(runs)
    summary["per_band"] = {
        band: block([r for r in runs if r.get("band") == band])
        for band in sorted({r.get("band") for r in runs if r.get("band")})
    }
    return summary


def run_curated_benchmark(
    manifest,
    algo,
    out_dir,
    method_python=None,
    metrics_python=None,
    config=None,
    bands=None,
    tracts=None,
    mag_limit=26.0,
    thresh=1.5,
    previews=False,
    progress=None,
):
    """Benchmark ``algo`` over a curated DC2 region block.

    For every manifest region (optionally filtered by ``bands`` and/or
    ``tracts``): denoise via the method's CLI, evaluate against the
    region's truth catalog via the metrics CLI (projected by WCS, per
    band), and optionally render a before/after/residual PNG preview.
    Returns the consolidated report grouped overall and per band.

    ``tracts`` is how a learned method is scored without data leakage:
    evaluate only on tracts held out from its training split.
    """
    regions = [
        r
        for r in manifest["regions"]
        if (not bands or r["band"] in bands) and (not tracts or r.get("tract") in tracts)
    ]
    if not regions:
        raise FileNotFoundError("manifest has no regions matching the band/tract filter")

    out_dir = Path(out_dir)
    denoised_dir = out_dir / f"denoised_{algo}"
    metrics_dir = out_dir / f"metrics_{algo}"
    denoised_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for index, region in enumerate(regions, start=1):
        if progress:
            progress(index, len(regions), region["id"])
        denoised_path = denoised_dir / f"{region['id']}.fits"
        record = {
            "region": region["id"],
            "band": region["band"],
            "tract": region["tract"],
            "output": str(denoised_path),
        }
        start = time.perf_counter()
        try:
            status = run_method_subprocess(
                algo, region["path"], denoised_path, config=config, python=method_python
            )
            record["wall_time_s"] = round(time.perf_counter() - start, 4)
            record["status"] = status.get("status", "ok")
            if "peak_rss_mb" in status:
                record["peak_rss_mb"] = status["peak_rss_mb"]

            if region.get("truth"):
                # threshold anchored to the noisy input's RMS so every
                # method is scored at the same absolute detection depth
                results = run_metrics_subprocess(
                    denoised_path,
                    region["truth"],
                    metrics_dir / f"{region['id']}.json",
                    python=metrics_python,
                    band=region["band"],
                    mag_limit=mag_limit,
                    thresh=thresh,
                    thresh_image=region["path"],
                )
                record["metrics"] = {
                    key: results.get(key) for key in (*CURATED_METRIC_KEYS, "n_detected", "n_truth")
                }
                record["metrics_json"] = str(metrics_dir / f"{region['id']}.json")

            if previews:
                from .previews import render_preview

                preview_dir = out_dir / f"previews_{algo}"
                preview_dir.mkdir(parents=True, exist_ok=True)
                record["preview"] = render_preview(
                    region["path"],
                    denoised_path,
                    preview_dir / f"{region['id']}.png",
                    title=f"{region['id']} — {algo}",
                )
        except Exception as exc:
            record["wall_time_s"] = round(time.perf_counter() - start, 4)
            record["status"] = "error"
            record["error"] = str(exc)
        runs.append(record)

    return {
        "module": "astro-denoise-bench",
        "kind": "curated-benchmark",
        "algorithm": algo,
        "bands": sorted({r["band"] for r in regions}),
        "tracts": sorted({r["tract"] for r in regions}),
        "mag_limit": mag_limit,
        "thresh": thresh,
        "thresh_reference": "input",
        "config": str(config) if config else None,
        "n_regions": len(regions),
        "runs": runs,
        "summary": summarize_curated(runs),
    }
