"""Paper figures from completed curated-benchmark reports.

Reads finished ``report_<algo>.json`` files (produced by
``run_curated_benchmark``) plus their per-region metrics JSON siblings
and renders the comparison plots used in the write-up. Nothing here
re-runs a denoising method; the threshold-sweep and matching-radius
figures re-invoke only the (cheap) metrics CLI, and cache their raw
results so a second call is free unless ``--refresh-sweep`` is passed.

Figures:

* ``completeness_vs_mag.png`` — completeness vs. truth magnitude, one
  panel per band, mean line per method over its 3 regions. Pure from
  the already-computed ``completeness_vs_mag`` block of each region's
  metrics JSON — no re-run.
* ``completeness_vs_threshold.png`` / ``purity_vs_threshold.png`` — a
  detection-threshold sweep. Re-invokes the metrics CLI with
  ``--thresh-sweep`` per region (cached to ``sweep_<algo>.json``).
* ``completeness_purity_tradeoff.png`` — mean purity vs. mean
  completeness per band with std error bars, one marker per method.
  Pure from each report's ``summary.per_band`` block — no re-run.
* ``matching_radius_sensitivity.png`` (D8) — re-invokes metrics at
  ``--radius`` 1/2/3 on the representative ``r`` band, one line per
  method (cached to ``matching_radius_sensitivity.json``).

All outputs land in a single directory (default ``<curated>/figures/``)
alongside a short ``README.md`` documenting provenance.
"""

from __future__ import annotations

import json
import statistics
import tempfile
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .bench import run_metrics_subprocess

BAND_ORDER = ("u", "g", "r", "i", "z", "y")
DEFAULT_THRESH_SWEEP = "1,1.5,2,3,5,8"
D8_BAND = "r"
D8_RADII = (1.0, 2.0, 3.0)

ALGO_COLORS = {"passthrough": "tab:gray", "bm3d": "tab:blue", "unet": "tab:orange"}


def _color_for(algo, index):
    return ALGO_COLORS.get(algo, f"C{index}")


def _mean_ignore_none(values):
    finite = [v for v in values if v is not None and np.isfinite(v)]
    return float(statistics.fmean(finite)) if finite else None


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def load_report(path):
    return json.loads(Path(path).read_text())


def load_region_metrics(path):
    return json.loads(Path(path).read_text())


def default_report_paths(curated_dir):
    """The three t3829 reports this module is built around."""
    curated_dir = Path(curated_dir)
    return [
        curated_dir / "report_raw.json",
        curated_dir / "report_bm3d_t3829.json",
        curated_dir / "report_unet.json",
    ]


# ---------------------------------------------------------------------------
# generic aggregation
# ---------------------------------------------------------------------------


def _elementwise_stats(series_list):
    """Mean/min/max at each position across several equal-length series.

    ``None`` entries are ignored per position; a position with no finite
    value across every series yields ``None`` for mean/min/max.
    """
    if not series_list:
        return [], [], []
    length = len(series_list[0])
    mean, lo, hi = [], [], []
    for i in range(length):
        values = [s[i] for s in series_list if s[i] is not None and np.isfinite(s[i])]
        if values:
            mean.append(float(statistics.fmean(values)))
            lo.append(float(min(values)))
            hi.append(float(max(values)))
        else:
            mean.append(None)
            lo.append(None)
            hi.append(None)
    return mean, lo, hi


def _drop_none(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys, strict=True) if y is not None]
    if not pairs:
        return [], []
    xs_out, ys_out = zip(*pairs, strict=True)
    return list(xs_out), list(ys_out)


def _drop_none_triplet(xs, los, his):
    pairs = [
        (x, lo, hi)
        for x, lo, hi in zip(xs, los, his, strict=True)
        if lo is not None and hi is not None
    ]
    if not pairs:
        return [], [], []
    xs_out, lo_out, hi_out = zip(*pairs, strict=True)
    return list(xs_out), list(lo_out), list(hi_out)


# ---------------------------------------------------------------------------
# completeness vs magnitude (pure — reads existing per-region JSON only)
# ---------------------------------------------------------------------------


def _common_mag_grid(region_cvms, n_points=12):
    centers = [c for cvm in region_cvms for c in cvm["mag_centers"]]
    if not centers:
        return []
    lo, hi = min(centers), max(centers)
    if lo == hi:
        return [lo]
    return list(np.linspace(lo, hi, n_points))


def _resample_completeness(cvm, grid):
    """Linearly interpolate one region's completeness curve onto ``grid``.

    Null entries are dropped before interpolating; grid points outside
    the region's own magnitude range become ``None`` (no extrapolation).
    """
    if not grid:
        return []
    xs = np.asarray(cvm["mag_centers"], dtype=float)
    ys = np.asarray([np.nan if v is None else v for v in cvm["completeness"]], dtype=float)
    finite = np.isfinite(ys)
    if finite.sum() < 2:
        return [None] * len(grid)
    order = np.argsort(xs[finite])
    xs_f, ys_f = xs[finite][order], ys[finite][order]
    interp = np.interp(grid, xs_f, ys_f, left=np.nan, right=np.nan)
    return [None if not np.isfinite(v) else float(v) for v in interp]


def aggregate_completeness_vs_mag(region_cvms, n_points=12):
    """Mean/min/max completeness-vs-mag curve over several regions.

    Regions report different ``mag_centers`` (bin edges are derived
    per-region from the truth catalog's own magnitude range), so each
    region's curve is first resampled onto a grid spanning the union of
    the regions' magnitude ranges before averaging.
    """
    grid = _common_mag_grid(region_cvms, n_points)
    resampled = [_resample_completeness(cvm, grid) for cvm in region_cvms]
    mean, lo, hi = _elementwise_stats(resampled)
    return {"mag_grid": grid, "mean": mean, "min": lo, "max": hi}


def build_completeness_vs_mag_data(reports, bands=BAND_ORDER):
    """``{band: {algo: aggregate_completeness_vs_mag(...)}}`` for every report."""
    data = {}
    for report in reports:
        algo = report["algorithm"]
        for band in bands:
            region_cvms = []
            for run in report["runs"]:
                if run.get("band") != band or not run.get("metrics_json"):
                    continue
                meta = load_region_metrics(run["metrics_json"])
                cvm = meta.get("completeness_vs_mag")
                if cvm:
                    region_cvms.append(cvm)
            if region_cvms:
                data.setdefault(band, {})[algo] = aggregate_completeness_vs_mag(region_cvms)
    return data


def _band_grid(bands, panel_size=(4.3, 3.5), max_cols=3, **subplot_kwargs):
    """A grid of one subplot per band (up to ``max_cols`` columns), unused axes hidden."""
    n = max(len(bands), 1)
    ncols = min(n, max_cols)
    nrows = -(-n // ncols)  # ceil division
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(panel_size[0] * ncols, panel_size[1] * nrows),
        constrained_layout=True,
        **subplot_kwargs,
    )
    flat = np.atleast_1d(axes).flatten()
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat[:n]


def plot_completeness_vs_mag(data, out_path, bands=BAND_ORDER):
    """One subplot per band (up to 3 columns); mean line + min-max shading per method."""
    fig, axes = _band_grid(bands, sharey=True)
    for ax, band in zip(axes, bands, strict=True):
        for index, (algo, curve) in enumerate(sorted(data.get(band, {}).items())):
            color = _color_for(algo, index)
            xs, ys = _drop_none(curve["mag_grid"], curve["mean"])
            if xs:
                ax.plot(xs, ys, label=algo, color=color, marker="o", markersize=3)
            xs_b, lo_b, hi_b = _drop_none_triplet(curve["mag_grid"], curve["min"], curve["max"])
            if xs_b:
                ax.fill_between(xs_b, lo_b, hi_b, color=color, alpha=0.15)
        ax.set_title(f"band {band}")
        ax.set_xlabel("truth mag (AB)")
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("completeness")
    axes[0].legend(fontsize=8)
    fig.suptitle("Completeness vs. truth magnitude (mean over 3 regions)")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return str(out_path)


# ---------------------------------------------------------------------------
# detection-threshold sweep (re-invokes metrics --thresh-sweep, cached)
# ---------------------------------------------------------------------------


def _region_sweep(run, meta, thresholds, metrics_python):
    """Run the metrics CLI ``--thresh-sweep`` for one region; return its detection_rate block."""
    params = meta["parameters"]
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "sweep.json"
        results = run_metrics_subprocess(
            run["output"],
            meta["truth"],
            out_path,
            python=metrics_python,
            band=run["band"],
            mag_limit=params.get("mag_limit"),
            thresh=params.get("thresh", 1.5),
            radius=params.get("radius", 2.0),
            thresh_sweep=thresholds,
            thresh_image=params.get("thresh_image"),
        )
    return results.get("detection_rate", {})


def run_threshold_sweep(
    report, thresholds=DEFAULT_THRESH_SWEEP, metrics_python=None, progress=None
):
    """Run the detection-threshold sweep for every ok region of ``report``."""
    ok_runs = [r for r in report["runs"] if r.get("status") == "ok" and r.get("metrics_json")]
    regions = {}
    for index, run in enumerate(ok_runs, start=1):
        if progress:
            progress(index, len(ok_runs), f"sweep {report['algorithm']}/{run['region']}")
        meta = load_region_metrics(run["metrics_json"])
        dr = _region_sweep(run, meta, thresholds, metrics_python)
        regions[run["region"]] = {
            "band": run["band"],
            "thresholds": dr.get("thresholds"),
            "completeness": dr.get("completeness"),
            "purity": dr.get("purity"),
            "n_detected": dr.get("n_detected"),
        }
    return {"algorithm": report["algorithm"], "thresholds_arg": thresholds, "regions": regions}


def load_or_run_sweep(
    report,
    cache_path,
    thresholds=DEFAULT_THRESH_SWEEP,
    metrics_python=None,
    refresh=False,
    progress=None,
):
    cache_path = Path(cache_path)
    if cache_path.exists() and not refresh:
        return json.loads(cache_path.read_text())
    sweep = run_threshold_sweep(
        report, thresholds=thresholds, metrics_python=metrics_python, progress=progress
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(sweep, indent=2))
    return sweep


def aggregate_sweep_by_band(sweep, bands=BAND_ORDER):
    """``{band: {"thresholds":..., "mean_completeness":..., ...}}`` from a sweep cache."""
    by_band = {}
    for band in bands:
        entries = [
            e for e in sweep["regions"].values() if e["band"] == band and e.get("thresholds")
        ]
        if not entries:
            continue
        mean_c, lo_c, hi_c = _elementwise_stats([e["completeness"] for e in entries])
        mean_p, lo_p, hi_p = _elementwise_stats([e["purity"] for e in entries])
        by_band[band] = {
            "thresholds": entries[0]["thresholds"],
            "mean_completeness": mean_c,
            "min_completeness": lo_c,
            "max_completeness": hi_c,
            "mean_purity": mean_p,
            "min_purity": lo_p,
            "max_purity": hi_p,
        }
    return by_band


def plot_metric_vs_threshold(data_by_algo_band, metric, out_path, bands=BAND_ORDER, ylabel=None):
    """``data_by_algo_band``: ``{algo: aggregate_sweep_by_band(...)}}``."""
    label = ylabel or metric
    fig, axes = _band_grid(bands, sharey=True)
    for ax, band in zip(axes, bands, strict=True):
        for index, (algo, by_band) in enumerate(sorted(data_by_algo_band.items())):
            block = by_band.get(band)
            if not block:
                continue
            color = _color_for(algo, index)
            xs, ys = _drop_none(block["thresholds"], block[f"mean_{metric}"])
            if xs:
                ax.plot(xs, ys, label=algo, color=color, marker="o", markersize=3)
            xs_b, lo_b, hi_b = _drop_none_triplet(
                block["thresholds"], block[f"min_{metric}"], block[f"max_{metric}"]
            )
            if xs_b:
                ax.fill_between(xs_b, lo_b, hi_b, color=color, alpha=0.15)
        ax.set_title(f"band {band}")
        ax.set_xlabel("threshold (x background RMS)")
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel(label)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{label.capitalize()} vs. detection threshold (mean over 3 regions)")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return str(out_path)


# ---------------------------------------------------------------------------
# completeness/purity trade-off (pure — reads report summaries only)
# ---------------------------------------------------------------------------


def build_tradeoff_data(reports, bands=BAND_ORDER):
    data = {}
    for report in reports:
        algo = report["algorithm"]
        per_band = report.get("summary", {}).get("per_band", {})
        for band in bands:
            block = per_band.get(band)
            if (
                not block
                or block.get("mean_completeness") is None
                or block.get("mean_purity") is None
            ):
                continue
            data.setdefault(band, {})[algo] = {
                "mean_completeness": block["mean_completeness"],
                "std_completeness": block.get("std_completeness") or 0.0,
                "mean_purity": block["mean_purity"],
                "std_purity": block.get("std_purity") or 0.0,
            }
    return data


def plot_tradeoff(data, out_path, bands=BAND_ORDER):
    fig, axes = _band_grid(bands, sharex=True, sharey=True)
    for ax, band in zip(axes, bands, strict=True):
        for index, (algo, point) in enumerate(sorted(data.get(band, {}).items())):
            color = _color_for(algo, index)
            ax.errorbar(
                point["mean_completeness"],
                point["mean_purity"],
                xerr=point["std_completeness"],
                yerr=point["std_purity"],
                fmt="o",
                color=color,
                label=algo,
                capsize=3,
            )
        ax.set_title(f"band {band}")
        ax.set_xlabel("completeness")
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("purity")
    axes[0].legend(fontsize=8)
    fig.suptitle("Completeness-purity trade-off per band (mean +/- std over 3 regions)")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return str(out_path)


# ---------------------------------------------------------------------------
# D8 — matching-radius sensitivity (re-invokes metrics at --radius, cached)
# ---------------------------------------------------------------------------


def run_radius_sensitivity(
    reports, band=D8_BAND, radii=D8_RADII, metrics_python=None, progress=None
):
    """Re-run metrics at each radius on ``band``'s regions for every report."""
    runs_by_algo = {
        report["algorithm"]: [
            r
            for r in report["runs"]
            if r.get("band") == band and r.get("status") == "ok" and r.get("metrics_json")
        ]
        for report in reports
    }
    total = sum(len(radii) * len(runs) for runs in runs_by_algo.values())
    done = 0
    algorithms = {}
    for algo, runs in runs_by_algo.items():
        per_radius = {"completeness": [], "purity": []}
        for radius in radii:
            region_completeness, region_purity = [], []
            for run in runs:
                done += 1
                if progress:
                    progress(done, total, f"radius {radius} {algo}/{run['region']}")
                meta = load_region_metrics(run["metrics_json"])
                params = meta["parameters"]
                with tempfile.TemporaryDirectory() as tmp:
                    out_path = Path(tmp) / "radius.json"
                    res = run_metrics_subprocess(
                        run["output"],
                        meta["truth"],
                        out_path,
                        python=metrics_python,
                        band=band,
                        mag_limit=params.get("mag_limit"),
                        thresh=params.get("thresh", 1.5),
                        radius=radius,
                        thresh_image=params.get("thresh_image"),
                    )
                region_completeness.append(res.get("completeness"))
                region_purity.append(res.get("purity"))
            per_radius["completeness"].append(_mean_ignore_none(region_completeness))
            per_radius["purity"].append(_mean_ignore_none(region_purity))
        algorithms[algo] = per_radius
    return {"band": band, "radii": list(radii), "algorithms": algorithms}


def load_or_run_radius_sensitivity(
    reports,
    cache_path,
    band=D8_BAND,
    radii=D8_RADII,
    metrics_python=None,
    refresh=False,
    progress=None,
):
    cache_path = Path(cache_path)
    if cache_path.exists() and not refresh:
        return json.loads(cache_path.read_text())
    data = run_radius_sensitivity(
        reports, band=band, radii=radii, metrics_python=metrics_python, progress=progress
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data, indent=2))
    return data


def plot_radius_sensitivity(data, out_path):
    fig, (ax_c, ax_p) = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    radii = data["radii"]
    for index, (algo, curves) in enumerate(sorted(data["algorithms"].items())):
        color = _color_for(algo, index)
        xs, ys = _drop_none(radii, curves["completeness"])
        if xs:
            ax_c.plot(xs, ys, marker="o", label=algo, color=color)
        xs, ys = _drop_none(radii, curves["purity"])
        if xs:
            ax_p.plot(xs, ys, marker="o", label=algo, color=color)
    ax_c.set_title("completeness")
    ax_p.set_title("purity")
    for ax in (ax_c, ax_p):
        ax.set_xlabel("matching radius (px)")
        ax.set_ylim(0, 1.05)
    ax_c.set_ylabel("value")
    ax_c.legend(fontsize=8)
    fig.suptitle(f"Matching-radius sensitivity — band {data['band']} (D8)")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return str(out_path)


def radius_sensitivity_summary(data):
    """One-line summary of how much completeness/purity move across radii."""
    radii = data["radii"]
    if len(radii) < 2:
        return "matching-radius sensitivity: not enough radii to compare"
    deltas = []
    for algo, curves in data["algorithms"].items():
        completeness = curves["completeness"]
        purity = curves["purity"]
        if (
            completeness[0] is None
            or completeness[-1] is None
            or purity[0] is None
            or purity[-1] is None
        ):
            continue
        deltas.append((algo, completeness[-1] - completeness[0], purity[-1] - purity[0]))
    if not deltas:
        return "matching-radius sensitivity: no data"
    most_sensitive = max(deltas, key=lambda item: abs(item[1]) + abs(item[2]))[0]
    parts = "; ".join(
        f"{algo} dCompleteness={dc:+.3f} dPurity={dp:+.3f}" for algo, dc, dp in deltas
    )
    return (
        f"matching-radius sensitivity (band {data['band']}, r={radii[0]}->{radii[-1]}): "
        f"{parts}; most sensitive: {most_sensitive}"
    )


# ---------------------------------------------------------------------------
# README + top-level orchestration
# ---------------------------------------------------------------------------

README_TEMPLATE = """# Figures

Generated by `astro-denoise-bench --figures` from the completed t3829
curated-benchmark reports:

- `report_raw.json` — passthrough (no denoising) baseline
- `report_bm3d_t3829.json` — BM3D
- `report_unet.json` — U-Net

## completeness_vs_mag.png

Completeness vs. truth AB magnitude, one panel per band (u, g, r, i, z,
y). Per method, the line is the mean over its 3 regions of the
region's `completeness_vs_mag` block (read from
`metrics_<algo>/<region>.json`, no re-run); the shaded band is the
region-to-region min-max. Regions report different magnitude bin
edges (derived from each region's own truth range), so curves are
resampled onto a shared magnitude grid before averaging.

## completeness_vs_threshold.png / purity_vs_threshold.png

Detection-threshold sweep at thresholds `{thresholds}` (background-RMS
units), anchored to each region's own noisy-input RMS
(`--thresh-image`). Built by re-invoking
`astro-denoise-metrics --thresh-sweep` per region (cached to
`sweep_<algo>.json`; pass `--refresh-sweep` to force a re-run). Per
band, per method: mean over the method's 3 regions, shaded band is the
region-to-region min-max.

## completeness_purity_tradeoff.png

Mean purity vs. mean completeness per band, with std error bars across
the 3 regions, one marker per method. Pure from each report's
`summary.per_band` block — no re-run.

## matching_radius_sensitivity.png (D8)

Matching-radius sensitivity check on the representative `{band}` band:
completeness and purity at `--radius` {radii}, mean over the band's 3
regions, one line per method. Cached to
`matching_radius_sensitivity.json` (pass `--refresh-sweep` to force a
re-run).

{summary_line}
"""


def write_readme(out_dir, thresholds, band, radii, summary_line):
    text = README_TEMPLATE.format(
        thresholds=thresholds,
        band=band,
        radii=", ".join(str(r) for r in radii),
        summary_line=summary_line,
    )
    path = Path(out_dir) / "README.md"
    path.write_text(text)
    return str(path)


def generate_all_figures(
    report_paths,
    out_dir,
    metrics_python=None,
    thresholds=DEFAULT_THRESH_SWEEP,
    refresh_sweep=False,
    skip_sweep=False,
    skip_d8=False,
    d8_band=D8_BAND,
    d8_radii=D8_RADII,
    progress=None,
):
    """Render every paper figure into ``out_dir``; returns produced paths + D8 summary."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports = [load_report(p) for p in report_paths]

    produced = {
        "completeness_vs_mag": plot_completeness_vs_mag(
            build_completeness_vs_mag_data(reports), out_dir / "completeness_vs_mag.png"
        )
    }

    if not skip_sweep:
        agg_by_algo = {}
        for report in reports:
            algo = report["algorithm"]
            sweep = load_or_run_sweep(
                report,
                out_dir / f"sweep_{algo}.json",
                thresholds=thresholds,
                metrics_python=metrics_python,
                refresh=refresh_sweep,
                progress=progress,
            )
            agg_by_algo[algo] = aggregate_sweep_by_band(sweep)
        produced["completeness_vs_threshold"] = plot_metric_vs_threshold(
            agg_by_algo,
            "completeness",
            out_dir / "completeness_vs_threshold.png",
            ylabel="completeness",
        )
        produced["purity_vs_threshold"] = plot_metric_vs_threshold(
            agg_by_algo, "purity", out_dir / "purity_vs_threshold.png", ylabel="purity"
        )

    produced["completeness_purity_tradeoff"] = plot_tradeoff(
        build_tradeoff_data(reports), out_dir / "completeness_purity_tradeoff.png"
    )

    d8_summary = "matching-radius sensitivity: skipped (--skip-d8)"
    if not skip_d8:
        d8_data = load_or_run_radius_sensitivity(
            reports,
            out_dir / "matching_radius_sensitivity.json",
            band=d8_band,
            radii=d8_radii,
            metrics_python=metrics_python,
            refresh=refresh_sweep,
            progress=progress,
        )
        produced["matching_radius_sensitivity"] = plot_radius_sensitivity(
            d8_data, out_dir / "matching_radius_sensitivity.png"
        )
        d8_summary = radius_sensitivity_summary(d8_data)

    produced["readme"] = write_readme(out_dir, thresholds, d8_band, d8_radii, d8_summary)
    return {"produced": produced, "d8_summary": d8_summary}
