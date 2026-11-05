"""Unit tests for the paper-figures module.

No real metrics subprocess is ever invoked here: aggregation helpers
are exercised with tiny synthetic data, and the sweep/tradeoff/D8
plotting helpers are tested by injecting synthetic sweep dicts
directly. Only ``generate_all_figures`` with ``skip_sweep`` and
``skip_d8`` set is run end-to-end, against synthetic report + metrics
JSON fixtures written under ``tmp_path``.
"""

import json

import pytest

from astro_denoise_bench import figures as fig


def _png_written(path):
    return path.is_file() and path.stat().st_size > 0


class TestElementwiseStats:
    def test_ignores_none_per_position(self):
        mean, lo, hi = fig._elementwise_stats([[0.5, 0.6], [0.7, None]])
        assert mean == pytest.approx([0.6, 0.6])
        assert lo == pytest.approx([0.5, 0.6])
        assert hi == pytest.approx([0.7, 0.6])

    def test_all_none_position_yields_none(self):
        mean, lo, hi = fig._elementwise_stats([[None, 1.0], [None, 2.0]])
        assert mean == [None, 1.5]
        assert lo == [None, 1.0]
        assert hi == [None, 2.0]

    def test_empty_series_list(self):
        assert fig._elementwise_stats([]) == ([], [], [])


class TestCompletenessVsMagAggregation:
    def test_averages_over_regions_ignoring_null(self):
        region_a = {"mag_centers": [20.0, 21.0, 22.0], "completeness": [1.0, 0.8, None]}
        region_b = {"mag_centers": [20.0, 21.0, 22.0], "completeness": [1.0, 0.9, 0.5]}
        result = fig.aggregate_completeness_vs_mag([region_a, region_b], n_points=3)
        assert result["mag_grid"] == pytest.approx([20.0, 21.0, 22.0])
        # first two points: both regions have a value -> plain mean
        assert result["mean"][0] == pytest.approx(1.0)
        assert result["mean"][1] == pytest.approx(0.85)
        # last point: region_a is null -> only region_b's value survives
        assert result["mean"][2] == pytest.approx(0.5)
        assert result["min"][2] == pytest.approx(0.5)
        assert result["max"][2] == pytest.approx(0.5)

    def test_resamples_regions_with_different_bin_edges(self):
        region_a = {"mag_centers": [20.0, 22.0], "completeness": [1.0, 0.0]}
        region_b = {"mag_centers": [20.0, 21.0, 22.0], "completeness": [1.0, 0.5, 0.0]}
        result = fig.aggregate_completeness_vs_mag([region_a, region_b], n_points=3)
        # grid spans the union of both ranges (20-22); midpoint ~21
        assert result["mag_grid"][0] == pytest.approx(20.0)
        assert result["mag_grid"][-1] == pytest.approx(22.0)
        assert result["mean"][0] == pytest.approx(1.0)
        assert result["mean"][-1] == pytest.approx(0.0)

    def test_single_region_no_crash(self):
        region = {"mag_centers": [20.0, 21.0], "completeness": [1.0, 0.5]}
        result = fig.aggregate_completeness_vs_mag([region], n_points=2)
        assert result["mean"] == pytest.approx([1.0, 0.5])

    def test_empty_input(self):
        result = fig.aggregate_completeness_vs_mag([])
        assert result == {"mag_grid": [], "mean": [], "min": [], "max": []}


class TestAggregateSweepByBand:
    def test_averages_thresholds_across_regions(self):
        sweep = {
            "algorithm": "bm3d",
            "regions": {
                "r0": {
                    "band": "r",
                    "thresholds": [1, 2],
                    "completeness": [0.5, 0.6],
                    "purity": [0.9, None],
                },
                "r1": {
                    "band": "r",
                    "thresholds": [1, 2],
                    "completeness": [0.7, None],
                    "purity": [0.8, 0.85],
                },
                "g0": {
                    "band": "g",
                    "thresholds": [1, 2],
                    "completeness": [0.2, 0.3],
                    "purity": [0.4, 0.5],
                },
            },
        }
        by_band = fig.aggregate_sweep_by_band(sweep, bands=("g", "r"))
        assert set(by_band) == {"g", "r"}
        r_block = by_band["r"]
        assert r_block["thresholds"] == [1, 2]
        assert r_block["mean_completeness"] == pytest.approx([0.6, 0.6])
        assert r_block["mean_purity"] == pytest.approx([0.85, 0.85])

    def test_band_with_no_entries_is_absent(self):
        sweep = {"algorithm": "x", "regions": {}}
        assert fig.aggregate_sweep_by_band(sweep, bands=("r",)) == {}


class TestRadiusSensitivitySummary:
    def test_reports_most_sensitive_method(self):
        data = {
            "band": "r",
            "radii": [1.0, 2.0, 3.0],
            "algorithms": {
                "bm3d": {"completeness": [0.3, 0.32, 0.34], "purity": [0.95, 0.94, 0.93]},
                "unet": {"completeness": [0.2, 0.4, 0.6], "purity": [0.9, 0.7, 0.5]},
            },
        }
        line = fig.radius_sensitivity_summary(data)
        assert "band r" in line
        assert "r=1.0->3.0" in line
        assert "most sensitive: unet" in line

    def test_too_few_radii(self):
        data = {"band": "r", "radii": [1.0], "algorithms": {}}
        assert "not enough radii" in fig.radius_sensitivity_summary(data)

    def test_no_finite_data(self):
        data = {
            "band": "r",
            "radii": [1.0, 2.0],
            "algorithms": {"bm3d": {"completeness": [None, None], "purity": [None, None]}},
        }
        assert "no data" in fig.radius_sensitivity_summary(data)


class TestPlottingHelpersWritePngs:
    def test_plot_completeness_vs_mag(self, tmp_path):
        data = {
            "r": {
                "bm3d": {
                    "mag_grid": [20.0, 21.0],
                    "mean": [0.9, 0.5],
                    "min": [0.8, 0.4],
                    "max": [1.0, 0.6],
                },
                "unet": {
                    "mag_grid": [20.0, 21.0],
                    "mean": [0.8, 0.4],
                    "min": [0.7, 0.3],
                    "max": [0.9, 0.5],
                },
            }
        }
        out = tmp_path / "completeness_vs_mag.png"
        fig.plot_completeness_vs_mag(data, out, bands=("r",))
        assert _png_written(out)

    def test_plot_metric_vs_threshold(self, tmp_path):
        data = {
            "bm3d": {
                "r": {
                    "thresholds": [1, 2, 3],
                    "mean_completeness": [0.9, 0.7, 0.4],
                    "min_completeness": [0.8, 0.6, 0.3],
                    "max_completeness": [1.0, 0.8, 0.5],
                    "mean_purity": [0.6, 0.8, 0.95],
                    "min_purity": [0.5, 0.7, 0.9],
                    "max_purity": [0.7, 0.9, 1.0],
                }
            }
        }
        out_c = tmp_path / "completeness_vs_threshold.png"
        out_p = tmp_path / "purity_vs_threshold.png"
        fig.plot_metric_vs_threshold(
            data, "completeness", out_c, bands=("r",), ylabel="completeness"
        )
        fig.plot_metric_vs_threshold(data, "purity", out_p, bands=("r",), ylabel="purity")
        assert _png_written(out_c)
        assert _png_written(out_p)

    def test_plot_tradeoff(self, tmp_path):
        data = {
            "r": {
                "bm3d": {
                    "mean_completeness": 0.5,
                    "std_completeness": 0.05,
                    "mean_purity": 0.9,
                    "std_purity": 0.02,
                },
                "unet": {
                    "mean_completeness": 0.6,
                    "std_completeness": 0.03,
                    "mean_purity": 0.85,
                    "std_purity": 0.04,
                },
            }
        }
        out = tmp_path / "tradeoff.png"
        fig.plot_tradeoff(data, out, bands=("r",))
        assert _png_written(out)

    def test_plot_radius_sensitivity(self, tmp_path):
        data = {
            "band": "r",
            "radii": [1.0, 2.0, 3.0],
            "algorithms": {
                "bm3d": {"completeness": [0.3, 0.32, 0.34], "purity": [0.95, 0.94, 0.93]},
                "unet": {"completeness": [0.2, 0.4, 0.6], "purity": [0.9, 0.7, 0.5]},
            },
        }
        out = tmp_path / "radius.png"
        fig.plot_radius_sensitivity(data, out)
        assert _png_written(out)


def _write_region_metrics(path, truth, band, mag_limit, mag_centers, completeness):
    path.write_text(
        json.dumps(
            {
                "module": "astro-denoise-metrics",
                "input": "denoised.fits",
                "truth": str(truth),
                "parameters": {
                    "radius": 2.0,
                    "thresh": 1.5,
                    "minarea": 5,
                    "band": band,
                    "mag_limit": mag_limit,
                    "thresh_image": "region.fits",
                    "thresh_rms": 0.01,
                },
                "completeness_vs_mag": {
                    "mag_bins": mag_centers,
                    "mag_centers": mag_centers,
                    "completeness": completeness,
                    "n_truth": [10] * len(mag_centers),
                    "n_recovered": [5] * len(mag_centers),
                },
            }
        )
    )


def _make_report(tmp_path, algo, band, mean_completeness, mean_purity):
    metrics_dir = tmp_path / f"metrics_{algo}"
    metrics_dir.mkdir(exist_ok=True)
    truth = tmp_path / "truth.parquet"
    truth.touch()
    runs = []
    for i in range(3):
        metrics_json = metrics_dir / f"region{i}.json"
        _write_region_metrics(
            metrics_json,
            truth,
            band,
            26.0,
            [20.0, 21.0, 22.0],
            [1.0, 0.8 + 0.01 * i, 0.3],
        )
        runs.append(
            {
                "region": f"region{i}",
                "band": band,
                "status": "ok",
                "output": str(tmp_path / f"denoised_{algo}" / f"region{i}.fits"),
                "metrics_json": str(metrics_json),
                "metrics": {"completeness": mean_completeness, "purity": mean_purity},
            }
        )
    return {
        "algorithm": algo,
        "bands": [band],
        "runs": runs,
        "summary": {
            "per_band": {
                band: {
                    "mean_completeness": mean_completeness,
                    "std_completeness": 0.02,
                    "mean_purity": mean_purity,
                    "std_purity": 0.01,
                }
            }
        },
    }


class TestGenerateAllFiguresPureSubset:
    def test_produces_pure_figures_without_subprocess(self, tmp_path):
        report_bm3d = _make_report(tmp_path, "bm3d", "r", 0.5, 0.9)
        report_unet = _make_report(tmp_path, "unet", "r", 0.6, 0.85)
        report_paths = []
        for report in (report_bm3d, report_unet):
            path = tmp_path / f"report_{report['algorithm']}.json"
            path.write_text(json.dumps(report))
            report_paths.append(path)

        out_dir = tmp_path / "figures"
        result = fig.generate_all_figures(report_paths, out_dir, skip_sweep=True, skip_d8=True)

        assert _png_written(out_dir / "completeness_vs_mag.png")
        assert _png_written(out_dir / "completeness_purity_tradeoff.png")
        assert (out_dir / "README.md").is_file()
        assert not (out_dir / "sweep_bm3d.json").exists()
        assert not (out_dir / "matching_radius_sensitivity.json").exists()
        assert "skipped" in result["d8_summary"]
        assert "completeness_vs_threshold" not in result["produced"]
