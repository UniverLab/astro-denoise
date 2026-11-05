"""Interactive dashboard of the astro-denoise platform.

Textual TUI with three views:

* **Methods** — algorithm modules coupled to the platform: installed
  ``astro_denoise.methods`` entry points plus sibling ``astro-denoise-*``
  repos detected on disk (with their venv status), so new methods are
  discoverable before they are even installed.
* **Dataset** — the curated DC2 region block (seed, bands, tracts and
  one row per region) read from ``manifest.json``.
* **Results** — comparison of benchmark reports (``report*.json``)
  found next to the manifest: one row per algorithm x band; selecting
  a row drills down into the per-region metrics of that algorithm.

Data assembly lives in pure helpers so it stays unit-testable without
driving the TUI event loop.
"""

from __future__ import annotations

import json
from pathlib import Path

from .bench import CURATED_METRIC_KEYS, discover_methods

NON_METHOD_MODULES = {"bench", "metrics"}


def scan_method_repos(code_root=None):
    """Detect sibling astro-denoise-* method repos and their venv state."""
    if code_root is None:
        code_root = Path(__file__).resolve().parents[3]
    code_root = Path(code_root)
    repos = []
    for repo in sorted(code_root.glob("astro-denoise-*")):
        if not repo.is_dir():
            continue
        name = repo.name.removeprefix("astro-denoise-")
        if name in NON_METHOD_MODULES:
            continue
        package = repo / "src" / f"astro_denoise_{name}"
        repos.append(
            {
                "name": name,
                "path": str(repo),
                "has_package": package.is_dir(),
                "has_venv": (repo / ".venv").is_dir(),
            }
        )
    return repos


def method_rows(code_root=None):
    """Rows for the Methods table: entry points + on-disk repos."""
    installed = discover_methods()
    rows = []
    seen = set()
    for name, ep in sorted(installed.items()):
        rows.append((name, "entry point", ep.value, "yes"))
        seen.add(name)
    for repo in scan_method_repos(code_root):
        if repo["name"] in seen:
            continue
        state = "venv ready" if repo["has_venv"] else "no venv"
        rows.append((repo["name"], "on disk", repo["path"], state))
    return rows


def dataset_summary(manifest):
    """One-line description of the curated block for the Dataset view."""
    per_band = len(manifest["regions"]) // max(len(manifest["bands"]), 1)
    seed = manifest.get("seed", "n/a")
    return (
        f"{manifest['n_regions']} regions | "
        f"bands: {', '.join(manifest['bands'])} (~{per_band}/band) | "
        f"tracts: {', '.join(str(t) for t in manifest['tracts'])} | "
        f"size: {manifest['size']}px | seed: {seed}"
    )


def dataset_rows(manifest):
    """Rows for the Dataset table from a loaded manifest."""
    return [
        (
            region["id"],
            region["band"],
            str(region["tract"]),
            f"{region['center_x']:.0f},{region['center_y']:.0f}",
            f"{region['size']}px",
            "yes" if region.get("truth") else "no",
        )
        for region in manifest["regions"]
    ]


def find_reports(manifest_path):
    """Benchmark reports living next to the manifest."""
    base = Path(manifest_path).parent
    return sorted(base.glob("report*.json"))


def _load_curated_report(path):
    try:
        report = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if report.get("kind") != "curated-benchmark":
        return None
    return report


def comparison_rows(report_paths):
    """One row per algorithm x band with the curated metric means."""
    rows = []
    for path in report_paths:
        report = _load_curated_report(path)
        if report is None:
            continue
        algo = report.get("algorithm", "?")
        summary = report.get("summary", {})
        blocks = {"all": summary, **summary.get("per_band", {})}
        for band, block in blocks.items():
            rows.append(
                (
                    algo,
                    band,
                    str(block.get("n_ok")),
                    *[str(block.get(f"mean_{key}")) for key in CURATED_METRIC_KEYS],
                    str(block.get("mean_wall_time_s")),
                )
            )
    return rows


def report_for_algo(report_paths, algo):
    """Path of the first curated report produced by ``algo``."""
    for path in report_paths:
        report = _load_curated_report(path)
        if report and report.get("algorithm") == algo:
            return path
    return None


def report_region_rows(report_path):
    """Per-region drill-down rows of one curated report."""
    report = _load_curated_report(report_path)
    if report is None:
        return []
    rows = []
    for run in report.get("runs", []):
        metrics = run.get("metrics", {})
        rows.append(
            (
                run.get("region", "?"),
                run.get("band", "?"),
                run.get("status", "?"),
                *[str(metrics.get(key)) for key in CURATED_METRIC_KEYS],
                str(run.get("wall_time_s")),
                str(run.get("peak_rss_mb", "-")),
            )
        )
    return rows


def run_tui(manifest_path=None):
    """Launch the dashboard; returns a process exit code."""
    try:
        from textual.app import App
        from textual.widgets import DataTable, Footer, Header, Static, TabbedContent, TabPane
    except ImportError:
        print("textual is required for --tui (uv sync --all-extras)")
        return 1

    from .dc2 import load_manifest

    manifest = None
    if manifest_path and Path(manifest_path).exists():
        manifest = load_manifest(manifest_path)

    metric_columns = [f"mean_{key}" for key in CURATED_METRIC_KEYS]

    class BenchApp(App):
        TITLE = "astro-denoise-bench"
        BINDINGS = [("q", "quit", "Quit"), ("r", "refresh", "Refresh")]
        CSS = """
        DataTable { height: 1fr; }
        #detail-title { padding: 1 0 0 0; color: $text-muted; }
        #dataset-summary { padding: 0 0 1 0; color: $accent; }
        """

        def compose(self):
            yield Header()
            with TabbedContent(initial="methods"):
                with TabPane("Methods", id="methods"):
                    yield DataTable(id="methods-table", cursor_type="row", zebra_stripes=True)
                with TabPane("Dataset", id="dataset"):
                    if manifest:
                        yield Static(dataset_summary(manifest), id="dataset-summary")
                        yield DataTable(id="dataset-table", cursor_type="row", zebra_stripes=True)
                    else:
                        yield Static(
                            "no manifest loaded — pass --manifest data/curated/manifest.json"
                        )
                with TabPane("Results", id="results"):
                    yield DataTable(id="results-table", cursor_type="row", zebra_stripes=True)
                    yield Static("select a row to drill down into its regions", id="detail-title")
                    yield DataTable(id="detail-table", cursor_type="row", zebra_stripes=True)
            yield Footer()

        def on_mount(self):
            self.populate()

        def action_refresh(self):
            self.populate()

        def populate(self):
            methods = self.query_one("#methods-table", DataTable)
            methods.clear(columns=True)
            methods.add_columns("method", "source", "location", "state")
            methods.add_rows(method_rows())

            if manifest:
                dataset = self.query_one("#dataset-table", DataTable)
                dataset.clear(columns=True)
                dataset.add_columns("region", "band", "tract", "center", "size", "truth")
                dataset.add_rows(dataset_rows(manifest))

            results = self.query_one("#results-table", DataTable)
            results.clear(columns=True)
            results.add_columns("algo", "band", "n_ok", *metric_columns, "wall_s")
            if manifest_path:
                results.add_rows(comparison_rows(find_reports(manifest_path)))

            detail = self.query_one("#detail-table", DataTable)
            detail.clear(columns=True)
            detail.add_columns("region", "band", "status", *metric_columns, "wall_s", "rss_mb")

        def on_data_table_row_selected(self, event):
            if event.data_table.id != "results-table" or not manifest_path:
                return
            algo = str(event.data_table.get_row(event.row_key)[0])
            report = report_for_algo(find_reports(manifest_path), algo)
            title = self.query_one("#detail-title", Static)
            detail = self.query_one("#detail-table", DataTable)
            detail.clear(columns=True)
            detail.add_columns("region", "band", "status", *metric_columns, "wall_s", "rss_mb")
            if report is None:
                title.update(f"no curated report found for {algo!r}")
                return
            title.update(f"regions of {algo} — {Path(report).name}")
            detail.add_rows(report_region_rows(report))

    BenchApp().run()
    return 0
