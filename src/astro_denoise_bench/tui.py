"""Interactive dashboard of the astro-denoise platform.

Textual TUI with three views:

* **Methods** — algorithm modules coupled to the platform: installed
  ``astro_denoise.methods`` entry points plus sibling ``astro-denoise-*``
  repos detected on disk (with their venv status), so new methods are
  discoverable before they are even installed.
* **Dataset** — the curated DC2 region block (band, tract, size, truth
  availability per region) read from ``manifest.json``.
* **Results** — comparison of benchmark reports (``report*.json``)
  found next to the manifest: one row per algorithm x band with the
  scientific metrics side by side.

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


def dataset_rows(manifest):
    """Rows for the Dataset table from a loaded manifest."""
    return [
        (
            region["id"],
            region["band"],
            str(region["tract"]),
            f"{region['size']}px",
            "yes" if region.get("truth") else "no",
        )
        for region in manifest["regions"]
    ]


def find_reports(manifest_path):
    """Benchmark reports living next to the manifest."""
    base = Path(manifest_path).parent
    return sorted(base.glob("report*.json"))


def comparison_rows(report_paths):
    """One row per algorithm x band with the curated metric means."""
    rows = []
    for path in report_paths:
        try:
            report = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if report.get("kind") != "curated-benchmark":
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

    class BenchApp(App):
        TITLE = "astro-denoise-bench"
        BINDINGS = [("q", "quit", "Quit"), ("r", "refresh", "Refresh")]

        def compose(self):
            yield Header()
            with TabbedContent(initial="methods"):
                with TabPane("Methods", id="methods"):
                    yield DataTable(id="methods-table")
                with TabPane("Dataset", id="dataset"):
                    if manifest:
                        yield Static(
                            f"curated block: {manifest['n_regions']} regions | "
                            f"bands: {', '.join(manifest['bands'])} | "
                            f"tracts: {manifest['tracts']}"
                        )
                        yield DataTable(id="dataset-table")
                    else:
                        yield Static(
                            "no manifest loaded — pass --manifest data/curated/manifest.json"
                        )
                with TabPane("Results", id="results"):
                    yield DataTable(id="results-table")
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
                dataset.add_columns("region", "band", "tract", "size", "truth")
                dataset.add_rows(dataset_rows(manifest))

            results = self.query_one("#results-table", DataTable)
            results.clear(columns=True)
            results.add_columns(
                "algo",
                "band",
                "n_ok",
                *[f"mean_{key}" for key in CURATED_METRIC_KEYS],
                "wall_s",
            )
            if manifest_path:
                results.add_rows(comparison_rows(find_reports(manifest_path)))

    BenchApp().run()
    return 0
