# Changelog

## [0.4.0] - 2026-11-05

### Changed

- The three packages are now one distribution, `astro-denoise`, with the extras `bench` and `metrics`.
- The harness and the metrics are imported as `astro_denoise.bench` and `astro_denoise.metrics`; `python -m astro_denoise_bench` and `python -m astro_denoise_metrics` become `python -m astro_denoise.bench` and `python -m astro_denoise.metrics`. The console scripts and the method plugin protocol do not change.
- Internal functions take option objects (`MetricsOptions`, `CuratedOptions`, `FigureOptions`); the command line and the report do not change.

### Security

- Dependencies updated to clear known vulnerabilities (pillow, anyio, certifi, filelock); science dependencies stay pinned.

### Fixed

- The lab's Python CI now runs on this repository.
