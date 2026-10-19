# astro-denoise

Benchmarking of noise-reduction methods on simulated DC2 images of the Vera Rubin Observatory.

This repository is the core of the project: the benchmark harness, the science metrics and the no-processing control method. The BM3D and U-Net methods live in their own repositories and plug in through the `astro_denoise.methods` entry point.

Status: 0.x, under active development. The three packages below are being merged into a single `astro-denoise` package.

| Package | Path |
|---|---|
| `astro-denoise-bench` | `packages/bench` |
| `astro-denoise-metrics` | `packages/metrics` |
| `astro-denoise-passthrough` | `packages/passthrough` |

## Documents

`paper/` holds the thesis documents as [TeXForge](https://github.com/UniverLab/texforge) projects: `propuesta` (the research proposal, APA format for Universidad de La Salle) and `marco-etico` (the ethical framework). Build one with `texforge build` inside its folder.

The DC2 data is not in this repository; it stays local.
