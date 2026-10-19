# astro-denoise

Benchmarking of noise-reduction methods on simulated DC2 images of the Vera Rubin Observatory.

This repository is the core of the project: the benchmark harness, the science metrics and the no-processing control method. The BM3D and U-Net methods live in their own repositories and plug in through the `astro_denoise.methods` entry point.

Status: 0.x, under active development. The three packages below are being merged into a single `astro-denoise` package.

| Package | Path |
|---|---|
| `astro-denoise-bench` | `packages/bench` |
| `astro-denoise-metrics` | `packages/metrics` |
| `astro-denoise-passthrough` | `packages/passthrough` |
