# astro-denoise

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23989629.svg)](https://doi.org/10.5281/zenodo.23989629)

Benchmarking of noise-reduction methods on simulated DC2 images of the Vera Rubin Observatory.

This repository is the core of the project: the benchmark harness, the science metrics and the no-processing control method. The BM3D and U-Net methods live in their own repositories and plug in through the `astro_denoise.methods` entry point.

Status: 0.x, under active development. One distribution, `astro-denoise`; install the `bench` and `metrics` extras for the harness and the metrics. Package notes are under `docs/`.

## Documents

`paper/` holds the thesis documents as [TeXForge](https://github.com/UniverLab/texforge) projects: `propuesta` (the research proposal, APA format for Universidad de La Salle) and `marco-etico` (the ethical framework). Build one with `texforge build` inside its folder.

The DC2 data is not in this repository; it stays local.
