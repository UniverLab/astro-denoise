# Top-50 Reference Selection — Rationale & Methodology (v2)

## Overview
This document explains the methodology used to select and curate the final 50 references from a pool of **304 candidate PDFs** in `asreview/pdfs/`.

The previous automated selection (`top50_package/`) used TF-IDF + Logistic Regression (ROC-AUC 0.762) trained on 40 labeled records and produced a **flawed list** with off-topic papers (gravitational waves, solar imaging, CMB, video denoising, medical imaging). This v2 selection was performed using Canopy RAG semantic search + manual validation.

---

## Methodology

### Step 1: RAG-based Relevance Scoring (5 queries)
1. `"astronomical image denoising deep learning LSST survey PSF benchmark"`
2. `"PSNR SSIM FID metric evaluation denoising restoration benchmark"`
3. `"SExtractor source detection photometry SNR astronomical denoising"`
4. `"diffusion model denoising score matching astronomical image"`
5. `"weak lensing mass map noise residual cosmic shear"`

Each paper was scored across queries. Papers appearing in ≥2 queries received priority.

### Step 2: PDF Metadata Extraction
For papers not in `references.bib`, first-page content was extracted using **PyMuPDF** (fitz) to verify authors, journal, DOI, and year.

### Step 3: Disqualification Filters
Papers were excluded if:
- Primary domain: GW events, solar physics, medical imaging, video surveillance, CMB-only
- PDF was a paywall/error page (e.g., IEEE splash page without content)
- Primary subject was non-optical wavelengths with no optical-survey relevance

---

## Selection Breakdown

| Group | Category | Count |
|-------|----------|-------|
| A | Core Astronomical Denoising/Deconvolution | 31 |
| B | Reviews, Surveys, Benchmarks | 8 |
| C | Supporting Methods (diffusion, Poisson, plug-and-play) | 11 |
| **Total** | | **50** |

---

## Group A: Core Papers (entries 1–31)

Key decision points:
- **Akhaury appears 3x** (2022 Frontiers, 2024 Swin A&A, 2025 Euclid+Rubin A&A): same group building progressively on DRUNet/SwinUNet for optical survey images — all three are directly relevant.
- **Wang & Sreejith (arXiv:2210.01666)**: covers both XXXX_H (conference preprint) and the published PASP paper — counted as one entry.
- **2025_MM (conditional GAN)**: excluded — confirmed to be solar denoising, not optical survey.
- **2022_S (deep denoising discovery)**: excluded — PDF was IEEE paywall page, no extractable content.

---

## Group B: Reviews (entries 32–39)
Selected for their survey scope and citation breadth:
- **Huertas-Company & Lanusse 2023**: Dawes Review 10 — definitive ML in astronomy survey
- **Ramzi et al. 2021**: Wavelets in the deep learning era — bridges classical and DL methods
- **Holzschuh et al. 2022**: MNRAS generative model survey for realistic galaxy images

---

## Group C: Supporting Methods (entries 40–50)
Supporting noise models and algorithms relevant to the benchmark:
- Poisson noise (bui2023, shoushtari2025, donath2024jolideco)
- Plug-and-play priors (chen2021plugandplay)
- Radio/interferometric analogues (wang2023radio)
- Diffusion for cosmological fields (schanz2023, ono2024)

---

## Files
- `references_top50.bib` — Clean BibTeX file, 50 entries, consistent naming convention `author_year_keyword`
- This file (`TOP50_SELECTION_v2.md`) — Selection rationale

## Prior File (Not Replaced)
- `references.bib` — Original 107-entry file (kept intact, not modified)
