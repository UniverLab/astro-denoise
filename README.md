# astro-denoise-metrics

> Módulo de métricas downstream científico para el benchmark astro-denoise.

Implementa la capa de evaluación de la plataforma [astro-denoise](../../CONTEXT_v2.md):
mide cómo un método de denoising afecta la **recuperación de fuentes**
(completitud/pureza contra catálogo truth) además de métricas clásicas de
calidad de imagen (PSNR, SSIM, σ_sky).

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .   # opcional: registra el console script astro-denoise-metrics
```

## API

Los catálogos son objetos indexables por columna (astropy `Table`, structured
array de numpy, dict de arrays) con `x`, `y` en coordenadas de píxel y
opcionalmente `flux` / `mag`.

| Función | Descripción |
|---------|-------------|
| `completeness(catalog, truth, radius)` | Fracción de fuentes verdaderas recuperadas (match 1-a-1 dentro de `radius` px) |
| `purity(catalog, truth, radius)` | Fracción de detecciones que corresponden a fuentes reales |
| `completeness_vs_mag(catalog, truth, mag_bins, radius=2.0)` | Curva de completitud binneada por magnitud truth |
| `photometric_accuracy(catalog, truth, radius=2.0)` | Bias mediano y dispersión robusta de flujo en pares matched |
| `psnr(original, denoised)` | Peak signal-to-noise ratio (dB) |
| `ssim(original, denoised)` | Índice de similitud estructural |
| `sky_sigma(image, mask=None)` | Ruido residual de fondo (std sigma-clipped) |
| `detect_sources(image, thresh=1.5, minarea=5)` | Detección con SEP (fallback photutils), retorna `x, y, flux` |

```python
from astro_denoise_metrics import completeness, purity, detect_sources

catalog = detect_sources(denoised_image, thresh=1.5)
c = completeness(catalog, truth, radius=2.0)
p = purity(catalog, truth, radius=2.0)
```

## CLI

```bash
python -m astro_denoise_metrics \
    --input denoised.fits \
    --truth truth.fits \
    --output results.json \
    [--original clean.fits] \
    [--radius 2.0] [--thresh 1.5] [--minarea 5]
```

- `--truth`: tabla binaria FITS con columnas `x`, `y`, `flux` (y opcionalmente
  `mag`); se reconocen alias estilo SExtractor (`X_IMAGE`, `xcentroid`, …).
- `--original`: imagen de referencia sin ruido; habilita PSNR/SSIM.
- `results.json` contiene todas las métricas; stdout emite una línea JSON de
  estado (`{"module": "astro-denoise-metrics", "status": "ok", ...}`) para que
  orquestadores como `astro-denoise-bench` parseen el resultado.
- El resumen legible se imprime por stderr vía `rich`.

## Ejemplo de output

```json
{
  "module": "astro-denoise-metrics",
  "version": "0.1.0",
  "n_detected": 142,
  "n_truth": 150,
  "completeness": 0.89,
  "purity": 0.94,
  "sky_sigma": 18.7,
  "photometric_accuracy": {"n_matched": 133, "flux_bias": -0.02, "flux_scatter": 0.08},
  "completeness_vs_mag": {"mag_centers": ["..."], "completeness": ["..."]},
  "psnr": 34.2,
  "ssim": 0.91
}
```

---
Este submódulo forma parte de la plataforma `astro-denoise`; véase
[`CONTEXT_v2.md`](../../CONTEXT_v2.md) para arquitectura y criterios.
