# astro-denoise-metrics

> Módulo de métricas downstream científico para el benchmark astro-denoise.

Implementa la capa de evaluación de la plataforma [astro-denoise](../../CONTEXT_v2.md):
mide cómo un método de denoising afecta la **recuperación de fuentes**
(completitud/pureza contra catálogo truth) además de métricas clásicas de
calidad de imagen (PSNR, SSIM, σ_sky).

## Instalación

```bash
uv sync --all-extras
```

Esto crea `.venv/` e instala el paquete en modo editable (`src/astro_denoise_metrics/`)
junto con las dependencias de desarrollo (pytest, ruff, mypy).

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
| `mse(original, denoised)` / `rmse(...)` / `mae(...)` | Métricas clásicas pixel-wise |
| `fourier_sharpness(image, freq_cutoff=0.5)` | Nitidez espectral S_F: fracción de potencia Fourier en alta frecuencia (AstroClearNet) |
| `flux_snr(catalog, sigma)` | SNR de apertura mediano de las detecciones (`flux / (σ·√npix)`) |
| `detection_rate(image, thresholds, truth=None, err=None)` | Barrido de umbrales: n_detected (+ completitud/pureza con truth) |
| `detect_sources(image, thresh=1.5, minarea=5, err=None)` | Detección con SEP (fallback photutils), retorna `x, y, flux, npix` |
| `background_rms(image)` | RMS global del fondo (SEP, fallback sigma-clipped); úselo sobre el input para fijar `err` |
| `load_dc2_truth(path, band)` | Lee `ra`/`dec`/`flux_<band>` de un Parquet DC2 (filtra duplicados) |
| `project_truth(truth, wcs, shape, mag_limit=None)` | Proyecta el truth al espacio de píxeles vía WCS |

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
    [--radius 2.0] [--thresh 1.5] [--minarea 5] \
    [--thresh-image noisy.fits] \
    [--thresh-sweep "1,1.5,2,3,5"]
```

- `--thresh-image`: imagen FITS (típicamente el **input ruidoso**) sobre la que
  se mide el RMS de fondo que escala `--thresh`. Sin ella, el umbral es
  relativo al RMS de la imagen ya denoiseada, lo que **no es comparable entre
  métodos**: un método que aplasta el fondo baja su propio σ y "detecta más
  profundo" gratis (infla n_detected y hunde la pureza). Con `--thresh-image`
  todos los métodos se umbralizan a la misma profundidad absoluta de flujo;
  el σ usado queda registrado en el JSON (`parameters.thresh_rms`). Esta es la
  decisión metodológica adoptada para la deuda D3 del proyecto: umbral =
  `--thresh × RMS(input)`, en unidades absolutas de la imagen.
- `--thresh-sweep`: barrido de umbrales de detección; agrega al JSON la curva
  `detection_rate` (n_detected + completitud/pureza por umbral), el insumo del
  trade-off completitud/pureza. Respeta `--thresh-image`, de modo que el
  barrido también es comparable entre métodos.
- Con `--original` se reportan además PSNR, SSIM, MSE, RMSE, MAE y el ratio de
  nitidez S_F (denoised/original).
- `--explain`: imprime una tabla con **qué mide cada métrica, cómo
  interpretarla y su referencia** en la literatura, y termina. No requiere
  otros argumentos:

```bash
python -m astro_denoise_metrics --explain
```

- `--truth`: tabla binaria FITS con columnas `x`, `y`, `flux` (y opcionalmente
  `mag`); se reconocen alias estilo SExtractor (`X_IMAGE`, `xcentroid`, …).
  También acepta un catálogo Parquet DC2 `truth_match` (ver abajo).
- `--original`: imagen de referencia sin ruido; habilita PSNR/SSIM.

### Catálogo truth DC2 (Parquet)

Si `--truth` apunta a un Parquet del DESC DC2 (`truth_tract*.parquet`, con
`ra`/`dec` en grados y flujos por banda en nJy), el CLI proyecta las fuentes
sobre el WCS de la imagen `--input` y filtra a sus límites:

```bash
python -m astro_denoise_metrics \
    --input denoised.fits \
    --truth truth_tract3828.parquet \
    --band z --mag-limit 26 \
    --output results.json
```

- `--band` (obligatorio con Parquet): selecciona `flux_<band>` (u/g/r/i/z/y).
- `--mag-limit`: descarta fuentes truth más débiles que esa magnitud AB
  (`m = -2.5·log10(f/nJy) + 31.4`), útil para restringir el truth al rango
  detectable de la imagen.
- La imagen `--input` debe conservar un WCS válido para su grilla de píxeles
  (al recortar coadds usar p. ej. `astropy.nddata.Cutout2D`, que lo ajusta).
- Con truth DC2, `photometric_accuracy` compara flujo instrumental (ADU)
  contra nJy: solo la dispersión relativa es interpretable, no el bias.
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

## Tests

```bash
uv run pytest
```

Cubren las métricas sobre catálogos/imágenes sintéticas y el protocolo CLI
end-to-end (FITS + truth catalog temporales).

## Desarrollo

```bash
uv run ruff format .
uv run ruff check .
uv run mypy src/
```

Las cuatro comprobaciones (format, lint, mypy, pytest) corren en CI vía el
workflow reusable [`UniverLab/workflows`](https://github.com/UniverLab/workflows).

## Estructura del proyecto

```
astro-denoise-metrics/
├── pyproject.toml
├── src/
│   └── astro_denoise_metrics/
│       ├── __init__.py
│       ├── __main__.py       # CLI entry point
│       ├── detect.py
│       └── metrics.py
├── tests/
└── README.md
```

---
Este submódulo forma parte de la plataforma `astro-denoise`; véase
[`CONTEXT_v2.md`](../../CONTEXT_v2.md) para arquitectura y criterios.
