# Módulo de métricas astro-denoise

> Módulo de métricas científico para el benchmark astro-denoise.

Implementa la capa de evaluación de la plataforma astro-denoise:
mide cómo un método de denoising afecta la **recuperación de fuentes**
(completitud/pureza contra catálogo truth) además de métricas clásicas de
calidad de imagen (PSNR, SSIM, σ_sky).

## Convenciones

- Los catálogos son indexables por columna y usan `x`, `y` en coordenadas de
  píxel, `flux` en unidades de la imagen y `mag` opcional. Las distancias de
  emparejamiento son euclídeas en píxeles.
- Si un catálogo no tiene `mag`, la magnitud se deriva como
  `m = -2.5·log10(flux)` solo cuando `flux > 0`; en otro caso la magnitud es
  indefinida.
- `N_det`, `N_truth` y `N_match` son los números de detecciones, fuentes truth
  y pares aceptados por el emparejamiento. `NaN` significa métrica indefinida
  (por ejemplo, denominador cero), no un valor válido de cero.
- `r` es el radio de emparejamiento en píxeles. `match_catalogs`, `completeness`
  y `purity` lo exigen como argumento; `photometric_accuracy`,
  `completeness_vs_mag` y `detection_rate` usan 2.0 px por defecto, y el CLI usa
  `--radius` con 2.0 px por defecto.

## 1. Emparejamiento de catálogos

### Definición

`match_catalogs` empareja cada detección con la fuente truth más cercana.
Para cada detección `i` se busca la fuente truth `j` que minimiza la
distancia euclídea; el par es candidato solo si `d_ij < r`. Los candidatos se
ordenan por distancia creciente y se aceptan de forma codiciosa mientras ambos
índices sigan libres, de modo que cada detección y cada fuente truth participan
en a lo sumo un par.

```text
d_ij = sqrt((x_det_i - x_truth_j)^2 + (y_det_i - y_truth_j)^2)
par (i, j) aceptado si d_ij < r, i no usado y j no usado;
asignación greedy por d_ij creciente
```

### Fuentes

- [2] ComEst estima la completitud a partir de la tasa de detección de fuentes
  sintéticas y la pureza comparando el catálogo detectado con el catálogo de
  entrada de las fuentes simuladas.
- [4] Roscani et al. evalúan métodos de denoising mediante la completitud y la
  pureza de los catálogos extraídos después del denoising.

### Desviaciones

- `r` es un parámetro libre: 2.0 px por defecto donde la API lo permite y
  `--radius` en el CLI.
- El límite es estricto: una distancia exactamente igual a `r` px no forma par
  porque la búsqueda usa `distance_upper_bound` de SciPy.
- El algoritmo es greedy por distancia tras buscar el vecino truth más cercano
  de cada detección; no resuelve empates ni ambigüedades más allá del orden por
  distancia.
- Si el vecino truth más cercano de una detección ya fue usado, la detección no
  se reempareja con el siguiente vecino dentro de `r`.

## 2. Completitud

### Definición

`completeness` es la fracción de fuentes truth recuperadas por el catálogo de
detecciones usando el emparejamiento uno-a-uno de la sección anterior.
`completeness_vs_mag` aplica la misma fracción dentro de bins de magnitud del
truth; un bin sin fuentes truth devuelve `NaN`.

```text
C = N_match / N_truth        (NaN si N_truth = 0)
```

### Fuentes

- [2] ComEst deriva la completitud como tasa de detección de fuentes
  sintéticas en función de flujo/magnitud y posición sobre imágenes CCD.
- [4] Roscani et al. miden la completitud de catálogos extraídos después de
  aplicar algoritmos de denoising.
- [3] El LSST DESC DC2 Simulated Sky Survey aporta el marco de cielo simulado y
  productos de imagen/catálogo sobre el que puede definirse una verdad de
  referencia.

### Desviaciones

- No hay inyección de fuentes en esta métrica: el truth se toma como dado y la
  completitud escalar no modela la tasa frente a flujo, posición o ruido; solo
  `completeness_vs_mag` binnea por magnitud.
- No se aplica corrección por matches casuales ni por fuentes fuera del campo.
- Para derivar magnitudes desde `flux` se usa `m = -2.5·log10(flux)`; si
  `flux <= 0` la magnitud es `NaN`.

## 3. Pureza

### Definición

`purity` es la fracción de detecciones que corresponden a una fuente truth
dentro del radio de emparejamiento.

```text
P = N_match / N_det          (NaN si N_det = 0)
```

### Fuentes

- [2] ComEst estima la pureza comparando el catálogo de fuentes detectadas con
  el catálogo de entrada de las fuentes simuladas.
- [4] Roscani et al. reportan la pureza de los catálogos obtenidos tras
  denoising imágenes simuladas y reales.

### Desviaciones

- No hay estimación de falsos positivos basada en ruido: la pureza es solo
  catálogo contra truth. NoiseChisel [5] sí estima falsos positivos usando el
  ruido ambiental como referencia.
- Una detección espuria que caiga dentro de `r` de una fuente truth puede contar
  como emparejada y por tanto como pura; el radio controla esa contaminación.
- Solo es comparable entre métodos cuando la profundidad de detección está
  fijada, por ejemplo con `--thresh-image`; sin ella cada método usa su propio
  RMS de fondo y `N_det` no es comparable.

## 4. Exactitud fotométrica

### Definición

`photometric_accuracy` usa los pares aceptados por el emparejamiento y compara
los flujos detectado y truth. Solo entran pares con `flux_truth` finito y
positivo y `flux_det` finito.

```text
rel_i = (F_det_i - F_truth_i) / F_truth_i
bias_flux = median(rel)
MAD(rel) = median(|rel - median(rel)|)
scatter_flux = 1.4826 * MAD(rel)

dmag_i = -2.5 * log10(F_det_i / F_truth_i)   (solo si F_det_i > 0)
bias_mag = median(dmag)
scatter_mag = 1.4826 * MAD(dmag)
```

### Fuentes

- [4] Roscani et al. evalúan la capacidad de los métodos de denoising para
  preservar los flujos y las formas de los objetos.
- [6] Huang et al. caracterizan la precisión fotométrica del pipeline HSC,
  incluyendo dispersión estadística en magnitud para estrellas y galaxias
  sintéticas.

### Desviaciones

- La dispersión es `1.4826 * MAD`, no la desviación estándar ni una sigma
  ajustada por un modelo.
- El sesgo es la mediana del error relativo, no la media ni una pendiente de
  calibración.
- Las estadísticas de magnitud solo usan detecciones con `F_det > 0`; los pares
  con flujo detectado no positivo siguen entrando en las estadísticas de flujo.
- Si el catálogo detectado está en ADU instrumentales y el truth DC2 está en
  nJy, los biases absolutos no son interpretables; solo las dispersiones
  relativas tienen significado.

## 5. PSNR

### Definición

`psnr` compara la imagen denoiseada con la referencia sin ruido
`--original`/`original`. Usa el error cuadrático medio y el rango dinámico de
la referencia.

```text
MSE = mean((I_ref - I_dn)^2)
R = nanmax(I_ref) - nanmin(I_ref)   (si R <= 0, se usa R = 1)
PSNR = 10 * log10(R^2 / MSE)
```

Si `MSE = 0`, PSNR es infinito.

### Fuentes

- [8] Huynh-Thu & Ghanbari discuten el ámbito de validez de PSNR en la
  evaluación de calidad de imagen/vídeo.
- [9] Hore & Ziou comparan PSNR y SSIM como métricas de calidad de imagen.

### Desviaciones

- El rango dinámico `R` es `max - min` de la imagen de referencia, no un valor
  fijo como 255 u 1.0; por eso los PSNR de imágenes con rangos distintos no
  son directamente comparables.
- Se usan los píxeles en bruto, sin máscara de fuentes ni whitening del ruido.
- Se mantiene a pesar de las cautelas de [8] porque es un diagnóstico pixel-wise
  estándar y complementario a las métricas científicas.

## 6. SSIM

### Definición

`ssim` usa `skimage.metrics.structural_similarity` con el mismo rango dinámico
`R` que PSNR. Para cada posición válida se combinan medias, varianzas y
covarianza locales de la referencia y la imagen denoiseada; el resultado es la
media del mapa SSIM tras recortar el borde.

```text
SSIM = mean_valid [ (2 mu_x mu_y + C1)(2 sigma_xy + C2)
                    / ((mu_x^2 + mu_y^2 + C1)(sigma_x^2 + sigma_y^2 + C2)) ]
C1 = (0.01 R)^2,  C2 = (0.03 R)^2
```

### Fuentes

- [10] Wang et al. definen SSIM a partir de luminancia, contraste y estructura
  con una ventana local y constantes `K1 = 0.01`, `K2 = 0.03`.
- [9] Hore & Ziou comparan PSNR y SSIM para evaluar calidad de imagen.

### Desviaciones

- La ventana es la de scikit-image por defecto: 7×7 uniforme, no la ventana
  gaussiana 11×11 con `sigma = 1.5` de Wang et al. [10].
- Se usa covarianza muestral y se recorta el borde según los valores por
  defecto de scikit-image.
- Las constantes `K1`/`K2` siguen la formulación de [10], pero el rango `R` es
  el `max - min` de la referencia, igual que en PSNR.

## 7. Ruido de fondo

### Definición

`sky_sigma` estima el ruido residual del fondo como la desviación estándar de
los píxeles que sobreviven al recorte iterativo sigma-clipped de
`astropy.stats.sigma_clipped_stats` con `sigma = 3.0` y `maxiters = 5`. El
argumento `mask` excluye píxeles donde `mask = True`. En el CLI la máscara se
construye con `source_mask`, es decir, las fuentes detectadas dilatadas.

```text
sigma_sky = std_sigma_clipped(image; mask, sigma = 3.0, maxiters = 5)
```

### Fuentes

- [1] SExtractor es la referencia clásica de extracción y medida de fuentes en
  imágenes astronómicas, donde el fondo y el ruido condicionan la detección.
- [5] NoiseChisel usa el ruido ambiental como referencia para estimar y restar
  falsos positivos en la detección de objetos nebulosos.

### Desviaciones

- Es un único sigma global, no una malla espacial de fondo ni una estimación
  local por región.
- En el CLI la máscara son las detecciones dilatadas (`source_mask`,
  `dilate = 3`), no una máscara manual ni una estimación de fondo por bloques.
- `background_rms` es una estadística distinta usada para anclar `--thresh` con
  `--thresh-image` (SEP `globalrms`, con fallback sigma-clipped); no es lo
  mismo que `sky_sigma`.

## Referencias

1. E. Bertin & S. Arnouts, “SExtractor: Software for source extraction,”
   Astronomy & Astrophysics Supplement Series 117, 393–404 (1996).
   [DOI](https://doi.org/10.1051/aas:1996164).
2. I.-N. Chiu, S. Desai & J. Liu, “ComEst: A Completeness Estimator of Source
   Extraction on Astronomical Imaging,” Astronomy and Computing (2016),
   arXiv:1605.02650. [DOI](https://doi.org/10.1016/j.ascom.2016.04.005).
3. LSST Dark Energy Science Collaboration, “The LSST DESC DC2 Simulated Sky
   Survey,” The Astrophysical Journal Supplement Series 253, 31 (2021),
   arXiv:2010.05926. [DOI](https://doi.org/10.3847/1538-4365/abd62c).
4. V. Roscani et al., “A comparative analysis of denoising algorithms for
   extragalactic imaging surveys,” Astronomy & Astrophysics 643, A43 (2020),
   arXiv:2009.01846. [DOI](https://doi.org/10.1051/0004-6361/201936278).
5. M. Akhlaghi & T. Ichikawa, “Noise Based Detection and Segmentation of
   Nebulous Objects,” The Astrophysical Journal Supplement Series 220, 1
   (2015), arXiv:1505.01664.
   [DOI](https://doi.org/10.1088/0067-0049/220/1/1).
6. S. Huang et al., “Characterization and photometric performance of the Hyper
   Suprime-Cam Software Pipeline,” Publications of the Astronomical Society of
   Japan 70, S6 (2018). [DOI](https://doi.org/10.1093/pasj/psx126).
7. J. Bosch et al., “The Hyper Suprime-Cam Software Pipeline,” Publications of
   the Astronomical Society of Japan 70, S5 (2018), arXiv:1705.06766.
   [DOI](https://doi.org/10.1093/pasj/psx080).
8. Q. Huynh-Thu & M. Ghanbari, “Scope of validity of PSNR in image/video
   quality assessment,” Electronics Letters 44(13), 800 (2008).
   [DOI](https://doi.org/10.1049/el:20080522).
9. A. Hore & D. Ziou, “Image quality metrics: PSNR vs. SSIM,” Proceedings of
   the International Conference on Pattern Recognition (ICPR), 2010.
   [DOI](https://doi.org/10.1109/ICPR.2010.579).
10. Z. Wang, A. C. Bovik, H. R. Sheikh & E. P. Simoncelli, “Image quality
    assessment: from error visibility to structural similarity,” IEEE
    Transactions on Image Processing 13(4), 600–612 (2004).
    [DOI](https://doi.org/10.1109/TIP.2003.819861).

## Instalación

```bash
uv sync --all-extras
```

Esto crea `.venv/` e instala el paquete en modo editable (`src/astro_denoise/metrics/`)
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
from astro_denoise.metrics import completeness, purity, detect_sources

catalog = detect_sources(denoised_image, thresh=1.5)
c = completeness(catalog, truth, radius=2.0)
p = purity(catalog, truth, radius=2.0)
```

## CLI

```bash
python -m astro_denoise.metrics \
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
  el σ usado queda registrado en el JSON (`parameters.thresh_rms`).
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
python -m astro_denoise.metrics --explain
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
python -m astro_denoise.metrics \
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
  estado (`{"module": "astro_denoise.metrics", "status": "ok", ...}`) para que
  orquestadores como `astro-denoise-bench` parseen el resultado.
- El resumen legible se imprime por stderr vía `rich`.

## Ejemplo de output

```json
{
  "module": "astro_denoise.metrics",
  "version": "0.4.0",
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
src/astro_denoise/metrics/
├── __init__.py
├── __main__.py       # CLI entry point
├── detect.py
├── explain.py
├── metrics.py
└── truth.py
```

---
Este módulo forma parte de la plataforma `astro-denoise`.
