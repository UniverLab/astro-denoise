# astro-denoise-bench

> Orquestador del ecosistema `astro-denoise`: benchmarking reproducible de
> algoritmos de denoising sobre imágenes astronómicas DESC DC2 / LSST.

Esta es la **implementación de referencia en Python**. No implementa
algoritmos ni métricas científicas — orquesta, mide y reporta. (La spec
original contempla una reimplementación futura en Rust para mediciones
computacionales desde fuera del proceso Python; el protocolo CLI es el mismo.)

## Responsabilidades

- **Descubrimiento de módulos** — detecta algoritmos instalados vía el grupo
  de entry points `astro_denoise.methods` (paquetes `astro-denoise-*`) y repos
  hermanos en disco (TUI).
- **Curación de dataset** — construye el bloque curado heterogéneo DC2:
  cutouts WCS-correctos + plano de varianza, muestreados de cada
  tract × patch × banda, descritos por un `manifest.json` reproducible.
- **Ejecución** — lanza cada algoritmo como proceso aislado con el protocolo
  CLI estándar (cada módulo conserva su propio venv) y mide tiempo de pared
  y memoria pico RSS (GNU time).
- **Evaluación** — en modo curado invoca el CLI de `astro-denoise-metrics`
  (su propio venv) contra el truth catalog Parquet real, por región y banda;
  en modo clásico usa PSNR/SSIM/σ_sky con fallback numpy. El umbral de
  detección se ancla al RMS de fondo de la **región de entrada** (se pasa
  `--thresh-image` a metrics), de modo que todos los métodos se evalúan a la
  misma profundidad absoluta: un método que aplasta el fondo ya no "detecta
  más profundo" gratis. El reporte lo registra como `thresh_reference:
  "input"`.
- **Reporte** — consolida métricas por región en un reporte JSON con resumen
  global y **por banda**; opcionalmente renderiza previews PNG
  antes/después/residual por región.
- **TUI** — dashboard interactivo (`--tui`) para ver métodos acoplados,
  el dataset curado y la comparación algoritmo × banda.

## Instalación

```bash
uv sync --all-extras
```

Esto crea `.venv/` e instala el paquete en modo editable (`src/astro_denoise_bench/`),
registrando el comando `astro-denoise-bench` y las dependencias de desarrollo
(pytest, ruff, mypy).

## CLI

```bash
# Listar módulos de métodos descubiertos (entry points instalados)
python -m astro_denoise_bench --list

# 1) Curar el bloque heterogéneo DC2 (regiones × bandas + manifest)
python -m astro_denoise_bench --curate \
    --coadd-root data/coadd-t3828-t3829-small \
    --truth-root data/truth_match \
    --curated-out data/curated \
    [--size 256] [--regions-per-image 3] [--bands z,r] [--seed 12345]

# 2) Benchmark sobre el bloque curado con el motor de métricas real
python -m astro_denoise_bench \
    --algo bm3d \
    --manifest data/curated/manifest.json \
    --python ../astro-denoise-bm3d/.venv/bin/python \
    --metrics-python ../astro-denoise-metrics/.venv/bin/python \
    [--bands z,r] [--tracts 3829] [--mag-limit 26] [--previews] \
    --output data/curated/report_bm3d.json

# `--tracts` restringe la evaluación a esos tracts: para métodos entrenados
# (p. ej. unet) evalúe SOLO en tracts excluidos de su entrenamiento, así el
# benchmark no puntúa regiones que la red ya vio.

# 3) Dashboard interactivo (métodos / dataset / comparación por banda)
python -m astro_denoise_bench --tui --manifest data/curated/manifest.json

# Explicación de cada métrica (delega en astro-denoise-metrics --explain)
python -m astro_denoise_bench --explain-metrics \
    --metrics-python ../astro-denoise-metrics/.venv/bin/python

# Crear un nuevo método con cero fricción (para otros investigadores)
python -m astro_denoise_bench --new-method wavelet [--methods-dir /ruta]
```

## Añadir un método nuevo (cero fricción)

`--new-method <nombre>` genera un repo `astro-denoise-<nombre>/` completo y
funcional: contrato `DenoiseMethod`, CLI con el protocolo JSON del bench,
entry point `astro_denoise.methods`, tests que **ya pasan**, y las mismas
convenciones del boilerplate ghscaff (layout `src/`, uv, CI centralizada,
MIT). El flujo del investigador:

```bash
python -m astro_denoise_bench --new-method wavelet
cd astro-denoise-wavelet
uv sync --all-extras && uv run pytest        # verde desde el minuto cero
# implementar denoise() en src/astro_denoise_wavelet/denoiser.py
python -m astro_denoise_bench --algo wavelet --manifest data/curated/manifest.json \
    --python .venv/bin/python --metrics-python <metrics>/.venv/bin/python
```

El esqueleto sin modificar es un pass-through: corre el benchmark tal cual y
sirve como baseline "sin denoising" en las comparaciones. El mismo template
puede publicarse como plantilla de ghscaff-boilerplate para el flujo
`ghscaff new` con repo GitHub.

### Modo clásico (pares sintéticos)

```bash
python -m astro_denoise_bench \
    --algo bm3d \
    --data-dir /path/to/data \
    --output report.json \
    [--config config.json] \
    [--python ../astro-denoise-bm3d/.venv/bin/python]
```

- `--python` / `--metrics-python` apuntan al intérprete del venv propio de
  cada módulo; por defecto usan el intérprete actual. Así cada módulo mantiene
  sus dependencias aisladas.
- `--config` se reenvía tal cual al módulo del método.
- En modo curado, si no se pasa `--config` y existe `<algo>_config.json`
  junto al manifest (p. ej. `bm3d_config.json` con
  `{"variance_hdu": "VARIANCE"}`), el runner lo usa automáticamente: así el
  método corre con la calibración del bloque (bm3d estima σ desde el plano
  de varianza en vez de MAD). El reporte registra en `config` la ruta usada
  (`null` si no hubo config).
- `--previews` escribe un PNG antes/después/residual por región
  (zscale + asinh compartido entre paneles).
- El muestreo de regiones es **pseudoaleatorio**: cada curación cubre zonas
  de cielo distintas; la semilla usada queda registrada en el manifest, así
  que un bloque publicado es exactamente regenerable con `--seed <valor>`.
- El reporte del modo curado agrupa las métricas globalmente y por banda
  (`summary.per_band`), e incluye memoria pico RSS por subproceso cuando
  GNU time está disponible.
- Cada `mean_<métrica>` del summary curado va acompañada de `std_<métrica>`
  (desviación estándar muestral entre regiones; `null` con menos de dos
  valores finitos) y `n_<métrica>`, para que las diferencias entre métodos
  puedan juzgarse contra la dispersión región-a-región.

## Distribución del bloque curado (Hugging Face)

El bloque curado es **autocontenido y portable**: el truth catalog se recorta
a la zona de cielo de las regiones (52 GB → cientos de KB) y se guarda dentro
del bloque con rutas relativas en el manifest. El dueño del dataset lo publica
una vez; cada usuario del benchmark lo descarga:

```bash
# Usuario del benchmark: descargar el bloque curado
python -m astro_denoise_bench --fetch-dataset <org>/astro-denoise-curated \
    --curated-out data/curated

# Dueño del dataset: publicar el bloque (requiere HF_TOKEN con permiso write)
python -m astro_denoise_bench --push-dataset <org>/astro-denoise-curated \
    --curated-out data/curated
```

- `--push-dataset` crea el repo dataset si no existe y excluye salidas de runs
  (`denoised_*`, `metrics_*`, `previews_*`, `report*.json`).
- Los pesos de los métodos entrenados se distribuirán por el mismo canal como
  repos de modelo.

## Layout del dataset

### Bloque curado DC2 (modo `--manifest`)

```
data/curated/
├── manifest.json          # contrato reproducible: regiones, bandas, truth (rutas relativas)
├── regions/               # cutouts FITS (imagen con WCS + HDU VARIANCE)
│   ├── t3828-p22-z-r0.fits
│   └── ...
├── truth/                 # truth catalogs recortados a la zona de las regiones
│   └── truth_tract3828.parquet
├── denoised_<algo>/       # salidas por método (no se distribuyen)
├── metrics_<algo>/        # JSON de métricas por región (no se distribuyen)
└── previews_<algo>/       # PNG antes/después/residual (no se distribuyen)
```

### Modo clásico (pares sintéticos, `--data-dir`)

```
data/
├── patch001_noisy.fits   # entrada del método
├── patch001_clean.fits   # referencia opcional (habilita PSNR/SSIM)
├── patch002_noisy.fits
└── ...
```

Archivos `*.fits` sin el sufijo `_noisy` también se aceptan como patches sin
referencia.

## Protocolo de comunicación

Cada módulo de algoritmo se invoca como proceso aislado:

```bash
python -m astro_denoise_<algo> --input patch.fits --output denoised.fits --config config.json
```

y debe devolver por stdout una línea JSON con metadatos mínimos:

```json
{"algorithm": "bm3d", "version": "0.1.0", "status": "ok"}
```

## Reporte

`report.json` contiene una entrada por patch (tiempo de pared, status,
metadatos del método, métricas) y un bloque `summary`:

```json
{
  "algorithm": "bm3d",
  "n_patches": 50,
  "summary": {
    "n_ok": 50,
    "n_failed": 0,
    "mean_wall_time_s": 1.82,
    "throughput_patches_per_s": 0.55,
    "mean_psnr": 33.9,
    "mean_ssim": 0.90,
    "mean_sky_sigma": 18.2
  },
  "runs": ["..."]
}
```

## Tests

```bash
uv run pytest
```

Además de las unidades (colección de patches, métricas fallback, summary),
la orquestación se prueba end-to-end contra un módulo de método falso
(`astro_denoise_fake`) materializado en un `PYTHONPATH` temporal, de modo
que el protocolo real de subprocesos se ejercita sin instalar denoisers.

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
astro-denoise-bench/
├── pyproject.toml
├── src/
│   └── astro_denoise_bench/
│       ├── __init__.py
│       ├── __main__.py       # CLI entry point
│       └── bench.py
├── tests/
└── README.md
```

---
Este módulo forma parte de la plataforma `astro-denoise`; véase
[`CONTEXT_v2.md`](../../CONTEXT_v2.md) para arquitectura y motivación.
