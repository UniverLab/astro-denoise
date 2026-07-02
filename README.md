# astro-denoise-bench

> Orquestador del ecosistema `astro-denoise`: benchmarking reproducible de
> algoritmos de denoising sobre imágenes astronómicas DESC DC2 / LSST.

Esta es la **implementación de referencia en Python**. No implementa
algoritmos ni métricas científicas — orquesta, mide y reporta. (La spec
original contempla una reimplementación futura en Rust para mediciones
computacionales desde fuera del proceso Python; el protocolo CLI es el mismo.)

## Responsabilidades

- **Descubrimiento de módulos** — detecta algoritmos instalados vía el grupo
  de entry points `astro_denoise.methods` (paquetes `astro-denoise-*`).
- **Ejecución** — lanza cada algoritmo como proceso aislado con el protocolo
  CLI estándar, de modo que cada módulo conserva su propio venv.
- **Evaluación** — invoca `astro_denoise_metrics` (PSNR/SSIM/σ_sky) cuando
  está disponible; incluye fallback numpy para PSNR/σ_sky.
- **Reporte** — consolida tiempos y métricas por patch en un reporte JSON
  comparativo con resumen agregado (throughput, medias).

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

# Ejecutar un benchmark
python -m astro_denoise_bench \
    --algo bm3d \
    --data-dir /path/to/data \
    --output report.json \
    [--config config.json] \
    [--python ../astro-denoise-bm3d/.venv/bin/python]
```

- `--python` apunta al intérprete del venv propio del módulo de método; por
  defecto usa el intérprete actual. Esto permite que cada módulo mantenga sus
  dependencias aisladas.
- `--config` se reenvía tal cual al módulo del método.

## Layout del dataset

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
