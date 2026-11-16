# astro-denoise-passthrough

> passthrough denoising method for the astro-denoise benchmark

Módulo de método para la plataforma [astro-denoise](https://github.com/UniverLab).
Generado con `astro-denoise-bench --new-method passthrough` — el esqueleto ya corre
end-to-end; solo falta tu algoritmo.

## Implementar el método

Edita `src/astro_denoise_passthrough/denoiser.py` y reemplaza el cuerpo de `denoise()`
(recibe y devuelve un `ndarray` 2-D float). Los parámetros llegan por el
JSON de `--config` al constructor.

## Desarrollo

```bash
uv sync --all-extras   # crea .venv e instala en editable
uv run pytest          # los tests del esqueleto ya pasan
uv run ruff format . && uv run ruff check . && uv run mypy src/
```

## Correr en el benchmark

El bench y las métricas son el mismo paquete `astro-denoise`; prepara el
checkout core una vez y corre desde ahí:

```bash
cd /ruta/a/astro-denoise
uv sync --all-extras
uv run python -m astro_denoise.bench \
    --algo passthrough \
    --manifest data/curated/manifest.json \
    --python /ruta/a/astro-denoise-passthrough/.venv/bin/python
```

`--python` es el `.venv` de este método; las métricas usan el intérprete del
bench (`astro_denoise.metrics`, extra `metrics`), sin otro venv.

## Protocolo

- Entry point `astro_denoise.methods` -> `passthrough` (descubrimiento automático).
- Contrato `DenoiseMethod` en `src/astro_denoise_passthrough/interface.py`.
- Métricas: `python -m astro_denoise.metrics --explain`.
