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

```bash
python -m astro_denoise_bench \
    --algo passthrough \
    --manifest data/curated/manifest.json \
    --python /ruta/a/astro-denoise-passthrough/.venv/bin/python \
    --metrics-python /ruta/a/astro-denoise-metrics/.venv/bin/python
```

El orquestador invoca `python -m astro_denoise_passthrough --input ... --output ... [--config ...]`
y espera una línea JSON de status por stdout (ya implementada en `__main__.py`).

## Protocolo

- Entry point `astro_denoise.methods` -> `passthrough` (descubrimiento automático).
- Contrato `DenoiseMethod` en `src/astro_denoise_passthrough/interface.py`.
- Las métricas del benchmark se explican con `astro-denoise-metrics --explain`.
