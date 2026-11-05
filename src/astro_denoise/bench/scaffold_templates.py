"""Templates of the files written by ``astro-denoise-bench --new-method``.

One module-level string per generated file. The placeholders ``@PKG@``,
``@REPO@``, ``@NAME@``, ``@NAME_CAP@`` and ``@DESCRIPTION@`` are rendered by
``scaffold._render`` with plain ``str.replace`` calls, so these are ordinary
strings and not f-strings: every brace below is a literal brace of the
generated file.
"""

PYPROJECT_TOML = """\
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "@REPO@"
version = "0.1.0"
description = "@DESCRIPTION@"
readme = "README.md"
requires-python = ">=3.10"
license = "MIT"
dependencies = [
    "numpy>=1.26",
    "astropy>=6.0",
    "rich>=13.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "ruff>=0.8", "mypy>=1.13"]

[project.entry-points."astro_denoise.methods"]
@NAME@ = "@PKG@.denoiser:@NAME_CAP@Denoiser"

[tool.setuptools.packages.find]
where = ["src"]
include = ["@PKG@*"]

[tool.ruff]
target-version = "py310"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "N", "UP", "B", "SIM"]

[tool.mypy]
python_version = "3.12"
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
"""

PKG_INIT = '''\
"""@REPO@: @DESCRIPTION@"""

__version__ = "0.1.0"

from .denoiser import @NAME_CAP@Denoiser

__all__ = ["@NAME_CAP@Denoiser", "__version__"]
'''

PKG_INTERFACE = '''\
"""Standard denoising-method contract of the astro-denoise platform.

Every algorithm module ships (a copy of) this ABC so methods stay
importable without a shared runtime dependency; `astro-denoise-bench`
only relies on the duck-typed ``denoise`` / ``name`` protocol.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class DenoiseMethod(ABC):
    """A denoising algorithm operating on 2-D image patches."""

    @abstractmethod
    def denoise(self, patch: np.ndarray) -> np.ndarray:
        """Return the denoised version of ``patch`` (same shape/dtype float)."""

    @abstractmethod
    def name(self) -> str:
        """Short identifier of the method."""

    def fit(self, patches: list[np.ndarray]) -> None:  # noqa: B027
        """Optional training/calibration step; no-op for classical methods."""
'''

PKG_DENOISER = '''\
"""@NAME@ denoiser implementing the astro-denoise contract.

Replace the body of :meth:`denoise` with your algorithm. It receives a
2-D float array (one patch / region) and must return an array of the
same shape. Config keys arrive through :class:`@NAME_CAP@Denoiser`'s
constructor (forwarded from the CLI ``--config`` JSON).
"""

from __future__ import annotations

import numpy as np

from .interface import DenoiseMethod


class @NAME_CAP@Denoiser(DenoiseMethod):
    """TODO: describe the method and its parameters."""

    def __init__(self, **config):
        self.config = config

    def name(self) -> str:
        return "@NAME@"

    def denoise(self, patch: np.ndarray) -> np.ndarray:
        patch = np.asarray(patch, dtype=np.float64)
        if patch.ndim != 2:
            raise ValueError(f"expected a 2-D patch, got shape {patch.shape}")
        # TODO: implement the actual denoising. The pass-through below
        # keeps the module runnable end-to-end from the first minute.
        return patch.copy()
'''

PKG_MAIN = '''\
"""CLI: denoise a FITS image following the astro-denoise bench protocol.

Usage:
    python -m @PKG@ --input input.fits --output output.fits \\
        [--config config.json]

stdout emits a single JSON status line consumed by the orchestrator:
    {"algorithm": "@NAME@", "version": "0.1.0", "status": "ok"}
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from astropy.io import fits
from rich.console import Console

from . import __version__
from .denoiser import @NAME_CAP@Denoiser


def load_config(path):
    if path is None:
        return {}
    return json.loads(Path(path).read_text())


def find_image_hdu(hdul):
    for index, hdu in enumerate(hdul):
        if hdu.data is not None and np.asarray(hdu.data).ndim == 2:
            return index
    raise ValueError("no 2-D image HDU found in input FITS")


def main(argv=None):
    console = Console(stderr=True)
    parser = argparse.ArgumentParser(
        prog="python -m @PKG@",
        description="@DESCRIPTION@",
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        denoiser = @NAME_CAP@Denoiser(**config)

        with fits.open(args.input) as hdul:
            image_hdu = find_image_hdu(hdul)
            image = np.asarray(hdul[image_hdu].data, dtype=np.float64)
            header = hdul[image_hdu].header.copy()

        start = time.perf_counter()
        denoised = denoiser.denoise(image)
        elapsed = time.perf_counter() - start

        header["HISTORY"] = f"@REPO@ {__version__}"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fits.PrimaryHDU(data=denoised.astype(np.float32), header=header).writeto(
            args.output, overwrite=True
        )
        console.print(f"[green]ok[/green] {args.input.name} -> {args.output.name}")
        print(
            json.dumps(
                {
                    "algorithm": "@NAME@",
                    "version": __version__,
                    "status": "ok",
                    "elapsed_s": round(elapsed, 4),
                }
            )
        )
        return 0
    except Exception as exc:
        console.print(f"[red]error:[/red] {exc}")
        print(
            json.dumps(
                {
                    "algorithm": "@NAME@",
                    "version": __version__,
                    "status": "error",
                    "error": str(exc),
                }
            )
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
'''

TEST_DENOISER = """\
import numpy as np
import pytest

from @PKG@ import @NAME_CAP@Denoiser


def test_preserves_shape():
    rng = np.random.default_rng(0)
    patch = rng.normal(100.0, 2.0, (64, 64))
    out = @NAME_CAP@Denoiser().denoise(patch)
    assert out.shape == patch.shape


def test_rejects_non_2d():
    with pytest.raises(ValueError):
        @NAME_CAP@Denoiser().denoise(np.zeros(10))
"""

TEST_CLI = """\
import json

import numpy as np
from astropy.io import fits

from @PKG@.__main__ import main


def test_cli_protocol(tmp_path, capsys):
    image = np.random.default_rng(0).normal(100.0, 2.0, (32, 32)).astype(np.float32)
    fits.PrimaryHDU(data=image).writeto(tmp_path / "in.fits")

    rc = main(["--input", str(tmp_path / "in.fits"), "--output", str(tmp_path / "out.fits")])
    assert rc == 0
    assert (tmp_path / "out.fits").exists()
    status = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert status["algorithm"] == "@NAME@"
    assert status["status"] == "ok"
"""

README = """\
# @REPO@

> @DESCRIPTION@

Módulo de método para la plataforma [astro-denoise](https://github.com/UniverLab).
Generado con `astro-denoise-bench --new-method @NAME@` — el esqueleto ya corre
end-to-end; solo falta tu algoritmo.

## Implementar el método

Edita `src/@PKG@/denoiser.py` y reemplaza el cuerpo de `denoise()`
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
python -m astro_denoise.bench \\
    --algo @NAME@ \\
    --manifest data/curated/manifest.json \\
    --python /ruta/a/@REPO@/.venv/bin/python \\
    --metrics-python /ruta/a/astro-denoise-metrics/.venv/bin/python
```

El orquestador invoca `python -m @PKG@ --input ... --output ... [--config ...]`
y espera una línea JSON de status por stdout (ya implementada en `__main__.py`).

## Protocolo

- Entry point `astro_denoise.methods` -> `@NAME@` (descubrimiento automático).
- Contrato `DenoiseMethod` en `src/@PKG@/interface.py`.
- Las métricas del benchmark se explican con `astro-denoise-metrics --explain`.
"""

GITIGNORE = """\
__pycache__/
*.py[cod]
*.egg-info/
build/
dist/
.venv/
venv/
.pytest_cache/
.mypy_cache/
.coverage
*.fits
.env
"""

CI_WORKFLOW = """\
name: CI

on:
  pull_request:
  workflow_dispatch:

jobs:
  python-ci:
    uses: UniverLab/workflows/.github/workflows/python-ci.yml@main
    with:
      python-version: "3.12"
"""

LICENSE = """\
MIT License

Copyright (c) 2026 UniverLab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
