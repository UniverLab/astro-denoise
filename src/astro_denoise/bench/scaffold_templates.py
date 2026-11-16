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
license = "Apache-2.0"
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

El bench y las métricas son el mismo paquete `astro-denoise`; prepara el
checkout core una vez y corre desde ahí:

```bash
cd /ruta/a/astro-denoise
uv sync --all-extras
uv run python -m astro_denoise.bench \\
    --algo @NAME@ \\
    --manifest data/curated/manifest.json \\
    --python /ruta/a/@REPO@/.venv/bin/python
```

`--python` es el `.venv` de este método; las métricas usan el intérprete del
bench (`astro_denoise.metrics`, extra `metrics`), sin otro venv.

## Protocolo

- Entry point `astro_denoise.methods` -> `@NAME@` (descubrimiento automático).
- Contrato `DenoiseMethod` en `src/@PKG@/interface.py`.
- Métricas: `python -m astro_denoise.metrics --explain`.
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
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

"""
