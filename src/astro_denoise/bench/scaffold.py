"""Zero-friction scaffold for new denoising method modules.

``--new-method <name>`` writes a complete, immediately runnable
``astro-denoise-<name>/`` repository implementing the platform
protocol: ``DenoiseMethod`` contract, CLI with JSON status line,
``astro_denoise.methods`` entry point, passing tests and the same
conventions as the ghscaff ``python-module`` boilerplate (src layout,
uv, centralized CI). A researcher only edits ``denoise()``.

The generated denoiser is a working pass-through so the whole loop
(scaffold -> uv sync -> pytest -> bench run) succeeds before a single
line of algorithm code is written.
"""

from __future__ import annotations

import re
from pathlib import Path

from .scaffold_templates import (
    CI_WORKFLOW,
    GITIGNORE,
    LICENSE,
    PKG_DENOISER,
    PKG_INIT,
    PKG_INTERFACE,
    PKG_MAIN,
    PYPROJECT_TOML,
    README,
    TEST_CLI,
    TEST_DENOISER,
)


def _validate_name(name):
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise ValueError(
            f"invalid method name {name!r}: use lowercase letters/digits/underscore "
            "(e.g. 'wavelet', 'noise2void' -> 'noise2void')"
        )
    return name


def _render(template, pkg, repo, name, description):
    return (
        template.replace("@PKG@", pkg)
        .replace("@REPO@", repo)
        .replace("@NAME_CAP@", name.capitalize())
        .replace("@NAME@", name)
        .replace("@DESCRIPTION@", description)
    )


def _files(name, description):
    pkg = f"astro_denoise_{name}"
    repo = f"astro-denoise-{name}"
    templates = {
        "pyproject.toml": PYPROJECT_TOML,
        f"src/{pkg}/__init__.py": PKG_INIT,
        f"src/{pkg}/interface.py": PKG_INTERFACE,
        f"src/{pkg}/denoiser.py": PKG_DENOISER,
        f"src/{pkg}/__main__.py": PKG_MAIN,
        "tests/test_denoiser.py": TEST_DENOISER,
        "tests/test_cli.py": TEST_CLI,
        "README.md": README,
        ".gitignore": GITIGNORE,
        ".github/workflows/ci.yml": CI_WORKFLOW,
        "LICENSE": LICENSE,
    }
    return {
        path: _render(template, pkg, repo, name, description)
        for path, template in templates.items()
    }


def scaffold_method(name, out_dir=None, description=None):
    """Write the ``astro-denoise-<name>`` skeleton; returns its path.

    Refuses to overwrite an existing directory.
    """
    name = _validate_name(name)
    description = description or f"{name} denoising method for the astro-denoise benchmark"
    repo_dir = Path(out_dir or Path.cwd()) / f"astro-denoise-{name}"
    if repo_dir.exists():
        raise FileExistsError(f"{repo_dir} already exists")

    for rel_path, content in _files(name, description).items():
        target = repo_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    return repo_dir
