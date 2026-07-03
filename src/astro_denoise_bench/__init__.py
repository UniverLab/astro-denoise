"""astro-denoise-bench: orchestrator for the astro-denoise benchmark ecosystem."""

__version__ = "0.2.0"

from .bench import discover_methods, run_benchmark, run_curated_benchmark
from .dc2 import curate, load_manifest

__all__ = [
    "discover_methods",
    "run_benchmark",
    "run_curated_benchmark",
    "curate",
    "load_manifest",
    "__version__",
]
