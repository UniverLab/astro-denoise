"""astro-denoise-bench: orchestrator for the astro-denoise benchmark ecosystem."""

__version__ = "0.1.0"

from .bench import discover_methods, run_benchmark

__all__ = ["discover_methods", "run_benchmark", "__version__"]
