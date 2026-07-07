"""astro-denoise-passthrough: passthrough denoising method for the astro-denoise benchmark"""

__version__ = "0.1.0"

from .denoiser import PassthroughDenoiser

__all__ = ["PassthroughDenoiser", "__version__"]
