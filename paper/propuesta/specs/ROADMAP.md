# astro-denoise: Hoja de Ruta y Plan Técnico (2026-05-19)

## 1. Módulos y responsabilidades

- **astro-denoise-bench** (Rust, CLI principal): Orquestador, descubrimiento plug&play, gestión de datasets, invocación de módulos, recolección de métricas, reportes reproducibles.
- **astro-denoise-metrics** (Python): Cálculo de métricas científicas (completitud, pureza, precisión fotométrica), métricas técnicas (PSNR, SSIM, σ_sky), CLI/API documentado.
- **astro-denoise-bm3d** (Python): Baseline clásico, BM3D C-optimizado, output reproducible, CLI/API.
- **astro-denoise-unet** (Python): Baseline deep learning, U-Net entrenada en DC2, inferencia reproducible, pesos publicables, CLI/API.
- Otros (Noise2Void, variantes DL, etc.) pueden añadirse como `astro-denoise-<método>` siguiendo el protocolo estándar.

## 2. Protocolo de benchmark y comunicación

- Todos los módulos usan input/output vía archivos y argumentos CLI normalizados.
- Invocación típica:

```
python -m astro_denoise_bm3d --input input.fits --output output.fits --config config.json
```

- La salida estándar debe devolver JSON con status, versión y metadatos claves.
- Nuevos métodos usan entry_points Python para descubrimiento automático en el CLI principal.

## 3. Métricas implementadas

### Técnicas (universales)
- PSNR, SSIM, σ_sky
- FLUX-SNR (señal/ruido sobre flujo de fuentes reales)

### Científicas (downstream)
- **Completitud** (Recall): Fracción de fuentes reales recuperadas (vs. truth catalog)
- **Pureza** (Precision): Fracción de detecciones que son reales (vs. truth catalog)
- **Precisión fotométrica**

## 4. Contratos de métodos (API breve)

```python
class DenoiseMethod(ABC):
    @abstractmethod
    def denoise(self, patch: np.ndarray) -> np.ndarray: ...
    @abstractmethod
    def name(self) -> str: ...
    def fit(self, patches: list[np.ndarray]) -> None: pass   # Opcional
```

## 5. Decisiones técnicas claves

- Stack: Python (PyTorch, astropy, scikit-image, photutils, SEP), Rust para CLI.
- Publicación/descubrimiento: PyPI & entry_points (plugin pattern).

## 6. Dataset, reproducibilidad y extensibilidad

- Dataset principal DC2 curado & versionado.
- Benchmark abierto, reproducible, modular y adaptable a nuevos métodos, datasets y métricas.

## 7. Visión y roadmap futuro

- Plataforma reproducible, extensible y centrada en métrica científica (completitud/pureza/precisión).
- Publicación de baseline clásico y deep learning con código y datos públicos.
- Framework para experimentos comunitarios y benchmarking robusto.
- Siguientes milestones: implementación CLI, baseline deep learning, integración automática de nuevos módulos y métricas.

## 8. Principios rectores
- Reproducibilidad y transparencia total.
- Extensibilidad modular.
- Ciencia primero: métricas científicas > visuales.
- Resultados auditables y comparables.
