"""Human-readable documentation of every metric the module reports.

Consumed by the CLI ``--explain`` flag so any benchmark user can ask
the tool itself what each number means, how to read it and where it
comes from, without leaving the terminal.
"""

from __future__ import annotations

#: metric key -> (qué mide, cómo interpretarla, referencia/origen)
METRIC_INFO: dict[str, tuple[str, str, str]] = {
    "completeness": (
        "Fracción de fuentes reales del truth catalog recuperadas por la "
        "detección tras el denoising (match 1-a-1 dentro de --radius px).",
        "1.0 = se recuperan todas. Cae hacia magnitudes débiles; compárela "
        "siempre junto a la pureza (subir una suele bajar la otra).",
        "Recall estándar en catálogos (GREAT3, LSST preparedness).",
    ),
    "purity": (
        "Fracción de las detecciones que corresponden a fuentes reales "
        "(no artefactos ni alucinaciones del método).",
        "1.0 = ninguna detección espuria. Un denoiser que 'inventa' "
        "estructura la degrada aunque la imagen se vea más limpia. "
        "Solo es comparable entre métodos con --thresh-image (σ medido "
        "sobre el input): si no, aplastar el fondo baja el σ propio y "
        "'detecta más profundo' gratis.",
        "Precision estándar en catálogos; núcleo del trade-off científico.",
    ),
    "completeness_vs_mag": (
        "Curva de completitud binneada por magnitud del truth.",
        "Más informativa que el escalar: muestra hasta qué magnitud el "
        "método recupera fuentes y dónde colapsa.",
        "Práctica estándar en surveys (curvas de profundidad).",
    ),
    "photometric_accuracy": (
        "Bias (mediana) y dispersión robusta (MAD·1.4826) del error "
        "relativo de flujo y del offset de magnitud en pares matched.",
        "Bias ~0 y scatter bajo = el denoising no altera la fotometría. "
        "Con truth DC2 (nJy) vs flujo instrumental (ADU) solo la "
        "dispersión relativa es interpretable, no el bias absoluto.",
        "Preservación de flujo, requisito de ciencia LSST.",
    ),
    "detection_rate": (
        "Barrido de umbrales de detección (--thresh-sweep): número de "
        "detecciones y completitud/pureza en cada umbral.",
        "Materializa la curva del trade-off: umbral bajo = más completitud "
        "y menos pureza; umbral alto = lo contrario.",
        "Source detection rate con SExtractor (BGRem 2025).",
    ),
    "flux_snr": (
        "SNR de apertura mediano de las detecciones: flux / (σ_sky · √npix) por fuente.",
        "Más alto = fuentes más significativas sobre el fondo residual. "
        "Útil para comparar cuánta señal útil deja cada método.",
        "FLUX-SNR del roadmap del proyecto (señal/ruido sobre fuentes).",
    ),
    "psnr": (
        "Peak signal-to-noise ratio (dB) contra la imagen de referencia sin ruido (--original).",
        "Más alto = reconstrucción pixel-wise más fiel. No garantiza "
        "validez científica: puede subir mientras la fotometría se daña.",
        "Estándar universal de comparación en denoising.",
    ),
    "ssim": (
        "Structural similarity index contra la referencia (--original): "
        "luminancia, contraste y estructura.",
        "1.0 = idéntica. Sensible a suavizado excesivo que PSNR perdona.",
        "Wang et al. 2004; estándar en la literatura revisada.",
    ),
    "mse": (
        "Error cuadrático medio contra la referencia (--original).",
        "Más bajo = mejor ajuste pixel-wise; penaliza errores grandes.",
        "Métrica clásica base (PSNR se deriva de ella).",
    ),
    "rmse": (
        "Raíz del MSE, en las mismas unidades de la imagen.",
        "Interpretable directamente como error típico por píxel.",
        "Métrica clásica.",
    ),
    "mae": (
        "Error absoluto medio contra la referencia (--original).",
        "Más robusto a outliers que MSE; error medio por píxel.",
        "Métrica clásica.",
    ),
    "sky_sigma": (
        "Ruido residual del fondo de cielo: std sigma-clipped de los "
        "píxeles sin fuentes (máscara de detecciones dilatada).",
        "Más bajo = fondo más limpio. Compárelo con el σ del input para "
        "medir la reducción efectiva de ruido.",
        "σ_sky de AstroClearNet (2025).",
    ),
    "sharpness_sf": (
        "Nitidez espectral S_F: fracción de la potencia de Fourier "
        "(sin DC) en frecuencias radiales altas (> 0.5 Nyquist).",
        "Sobre-suavizado la baja; artefactos de alta frecuencia "
        "'alucinados' la suben. Con --original se reporta el ratio "
        "denoised/original (~1 = conserva la nitidez).",
        "Sharpness Fourier-based de AstroClearNet (2025).",
    ),
    "n_detected / n_truth": (
        "Conteos crudos: detecciones en la imagen y fuentes del truth "
        "dentro del campo (tras --mag-limit si aplica).",
        "Contexto de las fracciones; con n_truth pequeño las métricas tienen alta varianza.",
        "—",
    ),
}


def explain_lines():
    """Yield (metric, qué mide, interpretación, referencia) rows."""
    for key, (what, how, ref) in METRIC_INFO.items():
        yield key, what, how, ref
