#!/usr/bin/env python3
"""Genera graficos para la seccion de estado del arte."""

from pathlib import Path

import matplotlib.pyplot as plt


OUT_DIR = Path(__file__).resolve().parents[1] / "assets" / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def chart_distribucion_anio() -> None:
    years = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
    counts = [2, 5, 4, 9, 7, 13, 10]  # n=50

    plt.figure(figsize=(8, 4.8))
    bars = plt.bar([str(y) for y in years], counts, color="#2E86AB")
    plt.title("Distribucion de articulos seleccionados por ano (n=50)")
    plt.xlabel("Ano de publicacion")
    plt.ylabel("Numero de articulos")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    for bar, value in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.15,
            str(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(OUT_DIR / "estado_arte_distribucion_anio.png", dpi=220)
    plt.close()


def chart_composicion_corpus() -> None:
    labels = ["Core astronomico", "Reviews", "Metodos soporte"]
    values = [31, 8, 11]  # n=50
    colors = ["#1B998B", "#E84855", "#F9C80E"]

    plt.figure(figsize=(6.6, 4.8))
    bars = plt.bar(labels, values, color=colors)
    plt.title("Composicion del corpus final (n=50)")
    plt.ylabel("Numero de articulos")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            f"{value} ({value / 50:.0%})",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(OUT_DIR / "estado_arte_composicion_corpus.png", dpi=220)
    plt.close()


def chart_metricas() -> None:
    labels = ["PSNR", "SSIM", "Flux/SNR", "Shape e1/e2", "Comp./purity", "Tiempo inferencia"]
    values = [44, 42, 28, 22, 19, 14]  # n=50

    plt.figure(figsize=(8, 4.8))
    plt.barh(labels, values, color="#6C5CE7")
    plt.title("Frecuencia de metricas reportadas en el estado del arte (n=50)")
    plt.xlabel("Numero de articulos que reportan la metrica")
    plt.xlim(0, 50)
    plt.grid(axis="x", linestyle="--", alpha=0.3)
    for i, value in enumerate(values):
        plt.text(value + 0.4, i, str(value), va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "estado_arte_metricas_frecuencia.png", dpi=220)
    plt.close()


def main() -> None:
    chart_distribucion_anio()
    chart_composicion_corpus()
    chart_metricas()
    print(f"Graficos generados en: {OUT_DIR}")


if __name__ == "__main__":
    main()
