"""Draw the README figure from published results, without rerunning the study.

Run from the repository with ``uv run python scripts/figure_presentation.py``.
The input tables remain the numerical source of truth.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE, GREEN, GREY = "#176B96", "#C56628", "#14816D", "#718096"
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 15,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#CBD5E0",
        "text.color": "#172B3A",
        "axes.labelcolor": "#172B3A",
        "xtick.color": "#425466",
        "ytick.color": "#425466",
        "axes.axisbelow": True,
    }
)


def number(value, decimals=2):
    return f"{value:,.{decimals}f}".replace(",", " ").replace(".", ",")


def finish(fig, axes, title, note, path="results/figures/presentation.png"):
    for ax in np.asarray(axes, dtype=object).ravel():
        ax.grid(axis="x", color="#EDF0F3", linewidth=0.8)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}".replace(".", ",")))
    fig.suptitle(title, x=0.02, ha="left", fontweight="bold", fontsize=16)
    fig.text(0.02, 0.015, note, ha="left", va="bottom", fontsize=9, color="#526575")
    fig.tight_layout(rect=(0, 0.075, 1, 0.91))
    destination = ROOT / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    d = pd.read_csv(ROOT / "results/tables/walk_forward_canada.csv", index_col=0)
    names = {
        "ridge": "Régression régularisée",
        "elastic_net": "Régression avec sélection",
        "random_forest": "Forêt d'arbres",
        "extra_trees": "Arbres plus aléatoires",
        "hist_gb": "Arbres corrigés successivement",
        "mlp": "Réseau de neurones",
        "rff_ridge": "Régression non linéaire",
        "ensemble": "Combinaison des modèles",
        "equipondere_long_only": "Répartition égale en actions",
    }
    values = d.loc[list(names), "sharpe_net"].sort_values()
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.barh(
        np.arange(len(values)),
        values,
        color=[BLUE if k == "equipondere_long_only" else GREY for k in values.index],
        height=0.6,
    )
    ax.axvline(0, color="#425466", linewidth=0.8)
    ax.set_yticks(np.arange(len(values)), [names[k] for k in values.index])
    for y, v in enumerate(values):
        ax.text(v + (0.025 if v >= 0 else -0.025), y, number(v), va="center", ha="left" if v >= 0 else "right")
    ax.set_xlim(values.min() - 0.18, values.max() + 0.16)
    ax.set_xlabel("Sharpe annualisé après coûts")
    finish(
        fig,
        [ax],
        "Les huit modèles restent derrière la répartition égale",
        "Canada · 2008 à 2024 · portefeuille de sociétés survivantes · coûts de 0,10 % par montant négocié\n"
        "Les modèles achètent et vendent à découvert. La référence détient les actions. Leurs expositions diffèrent.",
    )


if __name__ == "__main__":
    main()
