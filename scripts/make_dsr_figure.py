"""Figure : Sharpe déflaté par modèle, Canada et États-Unis côte à côte, contre le seuil 0,95.

Lit la colonne ``dsr`` de ``results/tables/walk_forward_canada.csv`` et ``walk_forward_usa.csv`` (aucun
chiffre retapé) et trace des barres alignées par modèle, palette Okabe-Ito, seuil 0,95 en pointillé.
Sortie : ``results/figures/dsr_par_modele.png`` (et .pdf), 200 DPI.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from memoire2.report import NOMS_FR, OKABE_ITO, _style  # noqa: E402
from memoire2.runner import RESULTS  # noqa: E402


def main() -> int:
    _style()
    dsr = {}
    for country in ("canada", "usa"):
        table = pd.read_csv(RESULTS / "tables" / f"walk_forward_{country}.csv", index_col=0)
        dsr[country] = table["dsr"].dropna()    # les repères sans apprentissage n'ont pas de DSR
    models = list(dsr["canada"].index)
    x = np.arange(len(models))
    largeur = 0.38

    fig, ax = plt.subplots()
    ax.bar(x - largeur / 2, dsr["canada"].reindex(models), largeur, label="Canada", color=OKABE_ITO[0])
    ax.bar(x + largeur / 2, dsr["usa"].reindex(models), largeur, label="États-Unis", color=OKABE_ITO[1])
    ax.axhline(0.95, ls="--", lw=1.0, color="#000000")
    ax.text(len(models) - 0.5, 0.95, "seuil 0,95", ha="right", va="bottom", fontsize=8)
    # les valeurs sont si proches de zéro que les barres se voient à peine : on les écrit au-dessus
    for decalage, country in ((-largeur / 2, "canada"), (largeur / 2, "usa")):
        for xi, v in zip(x, dsr[country].reindex(models), strict=True):
            ax.annotate(f"{v:.2f}".replace(".", ","), (xi + decalage, max(v, 0.0)),
                        ha="center", va="bottom", fontsize=6.5)
    from matplotlib.ticker import FuncFormatter

    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ",")))
    ax.set_xticks(x)
    ax.set_xticklabels([NOMS_FR.get(m, m) for m in models], rotation=25, ha="right")
    ax.set_xlabel("Modèle (réglages gelés sur 2004-2007)")
    ax.set_ylabel("Sharpe déflaté (probabilité)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Sharpe déflaté par modèle, walk-forward 2008-2024, net de 10 pb")
    ax.legend()

    out = RESULTS / "figures" / "dsr_par_modele.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
