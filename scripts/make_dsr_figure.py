"""Figure : le Sharpe observé de chaque modèle contre le seuil du Sharpe déflaté, Canada et États-Unis.

Le Sharpe déflaté est une probabilité qui, ici, vaut au plus 0,007 : la porter sur un axe de 0 à 1
donnait une figure vide, où huit paires de barres invisibles portaient toutes l'étiquette « 0,00 ».
La quantité qui varie et qui explique ce zéro est ailleurs : c'est l'écart entre le Sharpe réalisé
et le seuil SR0, le meilleur Sharpe qu'on attend sous H0 après le nombre d'essais réellement menés
(Bailey et López de Prado, 2014). La figure trace donc cet écart, modèle par modèle, et écrit la
probabilité en marge.

Tout est lu dans ``results/tables/walk_forward_{pays}.csv`` (colonnes ``sharpe_net``, ``dsr``,
``sharpe_seuil_dsr``, ``n_essais_dsr``) : aucun chiffre n'est retapé, le titre lui-même est déduit
des données. Sortie : ``results/figures/dsr_par_modele.png`` (et .pdf), 200 DPI.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from memoire2.report import NOMS_FR, OKABE_ITO, _style  # noqa: E402
from memoire2.runner import FEE, RESULTS  # noqa: E402

PAYS = {"canada": "Canada", "usa": "États-Unis"}


def _virgule(v: float, _pos: int = 0) -> str:
    return f"{v:.1f}".replace(".", ",")


def main() -> int:
    _style()
    tables = {}
    for pays in PAYS:
        t = pd.read_csv(RESULTS / "tables" / f"walk_forward_{pays}.csv", index_col=0)
        tables[pays] = t[t["dsr"].notna()]      # les deux repères sans apprentissage n'ont pas de DSR

    modeles = list(tables["canada"].index)
    seuils = {p: float(t["sharpe_seuil_dsr"].unique()[0]) for p, t in tables.items()}
    seuils_h = {p: float(t["sharpe_seuil_horizon_egal"].unique()[0]) for p, t in tables.items()}
    essais = {p: int(t["n_essais_dsr"].unique()[0]) for p, t in tables.items()}
    y = np.arange(len(modeles))[::-1]

    # les bornes viennent des données : le seuil doit tenir dans le cadre, et les deux colonnes de
    # probabilités s'écrivent dans une marge réservée à droite
    bas = min(float(t["sharpe_net"].min()) for t in tables.values())
    haut = max(max(seuils.values()), max(float(t["sharpe_net"].max()) for t in tables.values()))
    x_separation = haut + 0.12
    bornes = (bas - 0.12, haut + 1.30)
    x_mesure, x_modele = bornes[1] - 0.64, bornes[1] - 0.04

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True, sharex=True)
    for ax, (pays, nom) in zip(axes, PAYS.items(), strict=True):
        t = tables[pays].reindex(modeles)
        sharpe, seuil = t["sharpe_net"].to_numpy(), seuils[pays]
        dsr, dsr_h = t["dsr"].to_numpy(), t["dsr_horizon_egal"].to_numpy()
        # la colonne de droite porte des probabilités, pas des Sharpe : elle est sortie du cadre
        # de tracé par un fond gris, pour qu'aucun lecteur ne la lise sur l'axe des abscisses
        ax.axvspan(x_separation, bornes[1], color="#f2f2f2", zorder=0)
        ax.axvline(0.0, lw=0.8, color="#999999")
        ax.hlines(y, 0.0, sharpe, color=OKABE_ITO[0], lw=2.4, alpha=0.85)
        ax.plot(sharpe, y, "o", ms=5.5, color=OKABE_ITO[0])
        ax.axvline(seuil, ls="--", lw=1.2, color=OKABE_ITO[1])
        ax.text(seuil - 0.05, (len(modeles) - 1) / 2, f"seuil SR0 = {seuil:.2f}".replace(".", ","),
                color=OKABE_ITO[1], fontsize=8, ha="center", va="center", rotation=90)
        # les essais de la grille sont mesurés sur un horizon plus court que le Sharpe jugé, ce qui
        # gonfle le seuil : le seuil remis au même horizon est tracé à côté, statut MODÉLISÉ
        ax.axvline(seuils_h[pays], ls=":", lw=1.2, color="#666666")
        ax.text(seuils_h[pays] - 0.05, (len(modeles) - 1) / 2,
                f"à horizon égal = {seuils_h[pays]:.2f}".replace(".", ","),
                color="#666666", fontsize=8, ha="center", va="center", rotation=90)
        for yi, d, dh in zip(y, dsr, dsr_h, strict=True):
            for x, v in ((x_mesure, d), (x_modele, dh)):
                texte = "< 0,001" if v < 0.001 else f"{v:.3f}".replace(".", ",")
                ax.annotate(texte, (x, yi), fontsize=7.5, ha="right", va="center", color="#444444")
        ax.set_title(f"{nom} ({essais[pays]} essais)", fontsize=10.5)
        ax.set_xlabel("Ratio de Sharpe annualisé")
        ax.xaxis.set_major_formatter(FuncFormatter(_virgule))
        # les graduations s'arrêtent au seuil : la colonne grise ne porte pas de Sharpe
        ax.set_xticks(np.arange(np.ceil(bornes[0] * 2) / 2, np.floor(haut * 2) / 2 + 1e-9, 0.5))
        ax.set_xlim(*bornes)

    axes[0].set_yticks(y)
    # un Sharpe de seize ans dont douze mois seulement portent une position mesure surtout la durée
    # de l'inaction : le compte des mois actifs est écrit à côté du nom du modèle
    etiquettes = []
    for m in modeles:
        ligne = tables["canada"].loc[m]
        actifs, total = int(ligne["mois_actifs"]), int(ligne["mois_total"])
        nom_fr = NOMS_FR.get(m, m)
        etiquettes.append(nom_fr if actifs == total else f"{nom_fr}\n({actifs} mois actifs sur {total})")
    axes[0].set_yticklabels(etiquettes, fontsize=8.5)
    for ax in axes:
        # ax.text et non ax.annotate : au-dessus de la dernière ligne, une annotation est rognée
        ax.text(x_mesure, len(modeles) - 0.55, "SD\nmesuré", fontsize=7.5, ha="right",
                va="center", color="#444444", fontweight="bold")
        ax.text(x_modele, len(modeles) - 0.55, "SD\nhorizon égal", fontsize=7.5, ha="right",
                va="center", color="#444444", fontweight="bold")

    depasse = [m for p, t in tables.items() for m in modeles
               if float(t.reindex(modeles).loc[m, "sharpe_net"]) >= seuils[p]]
    dsr_max = max(float(t["dsr"].max()) for t in tables.values())
    dsr_h_max = max(float(t["dsr_horizon_egal"].max()) for t in tables.values())
    if depasse:
        titre = (f"{len(depasse)} modèles franchissent le seuil du Sharpe déflaté "
                 f"(Sharpe déflaté maximal {dsr_max:.3f})".replace(".", ","))
    else:
        titre = (f"Aucun des {len(modeles)} modèles n'atteint le Sharpe attendu du meilleur essai sous H0 : "
                 f"le Sharpe déflaté (SD) plafonne à {dsr_max:.3f}, "
                 f"{dsr_h_max:.3f} à horizon égal".replace(".", ","))
    fig.suptitle(titre, fontsize=11.5)
    fig.text(0.5, -0.06, f"Walk-forward 2008-2024, réglages gelés sur la validation 2004-2007, rendements "
                         f"nets de {FEE * 1e4:.0f} points de base par unité de rotation. Seuil SR0 : meilleur "
                         f"Sharpe attendu sous H0 après le nombre d'essais menés (Bailey et López de Prado, "
                         f"2014).\nLes essais de la grille sont jugés sur 48 mois de validation et les familles "
                         f"sur 196 mois de test : la colonne « horizon égal » remet la dispersion entre essais "
                         f"à l'horizon jugé (modélisé).",
             ha="center", fontsize=7.5, color="#444444")

    out = RESULTS / "figures" / "dsr_par_modele.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
