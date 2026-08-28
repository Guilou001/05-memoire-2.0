"""Test placebo : mêmes variables, cibles mélangées ; un R² HÉ qui survit ne mesure pas le classement.

Le R² hors échantillon à la Gu-Kelly-Xiu se calcule contre la prévision zéro : un modèle très régularisé
qui prédit simplement « les rendements sont en moyenne positifs » marque des points sans classer les
titres. Pour le montrer, ce script permute aléatoirement la colonne cible sur tout le panel (graine
fixée) : plus aucun lien entre variables et rendements. Il rejoue ensuite le même protocole (sélection
des réglages sur 2004-2007, gel, walk-forward 2008-2024) sur les cibles mélangées et compare le R² HÉ
obtenu au R² réel du run principal, lu dans ``results/predictions_<pays>.parquet``.

Sortie : ``results/tables/placebo.csv`` (une ligne par pays et modèle testé).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from memoire2 import metrics as mx
from memoire2.panel import build_panel
from memoire2.runner import RAW, RESULTS, select_hyperparameters, walk_forward_predictions

COUNTRY, MODEL, SEED = "canada", "extra_trees", 123


def main() -> int:
    panel = build_panel(COUNTRY, RAW)
    frozen, _ = select_hyperparameters(panel, verbose=False)   # même sélection que le run principal
    frozen = {MODEL: frozen[MODEL]}

    # R² réel : recalculé depuis les prédictions sauvegardées du run principal (même source que le README)
    joined = pd.read_parquet(RESULTS / f"predictions_{COUNTRY}.parquet")
    r2_reel = mx.r2_oos_gkx(joined["target"].values, joined[MODEL].values)

    # placebo : cibles permutées sur tout le panel, puis le même walk-forward gelé
    shuffled = panel.copy()
    shuffled["target"] = np.random.default_rng(SEED).permutation(shuffled["target"].values)
    preds = walk_forward_predictions(shuffled, frozen)
    r2_placebo = mx.r2_oos_gkx(shuffled.loc[preds.index, "target"].values, preds[MODEL].values)

    out = pd.DataFrame([{
        "country": COUNTRY, "model": MODEL, "seed": SEED,
        "r2_oos_reel": r2_reel, "r2_oos_placebo": r2_placebo,
        "source_reel": f"predictions_{COUNTRY}.parquet",
        "protocole": "cibles permutées sur tout le panel, réglages regelés sur 2004-2007, walk-forward 2008-2024",
    }])
    (RESULTS / "tables").mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS / "tables" / "placebo.csv", index=False)
    print(out.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
