"""Du panel de prédictions au portefeuille long short mensuel, net de coûts.

À chaque date de formation, les titres sont classés par prédiction : achat du quintile du haut, vente à
découvert du quintile du bas (10 titres de chaque côté pour un univers de 50), à poids égaux. La rotation
mensuelle (somme des variations absolues de poids, les deux côtés) est facturée à ``fee`` par unité
échangée (10 points de base par défaut, aller simple). Les rendements sont mensuels : la précision
quotidienne de la dérive intra-mois, testée dans la v1, ne change pas les conclusions et compliquerait
la validation croisée.

Un garde-fou hérité de la v1 : si toutes les prédictions d'une date sont égales (modèle constant), la date
est neutralisée (aucune position) au lieu de sélectionner les titres par ordre alphabétique.
"""

from __future__ import annotations

import pandas as pd


def quantile_weights(pred_row: pd.Series, quantile: float = 0.2) -> pd.Series:
    """Poids long short d'une date : +1/n sur le quantile du haut, -1/n sur celui du bas, 0 sinon."""
    p = pred_row.dropna()
    if p.empty or p.nunique() == 1:      # prédictions toutes égales : pas de classement, pas de position
        return pd.Series(0.0, index=pred_row.index)
    n = max(int(round(len(p) * quantile)), 1)
    ranks = p.rank(method="average")
    top = ranks.nlargest(n).index
    bottom = ranks.nsmallest(n).index
    w = pd.Series(0.0, index=pred_row.index)
    w.loc[top] = 1.0 / n
    w.loc[bottom] = -1.0 / n
    return w


def long_short_returns(predictions: pd.DataFrame, realized: pd.DataFrame, quantile: float = 0.2,
                       fee: float = 0.001) -> pd.DataFrame:
    """Rendements mensuels bruts et nets ; ``predictions`` et ``realized`` : dates x titres alignés.

    ``realized`` à la date t = rendement du mois [t, t+1] (la cible du panel). Rendement brut de la date t
    = somme des poids x rendements réalisés ; coût = fee x rotation depuis les poids de la date précédente.
    """
    predictions, realized = predictions.align(realized, join="inner")
    rows = []
    prev_w = pd.Series(0.0, index=predictions.columns)
    for date, pred_row in predictions.iterrows():
        w = quantile_weights(pred_row, quantile)
        gross = float((w * realized.loc[date]).sum())
        turnover = float((w - prev_w).abs().sum())
        rows.append({"date": date, "gross": gross, "net": gross - fee * turnover,
                     "turnover": turnover, "n_long": int((w > 0).sum()), "n_short": int((w < 0).sum())})
        # les poids dérivent avec les rendements du mois avant le rééquilibrage suivant
        drifted = w * (1 + realized.loc[date].fillna(0.0))
        gross_expo = drifted.abs().sum()
        prev_w = drifted * (w.abs().sum() / gross_expo) if gross_expo > 0 else drifted
    return pd.DataFrame(rows).set_index("date")


def momentum_vol_benchmark(panel: pd.DataFrame, mom_raw: pd.DataFrame, vol_raw: pd.DataFrame,
                           quantile: float = 0.2, fee: float = 0.001) -> pd.DataFrame:
    """Le contrôle de Nagel (2025) : classer par momentum BRUT / volatilité BRUTE, sans apprentissage.

    Si un modèle complexe ne bat pas ce classement mécanique, son gain n'est pas de la structure apprise.
    Le score se calcule sur ``mom_raw`` et ``vol_raw`` (dates x titres), les valeurs recalculées depuis les
    prix, et jamais sur les colonnes du panel : celles-ci sont des rangs transversaux dans [-0,5 ; 0,5],
    et le ratio de deux rangs n'est pas le signal (division par des rangs proches de zéro, corrélation de
    rang de 0,03 à 0,07 avec le vrai ratio, mesurée lors de l'audit du 2026-08-28). Le panel ne sert ici
    qu'à restreindre le contrôle aux mêmes couples (date, titre) que ceux vus par les modèles.
    """
    realized = panel["target"].unstack("ticker")
    available = panel["mom_12_2"].unstack("ticker").notna()
    score = mom_raw / vol_raw.where(vol_raw > 0)
    score = score.reindex(index=realized.index, columns=realized.columns).where(available)
    return long_short_returns(score, realized, quantile, fee)
