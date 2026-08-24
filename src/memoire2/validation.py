"""Découpages temporels sans fuite : walk-forward, et validation croisée purgée combinatoire (CPCV).

Pourquoi purger : la cible de la date de formation t est le rendement du mois [t, t+1]. Une observation
d'entraînement à t-1 a une cible qui chevauche [t-1, t] : elle ne recouvre pas le mois de test [t, t+1],
mais les caractéristiques (momentum, volatilité) se recouvrent sur 12 mois. On retire donc de
l'entraînement un « embargo » de mois de part et d'autre de chaque bloc de test, ce qui coupe tout
chevauchement d'information entre entraînement et test (López de Prado, 2018).

La CPCV découpe l'axe du temps en ``n_groups`` blocs contigus et prend toutes les combinaisons de
``k_test`` blocs comme test : chaque configuration est ainsi jugée sur de nombreux chemins hors
échantillon, ce qui alimente la probabilité de suroptimisation (PBO) au lieu d'un unique backtest.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


def contiguous_groups(dates: pd.DatetimeIndex, n_groups: int) -> pd.Series:
    """Attribue chaque date (unique, triée) à un bloc contigu 0..n_groups-1, de tailles quasi égales."""
    unique = pd.DatetimeIndex(sorted(set(dates)))
    labels = np.floor(np.arange(len(unique)) * n_groups / len(unique)).astype(int)
    return pd.Series(labels, index=unique)


def purge_train_dates(train_dates: pd.DatetimeIndex, test_dates: pd.DatetimeIndex,
                      embargo_months: int) -> pd.DatetimeIndex:
    """Retire de l'entraînement les mois à moins de ``embargo_months`` d'un mois de test."""
    if embargo_months <= 0:
        return train_dates
    test = pd.DatetimeIndex(sorted(set(test_dates)))
    keep = []
    for d in train_dates:
        gap = np.min(np.abs((test - d).days))
        if gap > embargo_months * 31:
            keep.append(d)
    return pd.DatetimeIndex(keep)


def cpcv_splits(dates: pd.DatetimeIndex, n_groups: int = 8, k_test: int = 2,
                embargo_months: int = 1) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Liste de (dates d'entraînement purgées, dates de test) pour chaque combinaison de blocs de test."""
    groups = contiguous_groups(dates, n_groups)
    splits = []
    for combo in combinations(range(n_groups), k_test):
        test_dates = groups.index[groups.isin(combo)]
        train_dates = groups.index[~groups.isin(combo)]
        splits.append((purge_train_dates(train_dates, test_dates, embargo_months), test_dates))
    return splits


def walk_forward_splits(dates: pd.DatetimeIndex, first_test: str, step_months: int = 12,
                        ) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Fenêtre croissante classique : entraînement sur tout le passé, test sur le bloc suivant."""
    unique = pd.DatetimeIndex(sorted(set(dates)))
    start = pd.Timestamp(first_test)
    splits = []
    t = start
    while t <= unique.max():
        t_end = t + pd.DateOffset(months=step_months)
        test = unique[(unique >= t) & (unique < t_end)]
        train = unique[unique < t]
        if len(test) and len(train) >= 24:
            splits.append((train, test))
        t = t_end
    return splits
