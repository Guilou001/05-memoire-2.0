"""Métriques : R² hors échantillon (convention Gu-Kelly-Xiu), Sharpe, t-stat Newey-West, Sharpe déflaté, PBO.

Le Sharpe déflaté (Bailey et López de Prado, 2014) répond à « ce Sharpe survivrait-il au nombre d'essais
tentés ? » : il compare le Sharpe observé au Sharpe maximal attendu sous H0 après ``n_trials`` essais,
en tenant compte de l'asymétrie et de l'aplatissement des rendements. La probabilité de suroptimisation
(PBO, méthode CSCV de Bailey et al., 2017) mesure la part des découpages où la meilleure configuration
en échantillon se retrouve sous la médiane hors échantillon.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def r2_oos_gkx(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² hors échantillon contre la prévision nulle (Gu, Kelly et Xiu, 2020) : 1 - SSE / somme(y²)."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / np.sum(y_true**2))


def sharpe_monthly(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) < 2 or r.std() == 0:
        return float("nan")
    return float(r.mean() / r.std() * np.sqrt(12))


def cagr_monthly(returns: pd.Series) -> float:
    r = returns.dropna()
    total = float((1 + r).prod())
    if total <= 0:
        return float("nan")
    return total ** (12 / len(r)) - 1


def max_drawdown(returns: pd.Series) -> float:
    wealth = (1 + returns.dropna()).cumprod()
    return float((wealth / wealth.cummax() - 1).min())


def newey_west_tstat(returns: pd.Series, lags: int = 6) -> float:
    """t-stat de la moyenne avec erreurs types robustes à l'autocorrélation (Newey-West)."""
    import statsmodels.api as sm

    r = returns.dropna().values
    model = sm.OLS(r, np.ones_like(r)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(model.tvalues[0])


def deflated_sharpe(observed_sr_annual: float, n_trials: int, n_obs: int,
                    skew: float, kurt: float, sr_variance_across_trials: float | None = None) -> float:
    """Probabilité que le Sharpe observé dépasse le meilleur Sharpe attendu sous H0 après n_trials essais.

    ``observed_sr_annual`` est annualisé (racine de 12) ; le calcul se fait en Sharpe par période.
    ``kurt`` est l'aplatissement NON centré (3 pour une loi normale). Retourne une probabilité dans [0, 1] ;
    au-dessus de 0,95, le Sharpe survit à la correction du nombre d'essais.
    """
    sr = observed_sr_annual / np.sqrt(12)
    var_trials = sr_variance_across_trials if sr_variance_across_trials is not None else sr**2 / max(n_obs, 2)
    emc = 0.5772156649015329
    z_max = ((1 - emc) * stats.norm.ppf(1 - 1.0 / n_trials)
             + emc * stats.norm.ppf(1 - 1.0 / (n_trials * np.e)))
    sr0 = float(np.sqrt(max(var_trials, 1e-12)) * z_max)
    denom = np.sqrt(max(1 - skew * sr + (kurt - 1) / 4.0 * sr**2, 1e-12) / max(n_obs - 1, 1))
    return float(stats.norm.cdf((sr - sr0) / denom))


def pbo_cscv(performance: pd.DataFrame) -> float:
    """Probabilité de suroptimisation par CSCV : ``performance`` = configurations x découpages (Sharpe).

    Pour chaque partition des découpages en deux moitiés (en échantillon / hors échantillon), on retient la
    meilleure configuration de la moitié IS et on regarde son rang dans la moitié OOS ; PBO = part des
    partitions où ce rang est sous la médiane. Implémentation sur les combinaisons de moitiés (jusqu'à 126
    partitions pour 9 découpages ou plus, plafonné pour rester rapide).
    """
    from itertools import combinations

    cols = list(performance.columns)
    n = len(cols)
    half = n // 2
    if half < 1 or len(performance) < 2:
        return float("nan")
    below_median = 0
    partitions = list(combinations(range(n), half))[:252]
    for insample in partitions:
        oos = [i for i in range(n) if i not in insample]
        perf_is = performance.iloc[:, list(insample)].mean(axis=1)
        perf_oos = performance.iloc[:, oos].mean(axis=1)
        best = perf_is.idxmax()
        rank = (perf_oos < perf_oos.loc[best]).mean()   # part des configs battues par la retenue
        if rank < 0.5:
            below_median += 1
    return float(below_median / len(partitions))
