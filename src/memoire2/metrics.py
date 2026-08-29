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


def expected_max_sharpe(n_trials: int, sr_variance_across_trials: float) -> float:
    """Le Sharpe PAR PÉRIODE qu'on attend du meilleur de ``n_trials`` essais sous H0 (seuil SR0).

    C'est l'ingrédient central du Sharpe déflaté : la loi du maximum de ``n_trials`` tirages
    gaussiens de variance ``sr_variance_across_trials``, approchée par Bailey et López de Prado
    (2014) à partir de la constante d'Euler-Mascheroni. Le seuil ne dépend que du nombre d'essais
    et de la dispersion des Sharpe ENTRE ces essais : il ignore le Sharpe observé, ce qui en fait
    la barre à franchir plutôt qu'une mesure de la stratégie.
    """
    if sr_variance_across_trials is None or not np.isfinite(sr_variance_across_trials) \
            or sr_variance_across_trials <= 0:
        raise ValueError(
            "sr_variance_across_trials doit être la variance, strictement positive, des ratios de Sharpe "
            "mesurés sur les essais réellement menés (par période, pas annualisés)")
    if n_trials < 1:
        raise ValueError("n_trials doit valoir au moins 1")
    if n_trials == 1:
        # ppf(1 - 1/1) = ppf(0) = -inf : sans rien à maximiser, le seuil sous H0 est nul, et le
        # Sharpe déflaté se ramène au Sharpe probabiliste contre zéro. La formule non gardée
        # renvoyait -inf, donc une probabilité de 1,0 pour n'importe quel Sharpe, même négatif.
        return 0.0
    emc = 0.5772156649015329
    z_max = ((1 - emc) * stats.norm.ppf(1 - 1.0 / n_trials)
             + emc * stats.norm.ppf(1 - 1.0 / (n_trials * np.e)))
    return float(np.sqrt(sr_variance_across_trials) * z_max)


def variance_essais_a_l_horizon(var_essais: float, n_obs_essais: int, n_obs_cible: int) -> float:
    """Ramène la variance des Sharpe ENTRE essais de l'horizon des essais à celui du Sharpe jugé.

    Bailey et López de Prado (2014) supposent des essais mesurés sur la même longueur que le Sharpe
    candidat. Ce n'est pas le cas ici : les réglages de la grille sont jugés sur les 48 mois de
    validation, les familles sur les 196 mois de test. Comme la variance d'échantillonnage d'un
    Sharpe par période vaut environ 1/T, les essais courts gonflent la dispersion, donc le seuil
    SR0, donc écrasent le Sharpe déflaté : le biais joue en faveur du verdict « rien ne survit »,
    ce qui interdit de le laisser silencieux.

    On sépare donc la dispersion VRAIE entre essais de l'erreur d'estimation, var_observée =
    var_vraie + 1/T_essais, avant de remettre le tout à l'horizon jugé :
    var_cible = max(var_observée - 1/T_essais, 0) + 1/T_cible.

    Statut : MODÉLISÉ. L'hypothèse est que les essais partagent une même dispersion vraie et que
    l'erreur d'estimation d'un Sharpe par période vaut 1/T (approximation gaussienne, Lo 2002).
    Le calcul exact demanderait de rejouer chaque réglage de la grille sur la fenêtre de test.
    """
    if n_obs_essais < 2 or n_obs_cible < 2:
        raise ValueError("les deux horizons doivent compter au moins deux périodes")
    var_vraie = max(float(var_essais) - 1.0 / n_obs_essais, 0.0)
    return var_vraie + 1.0 / n_obs_cible


def deflated_sharpe(observed_sr_annual: float, n_trials: int, n_obs: int,
                    skew: float, kurt: float, sr_variance_across_trials: float) -> float:
    """Probabilité que le Sharpe observé dépasse le meilleur Sharpe attendu sous H0 après n_trials essais.

    ``observed_sr_annual`` est annualisé (racine de 12) ; le calcul se fait en Sharpe par période.
    ``kurt`` est l'aplatissement NON centré (3 pour une loi normale).
    ``sr_variance_across_trials`` est la variance, ENTRE les essais réellement menés, de leurs ratios de
    Sharpe PAR PÉRIODE (mensuels ici) : Bailey et López de Prado (2014) en font l'ingrédient du meilleur
    Sharpe attendu sous H0. Le paramètre est obligatoire : l'ancienne valeur par défaut sr²/n_obs
    remplaçait la dispersion entre essais par l'erreur type d'un seul essai et annulait la déflation.
    Retourne une probabilité dans [0, 1] ; au-dessus de 0,95, le Sharpe survit à la correction.
    """
    sr = observed_sr_annual / np.sqrt(12)
    sr0 = expected_max_sharpe(n_trials, sr_variance_across_trials)
    denom = np.sqrt(max(1 - skew * sr + (kurt - 1) / 4.0 * sr**2, 1e-12) / max(n_obs - 1, 1))
    return float(stats.norm.cdf((sr - sr0) / denom))


def niveau_nul_pbo(n_configs: int) -> float:
    """La PBO qu'on attend quand aucune configuration n'est meilleure qu'une autre.

    Sous H0, la configuration retenue en échantillon occupe un rang uniforme parmi les
    ``n_configs`` rangs hors échantillon, et son rang relatif vaut (j + 1) / (n + 1). La part des
    rangs qui tombent sous 0,5 n'est exactement 0,5 que si ``n_configs`` est pair : pour 7
    configurations, elle vaut 3/7 = 0,43. Comparer une PBO mesurée à 0,50 sans le dire fausse la
    lecture ; ce repère est donc calculé et publié à côté de la mesure.
    """
    if n_configs < 2:
        return float("nan")
    rangs = (np.arange(n_configs) + 1.0) / (n_configs + 1.0)
    return float(np.mean(rangs < 0.5))


def pbo_cscv(performance: pd.DataFrame, n_partitions: int = 500, seed: int = 123) -> float:
    """Probabilité de suroptimisation par CSCV : ``performance`` = configurations x blocs DISJOINTS.

    Pour chaque partition des blocs en deux moitiés égales (en échantillon / hors échantillon), on
    retient la meilleure configuration de la moitié IS et on regarde son rang relatif dans la moitié
    OOS ; PBO = part des partitions où ce rang tombe sous la médiane.

    Les colonnes doivent être DISJOINTES dans le temps. Les alimenter avec les 28 chemins CPCV, qui
    se chevauchent (chaque bloc est testé dans 7 chemins sur 28), met les mêmes mois des deux côtés
    de chaque partition et effondre la PBO vers zéro : mesuré à 0,11 sur du bruit pur contre 0,62
    avec des blocs disjoints (contre-vérification du 2026-08-29). ``cpcv_performance`` rend donc un
    Sharpe par bloc disjoint, et non un Sharpe par chemin.

    Le rang relatif suit Bailey et al. (2017) : (j + 1) / (n + 1) où j est le nombre de
    configurations strictement battues. Le repère sous H0 se lit avec ``niveau_nul_pbo``.

    Bailey et al. énumèrent toutes les partitions équilibrées ; on en tire ``n_partitions`` au
    hasard, une permutation par tirage, avec une graine fixée : chaque bloc a la même chance d'être
    de chaque côté et le résultat est reproductible.
    """
    n = performance.shape[1]
    half = n // 2
    n_configs = len(performance)
    if half < 1 or n_configs < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    below_median = 0
    for _ in range(n_partitions):
        perm = rng.permutation(n)
        perf_is = performance.iloc[:, perm[:half]].mean(axis=1)
        perf_oos = performance.iloc[:, perm[half:]].mean(axis=1)
        best = perf_is.idxmax()
        battues = int((perf_oos < perf_oos.loc[best]).sum())
        if (battues + 1.0) / (n_configs + 1.0) < 0.5:
            below_median += 1
    return float(below_median / n_partitions)
