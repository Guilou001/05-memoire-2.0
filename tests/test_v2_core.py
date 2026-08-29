"""Tests du cœur scientifique : alignement sans fuite, purge, portefeuille, métriques."""

import numpy as np
import pandas as pd
import pytest

from memoire2 import metrics as mx
from memoire2.data import month_start_prices, monthly_returns
from memoire2.panel import rank_normalize
from memoire2.portfolio import long_short_returns, momentum_vol_benchmark, quantile_weights
from memoire2.validation import contiguous_groups, cpcv_splits, purge_train_dates, walk_forward_splits


@pytest.fixture()
def daily_prices():
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2019-01-01", "2021-12-31")
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (len(idx), 6)), axis=0)),
                          index=idx, columns=list("ABCDEF"))
    return prices


# ------------------------------------------------------------------ alignement
def test_month_start_price_uses_only_past_prices(daily_prices):
    pm = month_start_prices(daily_prices)
    # le 1er février 2019 est un vendredi ; le prix au tampon 2019-02-01 doit être celui du 1er ou d'avant
    stamp = pd.Timestamp("2019-02-01")
    expected = daily_prices.loc[:stamp].iloc[-1]
    pd.testing.assert_series_equal(pm.loc[stamp], expected, check_names=False)


def test_return_stamp_covers_previous_month(daily_prices):
    pm = month_start_prices(daily_prices)
    r = monthly_returns(pm)
    stamp = pd.Timestamp("2019-03-01")
    manual = pm.loc[stamp] / pm.loc[pd.Timestamp("2019-02-01")] - 1
    pd.testing.assert_series_equal(r.loc[stamp], manual, check_names=False)


def test_target_would_be_next_month():
    # la cible à t est le rendement tamponné t+1 : vérifié par construction dans build_panel (shift(-1)) ;
    # ici on vérifie la propriété sur un cas minimal
    r = pd.DataFrame({"A": [0.1, 0.2, 0.3]},
                     index=pd.date_range("2020-01-01", periods=3, freq="MS"))
    target = r.shift(-1)
    assert target.loc["2020-01-01", "A"] == pytest.approx(0.2)


def test_rank_normalize_is_cross_sectional_and_bounded():
    df = pd.DataFrame([[1.0, 2.0, 3.0], [30.0, 20.0, 10.0]],
                      index=pd.date_range("2020-01-01", periods=2, freq="MS"), columns=list("XYZ"))
    z = rank_normalize(df)
    assert z.loc["2020-01-01"].tolist() == [-0.5, 0.0, 0.5]
    assert z.loc["2020-02-01"].tolist() == [0.5, 0.0, -0.5]


# ------------------------------------------------------------------ validation
def test_purge_removes_embargo_neighbourhood():
    dates = pd.date_range("2020-01-01", periods=24, freq="MS")
    test = dates[10:12]
    train = dates.difference(test)
    purged = purge_train_dates(train, test, embargo_months=2)
    assert dates[9] not in purged and dates[12] not in purged and dates[13] not in purged
    assert dates[0] in purged and dates[23] in purged


def test_cpcv_counts_and_disjointness():
    dates = pd.date_range("2008-01-01", periods=96, freq="MS")
    splits = cpcv_splits(dates, n_groups=6, k_test=2, embargo_months=1)
    assert len(splits) == 15                       # C(6, 2)
    for train, test in splits:
        assert len(set(train) & set(test)) == 0
        assert min(abs((pd.DatetimeIndex(sorted(set(test))) - d).days).min() for d in train) > 31


def test_walk_forward_trains_only_on_past():
    dates = pd.date_range("2000-01-01", periods=120, freq="MS")
    for train, test in walk_forward_splits(dates, first_test="2006-01-01", step_months=12):
        assert train.max() < test.min()


def test_groups_are_contiguous():
    dates = pd.date_range("2010-01-01", periods=40, freq="MS")
    g = contiguous_groups(dates, 4)
    assert (g.sort_index().diff().dropna() >= 0).all()


# ------------------------------------------------------------------ portefeuille
def test_quantile_weights_sides_and_sum():
    pred = pd.Series({"A": 3.0, "B": 2.0, "C": 1.0, "D": -1.0, "E": -2.0})
    w = quantile_weights(pred, quantile=0.2)
    assert w["A"] == pytest.approx(1.0) and w["E"] == pytest.approx(-1.0)
    assert w.sum() == pytest.approx(0.0) and w.abs().sum() == pytest.approx(2.0)


def test_constant_predictions_give_no_position():
    pred = pd.Series(1.0, index=list("ABCDE"))
    assert quantile_weights(pred).abs().sum() == 0.0


def test_momentum_vol_benchmark_ranks_on_raw_signal_not_on_panel_ranks():
    # cinq titres, un mois : A a le plus fort momentum brut et la plus faible volatilité brute, E l'inverse.
    # Le panel porte des rangs DÉLIBÉRÉMENT inversés : si le contrôle les utilisait, il serait short A.
    dates = pd.to_datetime(["2020-01-01"])
    tickers = list("ABCDE")
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    panel = pd.DataFrame({
        "mom_12_2": [-0.5, -0.25, 0.0, 0.25, 0.5],          # rangs inversés par rapport au signal brut
        "vol_12m": [0.5, 0.25, 0.0, -0.25, -0.5],
        "target": [0.04, 0.01, 0.0, -0.01, -0.03],
    }, index=idx)
    mom_raw = pd.DataFrame([[0.50, 0.20, 0.05, -0.10, -0.30]], index=dates, columns=tickers)
    vol_raw = pd.DataFrame([[0.05, 0.10, 0.15, 0.20, 0.40]], index=dates, columns=tickers)
    out = momentum_vol_benchmark(panel, mom_raw, vol_raw, quantile=0.2, fee=0.0)
    assert out.loc[dates[0], "n_long"] == 1 and out.loc[dates[0], "n_short"] == 1
    # long A (score brut 10, le plus fort), short E (score brut -0,75, le plus faible) : brut = rA - rE
    assert out.loc[dates[0], "gross"] == pytest.approx(0.04 - (-0.03))


def test_momentum_vol_benchmark_ignores_zero_vol_and_missing_panel_rows():
    dates = pd.to_datetime(["2020-01-01"])
    tickers = list("ABCDE")
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    panel = pd.DataFrame({
        "mom_12_2": [0.1, 0.2, 0.3, 0.4, float("nan")],      # E absent du panel à cette date
        "vol_12m": [0.1] * 5,
        "target": [0.01, 0.02, 0.03, 0.04, 0.05],
    }, index=idx)
    mom_raw = pd.DataFrame([[0.5, 0.2, 0.1, -0.2, 0.9]], index=dates, columns=tickers)
    vol_raw = pd.DataFrame([[0.0, 0.1, 0.1, 0.1, 0.1]], index=dates, columns=tickers)  # A : vol nulle
    out = momentum_vol_benchmark(panel, mom_raw, vol_raw, quantile=0.2, fee=0.0)
    # A (vol nulle) et E (hors panel) sont exclus du classement : long B (0,2/0,1), short D (-0,2/0,1)
    assert out.loc[dates[0], "gross"] == pytest.approx(0.02 - 0.04)


def test_costs_reduce_net_of_gross():
    idx = pd.date_range("2020-01-01", periods=6, freq="MS")
    rng = np.random.default_rng(3)
    preds = pd.DataFrame(rng.normal(size=(6, 10)), index=idx, columns=list("ABCDEFGHIJ"))
    realized = pd.DataFrame(rng.normal(0, 0.05, size=(6, 10)), index=idx, columns=list("ABCDEFGHIJ"))
    out = long_short_returns(preds, realized, quantile=0.2, fee=0.001)
    assert (out["net"] <= out["gross"] + 1e-12).all()
    assert out["turnover"].iloc[0] == pytest.approx(2.0)   # mise en place : 100 % long + 100 % short


# ------------------------------------------------------------------ métriques
def test_r2_oos_gkx_zero_forecast_is_zero():
    y = np.array([0.1, -0.2, 0.05])
    assert mx.r2_oos_gkx(y, np.zeros(3)) == pytest.approx(0.0)


def test_sharpe_and_nw_tstat_signs():
    r = pd.Series(np.full(120, 0.01))
    assert mx.sharpe_monthly(r + np.random.default_rng(0).normal(0, 0.001, 120)) > 3


def test_deflated_sharpe_penalizes_trials():
    high = mx.deflated_sharpe(1.0, n_trials=2, n_obs=192, skew=0.0, kurt=3.0,
                              sr_variance_across_trials=0.01)
    low = mx.deflated_sharpe(1.0, n_trials=500, n_obs=192, skew=0.0, kurt=3.0,
                             sr_variance_across_trials=0.01)
    assert high > low


def test_deflated_sharpe_penalizes_dispersion_across_trials():
    tight = mx.deflated_sharpe(1.0, n_trials=33, n_obs=192, skew=0.0, kurt=3.0,
                               sr_variance_across_trials=0.001)
    wide = mx.deflated_sharpe(1.0, n_trials=33, n_obs=192, skew=0.0, kurt=3.0,
                              sr_variance_across_trials=0.05)
    assert tight > wide


def test_expected_max_sharpe_is_the_barrier_of_the_deflated_sharpe():
    # le seuil SR0 ne dépend pas du Sharpe observé, il monte avec le nombre d'essais et avec leur
    # dispersion, et un Sharpe pile au seuil laisse une probabilité de 0,5 (l'argument est nul)
    sr0 = mx.expected_max_sharpe(33, 0.01561)
    assert sr0 == pytest.approx(mx.expected_max_sharpe(33, 0.01561))
    assert mx.expected_max_sharpe(500, 0.01561) > sr0 > mx.expected_max_sharpe(2, 0.01561)
    assert mx.expected_max_sharpe(33, 0.05) > sr0
    au_seuil = mx.deflated_sharpe(sr0 * np.sqrt(12), n_trials=33, n_obs=192, skew=0.0, kurt=3.0,
                                  sr_variance_across_trials=0.01561)
    assert au_seuil == pytest.approx(0.5, abs=1e-9)


def test_deflated_sharpe_refuses_missing_trial_variance():
    # l'ancien défaut silencieux sr²/n_obs annulait la déflation : la variance mesurée est obligatoire
    with pytest.raises(TypeError):
        mx.deflated_sharpe(1.0, n_trials=33, n_obs=192, skew=0.0, kurt=3.0)
    with pytest.raises(ValueError):
        mx.deflated_sharpe(1.0, n_trials=33, n_obs=192, skew=0.0, kurt=3.0,
                           sr_variance_across_trials=float("nan"))
    with pytest.raises(ValueError):
        mx.deflated_sharpe(1.0, n_trials=33, n_obs=192, skew=0.0, kurt=3.0,
                           sr_variance_across_trials=0.0)


def test_expected_max_sharpe_un_seul_essai_ne_deflate_rien():
    # ppf(1 - 1/1) valait -inf : le seuil partait à -inf et le Sharpe déflaté rendait 1,0 pour
    # n'importe quel Sharpe observé, même franchement négatif
    assert mx.expected_max_sharpe(1, 0.01) == 0.0
    assert mx.deflated_sharpe(-2.0, n_trials=1, n_obs=100, skew=0.0, kurt=3.0,
                              sr_variance_across_trials=0.01) < 0.01
    with pytest.raises(ValueError):
        mx.expected_max_sharpe(0, 0.01)


def test_variance_essais_a_l_horizon_retire_le_bruit_d_estimation():
    # une dispersion observée entièrement imputable au bruit d'un horizon court (var <= 1/T) doit se
    # ramener au seul bruit de l'horizon cible : plus rien de « vrai » ne subsiste
    assert mx.variance_essais_a_l_horizon(1 / 48, 48, 196) == pytest.approx(1 / 196)
    assert mx.variance_essais_a_l_horizon(0.005, 48, 196) == pytest.approx(1 / 196)
    # une dispersion vraie survit et s'ajoute au bruit de l'horizon cible
    assert mx.variance_essais_a_l_horizon(1 / 48 + 0.01, 48, 196) == pytest.approx(0.01 + 1 / 196)
    # le seuil qui en découle est plus BAS que celui calculé sur les essais courts : c'est le sens
    # du biais que la correction annule
    assert mx.expected_max_sharpe(33, mx.variance_essais_a_l_horizon(0.01561, 48, 196)) \
        < mx.expected_max_sharpe(33, 0.01561)


def test_niveau_nul_pbo_depend_de_la_parite_du_nombre_de_configurations():
    assert mx.niveau_nul_pbo(8) == pytest.approx(0.5)
    assert mx.niveau_nul_pbo(7) == pytest.approx(3 / 7)


def test_pbo_colonnes_chevauchantes_ecrase_la_mesure():
    # reproduit le défaut trouvé à l'audit du 2026-08-29 : alimenter la CSCV avec des chemins qui
    # partagent leurs blocs met les mêmes mois des deux côtés de chaque partition, et la PBO
    # s'effondre alors que rien ne distingue les configurations
    from itertools import combinations

    disjoints, chevauchants = [], []
    for s in range(20):
        blocs = pd.DataFrame(np.random.default_rng(s).normal(size=(7, 8)))
        chemins = pd.DataFrame({i: blocs[list(c)].mean(axis=1)
                                for i, c in enumerate(combinations(range(8), 2))})
        disjoints.append(mx.pbo_cscv(blocs))
        chevauchants.append(mx.pbo_cscv(chemins))
    assert float(np.mean(chevauchants)) < float(np.mean(disjoints)) - 0.2


def test_pbo_unbiased_around_half_for_pure_noise():
    # avec peu de chemins, la PBO d'une seule matrice de bruit est très variable ; c'est sa MOYENNE
    # sur des tirages indépendants qui doit tourner autour de 0,5 (aucun vrai signal à sélectionner)
    pbos = [mx.pbo_cscv(pd.DataFrame(np.random.default_rng(s).normal(size=(20, 8)))) for s in range(30)]
    assert 0.35 <= float(np.mean(pbos)) <= 0.65


def test_pbo_low_when_one_config_dominates():
    rng = np.random.default_rng(2)
    perf = pd.DataFrame(rng.normal(0, 0.1, size=(10, 8)))
    perf.iloc[3] += 2.0                              # une configuration réellement meilleure partout
    assert mx.pbo_cscv(perf) < 0.2


def test_pbo_partitions_are_random_balanced_and_reproducible():
    # une configuration meilleure UNIQUEMENT sur la première moitié des découpages : l'ancien tirage
    # lexicographique (début toujours en échantillon) l'aurait déclarée suroptimisée presque partout ;
    # un tirage équilibré au hasard doit la voir gagner IS et OOS dans une bonne part des partitions
    perf = pd.DataFrame(np.zeros((6, 28)))
    perf.iloc[2, :14] = 1.0
    p = mx.pbo_cscv(perf, n_partitions=500, seed=123)
    assert p < 0.9                                    # l'ancien biais donnait ~1,0 sur ce motif
    assert p == mx.pbo_cscv(perf, n_partitions=500, seed=123)   # graine fixée : reproductible
