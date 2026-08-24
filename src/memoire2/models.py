"""Modèles : plan de features (caractéristiques + facteurs macro appris DANS le pli) et zoo de modèles.

La pièce centrale est ``MacroFactorDesign``, un transformeur scikit-learn : il standardise les variables
macro et en extrait ``n_factors`` composantes principales, AJUSTÉES UNIQUEMENT SUR LES DATES DU PLI
D'ENTRAÎNEMENT, puis construit la matrice finale : caractéristiques, facteurs macro, et interactions
caractéristiques x premiers facteurs (l'idée de Gu, Kelly et Xiu, 2020 : l'effet du momentum peut dépendre
de l'état macro). Fitter l'ACP dans le pli est ce qui évite la fuite d'information dans la validation
croisée purgée.

Le zoo couvre : ridge, filet élastique, forêt aléatoire, Extra Trees (sans élagage ccp_alpha, la leçon de
la v1), Hist Gradient Boosting, petit perceptron, et le ridge à traits de Fourier aléatoires (RFF) de
Kelly, Malamud et Zhou (2024). Tous scikit-learn : aucune dépendance à libomp.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from memoire2.panel import CHARACTERISTICS

SEED = 123


class MacroFactorDesign(BaseEstimator, TransformerMixin):
    """Caractéristiques + facteurs macro (ACP ajustée sur le pli d'entraînement) + interactions."""

    def __init__(self, n_factors: int = 8, n_interact: int = 3):
        self.n_factors = n_factors
        self.n_interact = n_interact

    def _split(self, X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        chars = X[[c for c in CHARACTERISTICS if c in X.columns]]
        macro = X[[c for c in X.columns if c.startswith("m_")]]
        return chars, macro

    def fit(self, X: pd.DataFrame, y=None):
        _, macro = self._split(X)
        # une observation par date (la macro est identique pour tous les titres d'une date)
        macro_by_date = macro.groupby(level="date").first()
        self.scaler_ = StandardScaler().fit(macro_by_date)
        k = min(self.n_factors, macro_by_date.shape[1], max(2, len(macro_by_date) - 1))
        self.pca_ = PCA(n_components=k, random_state=SEED).fit(self.scaler_.transform(macro_by_date))
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        chars, macro = self._split(X)
        factors = self.pca_.transform(self.scaler_.transform(macro.values))
        blocks = [chars.values, factors]
        for j in range(min(self.n_interact, factors.shape[1])):
            blocks.append(chars.values * factors[:, [j]])
        return np.hstack(blocks)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        chars = list(CHARACTERISTICS)
        names = chars + [f"facteur_macro_{j + 1}" for j in range(self.pca_.n_components_)]
        for j in range(min(self.n_interact, self.pca_.n_components_)):
            names += [f"{c} x f{j + 1}" for c in chars]
        return np.array(names)


def model_zoo(seed: int = SEED) -> dict[str, Pipeline]:
    """Une configuration raisonnable par famille (les grilles fines sont dans ``grid`` ci-dessous)."""
    design = MacroFactorDesign()
    rff = Pipeline([("design", MacroFactorDesign()), ("scale", StandardScaler()),
                    ("rff", RBFSampler(n_components=3000, gamma=0.5, random_state=seed)),
                    ("est", Ridge(alpha=1000.0, random_state=seed))])
    return {
        "ridge": Pipeline([("design", clone(design)), ("scale", StandardScaler()),
                           ("est", Ridge(alpha=10.0, random_state=seed))]),
        "elastic_net": Pipeline([("design", clone(design)), ("scale", StandardScaler()),
                                 ("est", ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=seed, max_iter=5000))]),
        "random_forest": Pipeline([("design", clone(design)),
                                   ("est", RandomForestRegressor(n_estimators=300, min_samples_leaf=50,
                                                                 max_features=0.3, random_state=seed, n_jobs=2))]),
        "extra_trees": Pipeline([("design", clone(design)),
                                 ("est", ExtraTreesRegressor(n_estimators=300, min_samples_leaf=50,
                                                             max_features=0.3, random_state=seed, n_jobs=2))]),
        "hist_gb": Pipeline([("design", clone(design)),
                             ("est", HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05,
                                                                   max_iter=300, l2_regularization=1.0,
                                                                   random_state=seed))]),
        "mlp": Pipeline([("design", clone(design)), ("scale", StandardScaler()),
                         ("est", MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1e-3, max_iter=500,
                                              early_stopping=True, random_state=seed))]),
        "rff_ridge": rff,
        "ensemble": None,  # moyenne des prédictions standardisées des autres, calculée dans runner.py
    }


def grid() -> dict[str, list[dict]]:
    """Petites grilles par famille, évaluées UNIQUEMENT sur la fenêtre de validation (2004-2007)."""
    return {
        "ridge": [{"est__alpha": a} for a in (1.0, 10.0, 100.0)],
        "elastic_net": [{"est__alpha": a} for a in (0.0005, 0.001, 0.01)],
        "random_forest": [{"est__min_samples_leaf": leaf} for leaf in (20, 50, 100)],
        "extra_trees": [{"est__min_samples_leaf": leaf} for leaf in (20, 50, 100)],
        "hist_gb": [{"est__learning_rate": lr, "est__max_depth": d} for lr in (0.03, 0.05) for d in (2, 3)],
        "mlp": [{"est__alpha": a} for a in (1e-4, 1e-3, 1e-2)],
        "rff_ridge": [{"est__alpha": a, "rff__n_components": n} for a in (100.0, 1000.0, 10000.0) for n in (1000, 3000)],
    }
