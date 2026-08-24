# Mémoire 2.0 : la même question, sans les biais

Reprise de mon mémoire de maîtrise (*Évaluation empirique d'actifs canadiens par l'apprentissage
automatique*, UQAM, décembre 2024) avec les méthodes de 2026 : chaque biais mesuré dans la
[version 1](https://github.com/Guilou001/memoire-uqam-2024) est corrigé ici, un par un, et chaque résultat
porte maintenant sa probabilité de n'être que du bruit.

[![ci](https://github.com/Guilou001/memoire-2.0/actions/workflows/ci.yml/badge.svg)](https://github.com/Guilou001/memoire-2.0/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.12-blue)
![licence](https://img.shields.io/badge/code-MIT-green)

**Résultat en une phrase (premiers chiffres, protocole rapide, à confirmer par le protocole complet en
cours).** Une fois l'information réellement disponible au moment de décider, les hyperparamètres choisis
hors de la période de test et les coûts de transaction facturés, **aucun modèle ne bat le simple
portefeuille équipondéré** sur les actions américaines 2008-2024 (équipondéré : Sharpe 0,72 ; meilleur
modèle net de coûts : 0,00 ; la rotation d'environ 2 par mois coûte à elle seule 2,4 % par an). C'est la
conclusion honnête que la version 1 du mémoire ne pouvait pas voir, et c'est exactement ce que la
littérature récente prédit (Jensen, Kelly, Malamud et Pedersen, 2026).

*English summary.* My 2024 MSc thesis asked whether machine learning fed with macroeconomic data predicts
Canadian and US stock returns, and whether long short portfolios built on those predictions make money.
This repository re-answers the question with 2026 best practices: real-time information alignment,
hyperparameters selected on a pre-test validation window and then frozen, per-stock characteristics
(momentum, volatility) interacted with macro factors learned inside each fold, combinatorial purged
cross-validation with embargo, net-of-cost portfolios, Newey-West t-statistics, deflated Sharpe ratios and
the probability of backtest overfitting. First (fast-protocol) result: net of 10 bp costs, nothing beats
the equal-weight benchmark on US stocks; full results are computing.

## 1. Pourquoi une version 2

La version 1 reproduit le mémoire à l'identique et documente quatre problèmes, chacun mesuré dans
[son dépôt](https://github.com/Guilou001/memoire-uqam-2024) : les hyperparamètres étaient choisis sur la
période de test ; les variables macroéconomiques étaient alignées un mois en avance sur le rendement à
prédire ; plusieurs modèles prédisaient la même valeur pour tous les titres (leurs portefeuilles se
réduisaient à l'ordre alphabétique des colonnes) ; et les coûts de transaction étaient à zéro. Or un
backtest n'a de valeur que si l'information utilisée était disponible au moment de décider, et que si les
frictions sont payées. Ce dépôt reconstruit donc tout le protocole autour de cette exigence, en gardant la
question, les données et les familles de modèles du mémoire.

## 2. Ce que la version 2 fait de différent, point par point

| Problème de la v1 | Correction ici | Où dans le code |
|---|---|---|
| Macro un mois dans le futur | La macro attachée à une date de formation est la dernière ligne **publiée** avant cette date (décalage de 2 mois, convention FRED-MD vérifiée) | `panel.py`, paramètre `macro_lag` |
| Hyperparamètres choisis sur le test | Grilles évaluées sur 2004-2007 (entraînement 2000-2003), puis **gelées** ; 2008-2024 ne sert qu'à mesurer | `runner.py`, `select_hyperparameters` |
| Un seul chemin de backtest | Validation croisée purgée combinatoire : 28 chemins hors échantillon avec purge et embargo, d'où une **distribution** de Sharpe par modèle et la probabilité de suroptimisation (PBO) | `validation.py`, `metrics.pbo_cscv` |
| Coûts à zéro | 10 points de base par unité échangée, rotation mesurée, tout se lit **net** | `portfolio.py` |
| Sharpe brut sans correction | Sharpe déflaté (Bailey et López de Prado, 2014) : la probabilité que le Sharpe survive au nombre d'essais tentés ; t-stat Newey-West | `metrics.py` |
| Macro seule comme prédicteur | Sept caractéristiques par titre (momentum 1, 3, 6 et 12-2 mois, volatilités, rendement quotidien maximal), normalisées en rangs par date, **interagies** avec des facteurs macro appris par ACP à l'intérieur de chaque pli | `panel.py`, `models.py` |
| Modèles constants non détectés | Une date où toutes les prédictions sont égales est neutralisée (aucune position) au lieu de sélectionner par ordre alphabétique | `portfolio.py`, `quantile_weights` |
| Étalage de modèles sans référence | Deux repères sans apprentissage : l'équipondéré, et le classement mécanique momentum/volatilité, le contrôle proposé par Nagel (2025) contre les faux gains de complexité | `portfolio.momentum_vol_benchmark` |

Le zoo de modèles reste dans l'esprit du mémoire (ridge, filet élastique, forêt aléatoire, Extra Trees,
gradient boosting, petit perceptron), avec deux ajouts de la littérature récente : le **ridge à traits de
Fourier aléatoires** de Kelly, Malamud et Zhou (2024), et un **ensemble** (moyenne des prédictions
standardisées). Tout est scikit-learn : pas de dépendance fragile.

## 3. La méthode, pas à pas

1. **Construire le panel.** Une ligne = un titre à une date de formation (un 1er de mois). La cible est le
   rendement du mois qui commence à cette date. Les caractéristiques n'utilisent que des prix antérieurs ;
   la macro attachée est la dernière publiée. Un test automatique vérifie chacun de ces alignements.
2. **Choisir les réglages, puis les geler.** Chaque famille de modèles essaie sa petite grille sur
   2004-2007. Le meilleur réglage est retenu une fois pour toutes : quand la période de test commence, plus
   rien ne bouge. Le nombre total d'essais est compté, car il entre dans le Sharpe déflaté.
3. **Mesurer deux fois.** D'abord en marche avant (réentraînement annuel sur fenêtre croissante,
   prédictions mensuelles), le protocole lisible ; ensuite en validation croisée purgée combinatoire, le
   protocole robuste, qui juge chaque modèle sur 28 chemins hors échantillon et alimente la PBO.
4. **Payer les coûts, compter les positions.** Chaque mois, achat du quintile du haut, vente à découvert du
   quintile du bas (10 titres de chaque côté), poids égaux, 10 pb par unité échangée.
5. **Ne croire un Sharpe que déflaté.** Chaque Sharpe net est accompagné de sa t-stat Newey-West et de son
   Sharpe déflaté ; la grille entière passe à la PBO.

## 4. Premiers résultats (mesurés, protocole rapide : ridge et gradient boosting, États-Unis)

| Portefeuille | TCAC net | Sharpe net | Perte max. | Rotation/mois |
|---|---:|---:|---:|---:|
| Ridge | −2,9 % | −0,12 | −49,8 % | 2,0 |
| Hist Gradient Boosting | −6,0 % | −0,34 | −75,9 % | 2,5 |
| Ensemble | −1,4 % | −0,03 | −54,3 % | 2,2 |
| Contrôle momentum/volatilité (sans apprentissage) | −5,6 % | −0,57 | −60,4 % | 1,7 |
| **Équipondéré long only** | **10,6 %** | **0,72** | −37,3 % | |

Comment lire ce tableau, en deux constats : d'abord, la rotation de 2 par mois facturée à 10 pb coûte
environ 2,4 % par an, ce qui suffit à effacer le peu de signal ; ensuite, le contrôle sans apprentissage
fait aussi mal que les modèles, signe qu'il n'y a pas, dans ces données mensuelles, de structure
exploitable après coûts. Le R² de validation le plus élevé revient d'ailleurs aux Extra Trees (0,02) et au
ridge à traits aléatoires (0,016) : positif, mais minuscule. Le protocole complet (7 familles, ensemble,
CPCV, PBO, volet canadien) tourne au moment d'écrire ces lignes ; ce tableau sera remplacé par les
résultats complets, avec les figures.

## 5. Reproduire

```bash
uv sync --locked --all-extras          # environnement verrouillé (Python 3.12, pandas 3, scikit-learn 1.9)
uv run pytest                          # 16 tests : alignements, purge, portefeuille, métriques
uv run python scripts/fetch_data.py    # prix Yahoo ; déposer macro_data.csv (LCDMA) et Fred-MD.csv à la main
uv run m2 run --country both           # protocole complet ; --fast pour un essai en 2 minutes
uv run m2 figures --country both
```

Les chiffres du tableau ci-dessus viennent de `results/tables/walk_forward_usa.csv` ; aucun chiffre du
README n'est retapé à la main.

## 6. Limites, avec leur statut

| Limite | Statut |
|---|---|
| Univers de titres survivants (les 50 titres actuels, pas ceux de l'époque) | reconnu ; hérité de la v1, un univers point-in-time exigerait des listes historiques de constituants |
| Pas de taille ni de fondamentaux par titre (prix seulement) | reconnu ; l'imputation propre de fondamentaux (Bryzgalova et al., 2025) est l'extension naturelle |
| Millésime macro final (révisions non simulées) ; le décalage de publication, lui, est corrigé | reconnu ; millésimes ALFRED en extension |
| Coûts fixes à 10 pb, sans impact de marché | modélisé ; paramètre `fee` |
| Fréquence mensuelle, dérive intra-mois simplifiée | choix assumé, testé sans effet sur les conclusions en v1 |

## 7. Références

Gu, Kelly et Xiu (2020, RFS) ; Kelly, Malamud et Zhou (2024, JF) et la réponse de Nagel (2025) ;
Jensen, Kelly, Malamud et Pedersen (2026, RFS) ; López de Prado (2018), Bailey et López de Prado (2014) ;
Arian, Norouzi et Seco (2024, KBS) ; Goulet Coulombe et Göbel (2024) ; Bryzgalova, Lerner, Lettau et
Pelger (2025, RFS). Données : Yahoo Finance (usage personnel), LCDMA (Fortin-Gagnon, Leroux, Stevanovic et
Surprenant, 2022), FRED-MD (McCracken et Ng, 2016).

Code MIT ; texte et figures CC BY 4.0. Guillaume Vaudescal, avec le dépôt frère
[memoire-uqam-2024](https://github.com/Guilou001/memoire-uqam-2024) pour la version reproduite du mémoire.
