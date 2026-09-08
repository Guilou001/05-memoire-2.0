#set document(title: "Un ordinateur peut-il mieux choisir des actions après avoir payé les frais ?", author: "Guillaume Vaudescal")
#set page(
  paper: "a4",
  margin: (x: 2.2cm, y: 2.4cm),
  numbering: "1 / 1",
  footer: context [
    #set text(size: 8pt, fill: luma(90))
    #grid(columns: (1fr, auto), align: (left, right),
      [memoire-2.0], [#counter(page).display("1 / 1", both: true)])
  ],
)
#set text(font: ("Helvetica", "Arial", "DejaVu Sans"), size: 10pt, lang: "fr")
#set par(justify: true, leading: 0.68em, spacing: 1.1em)
#set heading(numbering: none)
#show heading.where(level: 2): it => block(above: 1.6em, below: 0.8em, text(size: 13pt, it))
#show heading.where(level: 3): it => block(above: 1.2em, below: 0.6em, text(size: 11pt, it))
#show raw.where(block: true): it => block(
  fill: luma(246), inset: 8pt, radius: 3pt, width: 100%, text(size: 8.5pt, it))
#show raw.where(block: false): it => text(size: 9pt, fill: rgb("#1a3f66"), it)
#show quote.where(block: true): it => block(
  inset: (left: 10pt), stroke: (left: 1.5pt + luma(180)),
  text(style: "italic", fill: luma(45), it.body))
// la table NE DOIT PAS être enfermée dans un par() : Typst 0.15 la supprime alors
// entièrement, sans erreur. Le réglage se pose donc dans la portée du bloc.
#show table: it => block(above: 1.1em, below: 1.1em,
  [#set par(justify: false); #text(size: 8.8pt, it)])
#show figure: it => block(above: 1.4em, below: 1.4em, it)
#show figure.caption: it => text(size: 8.5pt, fill: luma(70), it)
#show link: it => text(fill: rgb("#0072B2"), it)

#align(center)[
  #block(width: 100%)[
    #text(size: 18pt, weight: "bold")[Un ordinateur peut-il mieux choisir des actions après avoir payé les frais ?]
    #v(0.6em)
    #text(size: 10pt, fill: luma(70))[Guillaume Vaudescal · 2026-09-08 · #link("https://github.com/Guilou001/05-memoire-2.0")[Guilou001/05-memoire-2.0]]
  ]
]
#v(1.2em)
#line(length: 100%, stroke: 0.6pt + luma(190))
#v(0.8em)

Un ordinateur peut chercher des liens entre l'économie et les prix des actions. Mais un résultat historique peut être trompeur si le programme utilise une information connue seulement plus tard, ou s'il oublie les frais.

Ce projet reprend la question de mon mémoire avec des contrôles supplémentaires. Les réglages des modèles sont choisis avant la période de test. Chaque achat et vente est facturé.

*Aucun des huit modèles ne dépasse la répartition égale dans les deux univers étudiés.* Cela décrit ces expériences sur 2008 à 2024. Ce n'est pas une conclusion générale sur toute utilisation de l'intelligence artificielle.

== Comparer le résultat avec une règle simple

#table(
  columns: 3,
  stroke: (x, y) => if y == 0 { (bottom: 0.6pt) } else { none },
  align: left + top,
  inset: 5pt,
    [*Portefeuille canadien*],
    [*Rendement annuel composé après frais*],
    [*Sharpe*],
    [Meilleur modèle testé, arbres corrigés successivement],
    [4,1 %],
    [0,31],
    [Même somme dans chaque action],
    [11,4 %],
    [0,85],
)

Le Sharpe mesure le rendement au-delà du taux sans risque, rapporté aux variations du rendement. Les valeurs proviennent de la #link("results/tables/walk_forward_canada.csv")[table canadienne]. Les #link("results/tables/walk_forward_usa.csv")[résultats américains] sont présentés séparément.

#figure(image("../results/figures/presentation.png", width: 100%), caption: [Sharpe des huit modèles et de la répartition égale au Canada])

Chaque barre montre le compromis entre rendement et variabilité après coûts. La barre bleue répartit l'argent également entre les actions. Elle dépasse les huit modèles sur cet échantillon, avec une exposition au marché différente.

== Pourquoi un signal ne suffit pas

Les modèles achètent les actions qu'ils classent en haut et vendent à découvert celles du bas. Vendre à découvert revient à vendre une action empruntée, puis à la racheter. Le pari porte donc sur l'écart entre deux groupes.

Changer souvent de classement oblige à beaucoup négocier. Le coût retenu est de *0,10 % par montant échangé*. Plusieurs modèles échangent l'équivalent de deux à trois portefeuilles par mois.

La comparaison est aussi répétée sur plusieurs découpages temporels. Un contrôle tient compte du nombre de réglages essayés, car essayer davantage augmente les chances de trouver un bon résultat par hasard.

== Les réserves à garder en tête

Les modèles achètent et vendent à découvert, tandis que la référence détient simplement les actions. Leurs expositions au marché diffèrent. Les sociétés retenues sont des survivantes et les données économiques utilisent des historiques révisés.

Le fichier américain contient aussi des séries en niveaux non transformés, contrairement au fichier canadien. Cette différence empêche d'attribuer l'écart entre pays à une cause unique. Tous ces défauts restent détaillés dans l'étude liée ci-dessous.

== Refaire les calculs

#raw("uv sync --locked --all-extras\nuv run pytest\nuv run m2 figures --country both", block: true, lang: "bash")

Le calcul complet utilise #raw("uv run m2 run --country both"). Il nécessite les fichiers macroéconomiques déposés manuellement selon l'étude détaillée et peut être long. La commande de figures utilise les résultats déjà disponibles. Le graphique de présentation se régénère hors réseau avec #raw("uv run python scripts/figure_presentation.py"), depuis les tableaux publiés.

== Pour aller plus loin

#link("docs/ETUDE_DETAILLEE.md")[Méthodes, résultats complets et références] · #link("rapport/rapport.pdf")[Présentation en PDF] · #link("LICENSE")[Licence].

== English summary

Eight tested models fail to beat equal weighting after transaction costs. The study controls timing and repeated trials, while retaining declared limits involving survivor stocks, revised macro data and untransformed US series.
