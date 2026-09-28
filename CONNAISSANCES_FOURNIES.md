# Ce qui est fourni à l'agent (et ce qu'il apprend)

Feuille de route v2, lot 1. Ce document liste les informations **données par
construction** à la mini-IAG. Ce sont des choix de conception légitimes, mais
ils délimitent ce que les expériences peuvent démontrer. Rien de ce qui figure
dans la colonne « fourni » ne doit être présenté comme une découverte autonome.

État décrit : la version du verdict du 27 septembre 2026
(`poids/verdict_2026-09-27/`, agent de l'étape 4a et carte mentale v2).

## Perception

| | Fourni | Appris |
|---|---|---|
| Observations | Une grille 7×7 **symbolique** et **entièrement visible** : un canal par type d'objet (mur, lave, objectif, agent, clé, porte) et un canal « clé en main ». Pas d'image, pas de bruit, pas de vue partielle. | Le vecteur latent qui résume la grille (encodeur JEPA). |
| Inventaire | Le canal « clé en main » dit directement si l'agent porte la clé. | — |
| Glace (option) | Le 8e canal de la glace a été **ajouté à la main** (`grow_senses`). Ce n'est pas la découverte d'un nouveau sens. | Les poids de ce canal (qui partent de zéro). |

## Actions

| | Fourni | Appris |
|---|---|---|
| Actions | Leur **nombre** (4) et leur caractère discret. | Leur **effet** : le modèle du monde prédit leurs conséquences, et la carte mentale apprend vers quelle case voisine mène chaque action (noyaux 3×3). |

## Événements, valeurs et buts

| | Fourni | Appris |
|---|---|---|
| Liste des événements | 4 événements nommés (objectif, clé, porte, lave). | — |
| Étiquettes d'événements | À l'entraînement, **le monde fournit les étiquettes** de chaque transition (quel événement a eu lieu). Le modèle du monde, le module de coût, la critique et la carte mentale apprennent à partir de ces étiquettes. | Prédire ces événements (y compris dans l'imagination). |
| Reconnaissance des événements pendant une tâche | `EventDetector` : des **règles écrites à la main** sur les canaux (la porte a disparu, le canal clé en main est passé à 1, l'agent est entré sur l'objectif ou la lave). | — |
| Valeurs | La lave est un danger, et les poids danger = 4 et succès = 1 ont été choisis par le concepteur (puis verrouillés). | — |
| Buts | Une tâche est une **séquence d'événements**. `TaskTracker` donne à l'agent le **prochain** événement à provoquer : les sous-buts (clé avant porte) sont **fournis**, pas découverts. | Comment atteindre chaque sous-but. |
| Tâches pratiquées | Aucune tâche du test. Les tâches publiques « objectif », « clé », « objectif puis clé » ont servi au développement et aux réglages. | — |

## Planification

| | Fourni | Appris |
|---|---|---|
| Carte mentale | La **forme du calcul** : un monde en cases, propagation locale des valeurs sur 25 itérations, factorisation en récompense / blocage / danger, trois cibles connues (objectif, clé, porte). Une seule couche : elle ne sait pas représenter « la porte devient franchissable après la clé ». | Ce qu'il y a sur chaque case (récompense, blocage, danger), où mène chaque action, les valeurs. |
| Planificateur | L'énumération des 1 024 plans de 5 actions, la formule du score, les poids (carte 30, pénalité de pas, nouveauté). | Les prédictions utilisées dans le score. |
| Actions inutiles | Règle écrite : si l'observation ne change pas, l'action est notée inutile dans cette situation. | — |
| Mémoire | Mémoire des états visités, **vidée à chaque épisode** et à chaque sous-but atteint. | — |
| Espace de travail | Seuls ses **poids d'attention** servent à choisir le plan ; le contenu qu'il diffuse est ignoré. Il ne prouve donc aucune coordination globale. | Son attention (étape 2). |

## Données et réglages

| | Fourni |
|---|---|
| Expérience d'entraînement | Un marcheur aléatoire sur 3 000 cartes standard publiques (150 000 transitions). |
| Cartes de la vie | Filtrées par une recherche exacte (`search.py`) pour qu'elles soient solubles : c'est l'environnement qui le fait, pas l'agent. |
| Oracle | Utilisé pour l'évaluation et les diagnostics, **jamais par l'agent** pour décider. |
| Réglages | Horizon, poids du danger, poids de la carte, etc. : choisis sur des cartes publiques de contrôle, parfois les mêmes graines que les diagnostics (écrit dans le README à chaque fois). |
| Entraînements | **Un seul** entraînement (une seule graine) : la variabilité d'un entraînement à l'autre n'a pas été mesurée. |

## Ce que ça implique pour les conclusions

- Le verdict du 27 septembre mesure l'**exécution** de sous-buts fournis, pas leur découverte.
- « Il a appris la physique seul » est vrai pour les conséquences des actions. C'est faux pour la reconnaissance des événements pendant une tâche, écrite à la main.
- La réussite repose sur une carte mentale dont la structure suppose un monde en cases.
- Sans plusieurs entraînements indépendants, on ne sait pas si le résultat est stable.

## Ajouts des lots 2 et 3 (développement, pas encore une version figée)

| | Fourni | Appris |
|---|---|---|
| Planification par événements | La liste des événements possibles (objectif, clé, porte), la recherche de suites d'au plus 3 événements, l'idée qu'un événement peut changer le monde, la conversion valeur → distance. | Quels événements rendent quoi atteignable (carte mentale + modèle des conséquences). |
| Modèle des conséquences | Sa forme : il prédit l'observation après l'arrivée sur une case désignée. | Ce que change l'arrivée sur chaque case (l'agent y est, la clé disparaît et le canal clé en main s'allume, la porte disparaît…). |
| Carte v3 | L'a priori « les propriétés d'une case ne dépendent pas de la position de l'agent » (le canal de l'agent est masqué pour les calculer). | Les propriétés des cases. |
| Blocages constatés | La règle « une action qui ne change rien signale une case bloquante dans cette situation (avec ou sans la clé) » ; la case visée est lue sur les noyaux APPRIS de la carte. | — |
| Situations de développement | Générées par mon générateur et vérifiées par l'oracle ; les réglages ont été choisis dessus. | — |

