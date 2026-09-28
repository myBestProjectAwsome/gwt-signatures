Feuille de route pour le projet gwt signatures

Ce document explique les corrections à apporter au prototype, les
expériences à construire et les conditions permettant de soutenir une
conclusion de généralité limitée. La priorité est de donner un objectif
final à l’agent, de lui laisser découvrir les prérequis, puis de
vérifier qu’il apprend une règle nouvelle à partir de ses interactions.

## La conclusion visée

Dans une famille définie de mondes à grille, le même agent réutilise ses
connaissances sur des configurations inédites, découvre des étapes
intermédiaires non fournies et améliore son comportement face à une
nouvelle règle. Ces capacités doivent être observées sur des épreuves
réservées, avec des budgets fixés et plusieurs entraînements
indépendants.

Cette conclusion ne certifie ni une IAG au sens large, ni une
conscience. Le terme mini IAG reste un nom de projet. La preuve
recherchée est expérimentale et limitée aux environnements effectivement
testés.

## Le point de départ

La revue porte sur le code et les résultats enregistrés dans
gwt-signatures.zip. Les modèles n’ont pas été réentraînés et les scores
n’ont pas été reproduits ; les poids entraînés ne sont pas inclus dans
cette archive.

Le résultat final enregistré rapporte 100 % de réussite sur H1 à H3, W1
et W2, et 99 % sur R0, avec 200 cartes par cas. Le diagnostic 12
attribue l’essentiel du progrès à la carte mentale. Le diagnostic 11
montre que l’apprentissage de la glace ne se traduit pas encore par une
amélioration fiable des actions.

## Les trois démonstrations à obtenir

| **Épreuve**      | **Question**                                                               |
|------------------|----------------------------------------------------------------------------|
| A Généralisation | Les compétences fonctionnent-elles sur de nouvelles structures de cartes ? |
| B Prérequis      | L’agent trouve-t-il les étapes nécessaires sans recevoir leur séquence ?   |
| C Adaptation     | L’expérience d’une règle nouvelle améliore-t-elle ses actions ?            |

Ordre recommandé : établir une référence reproductible, développer B sur
de petites cartes, corriger l’adaptation, puis exécuter une batterie
indépendante A B C.

# 1 Corriger les limites du prototype

## Les sous buts sont actuellement fournis

Dans evaluation/tasks.py, les tâches H2 et H3 contiennent respectivement
les séquences key puis door, et key puis door puis goal. TaskTracker
fournit le prochain événement à atteindre. Cela mesure l’exécution de
sous-buts connus, mais pas leur découverte.

Action : conserver ces tâches comme références historiques et créer une
nouvelle batterie où le seul événement demandé est goal. Ne pas ajouter
une règle cachée qui choisit automatiquement la clé dès qu’une porte
apparaît.

## La décision et son apprentissage doivent rester reliés

Agent.\_critic() utilise la carte mentale lorsqu’elle est présente.
ContinualLearner entraîne le prédicteur et éventuellement la critique
d’actions, mais pas cette carte. Le composant déterminant pour agir peut
donc rester figé pendant l’adaptation.

Action : entraîner le composant effectivement utilisé, ou faire arbitrer
le planificateur à partir de transitions apprises capables de modifier
le choix. Vérifier ce lien avec un scénario où une transition change et
où la meilleure action doit changer aussi.

## La référence sans connaissances doit pouvoir apprendre

build_relearner() initialise les modules aléatoirement, puis practice()
ne les entraîne pas tous. Une carte, une perception ou une évaluation
aléatoire non apprenable rend cette référence très faible.

Action : distinguer deux comparaisons. Une copie préentraînée figée
isole l’apport de l’adaptation. Une architecture repartant de zéro, dont
tous les modules nécessaires sont entraînables, mesure le bénéfice des
acquis antérieurs. Rapporter séparément les budgets de préentraînement
et d’adaptation.

## Les connaissances fournies doivent être déclarées

EventDetector utilise les canaux structurés pour reconnaître les
événements. MentalMap impose une propagation locale sur une grille et
trois cibles connues. Ces choix sont des connaissances de conception
légitimes, mais ils délimitent ce que l’expérience peut démontrer.

Action : écrire une liste des informations fournies : canaux, actions,
événements, récompenses, visibilité, inventaire et hypothèses sur les
déplacements. Ne pas présenter leur reconnaissance programmée comme une
découverte autonome.

## Le workspace doit être évalué selon son rôle réel

Dans Agent.act(), seuls les poids d’attention du workspace sont utilisés
; le contenu diffusé est ignoré. Sa présence ne prouve donc pas une
coordination cognitive globale.

Action : garder une variante sans workspace. Ne connecter sa diffusion à
la mémoire et au plan que si une expérience précise en justifie
l’utilité. Cette amélioration ne bloque pas les épreuves A B C.

# 2 Développer la découverte des prérequis

## Représenter les conséquences utiles à la décision

Le modèle du monde doit distinguer les états selon la position, la
possession de la clé et l’état de la porte. Deux états ayant la même
position peuvent autoriser des futurs différents. Utiliser d’abord les
observations structurées existantes évite d’ajouter simultanément un
problème de vision.

- Mesurer séparément les prédictions avant et après la prise de clé,
  devant une porte fermée et après son ouverture.

- Évaluer des séquences de plusieurs actions. Une bonne erreur moyenne
  sur les déplacements ordinaires peut masquer des erreurs critiques sur
  les événements rares.

- Séparer les faits observés des états prédits ; journaliser les
  conséquences imaginées et celles réellement constatées.

## Allonger la recherche sans explosion combinatoire

LatentPlanner énumère actuellement 1 024 séquences de cinq actions.
Augmenter directement cet horizon devient rapidement coûteux. Une
première évolution possible consiste à développer progressivement les
trajectoires et à ne conserver qu’un nombre limité de candidats, avec
détection des états répétés et maintien d’une diversité de candidats.

Le score doit pouvoir garder une trajectoire temporairement éloignée du
but mais utile, comme aller chercher une clé. Fixer un budget de
simulations et mesurer la sensibilité à ce budget. La recherche dans un
modèle appris peut exploiter ses erreurs : vérifier régulièrement les
plans par leurs résultats réels et replanifier après chaque action.

Un planificateur utilisant les vraies transitions peut servir d’oracle
de diagnostic pour vérifier la solvabilité. Il ne doit pas être utilisé
par l’agent évalué si la capacité revendiquée repose sur des transitions
apprises.

## Construire quatre situations de développement

| **Situation**               | **Décision attendue**                  |
|-----------------------------|----------------------------------------|
| Objectif accessible         | Aller directement au but.              |
| Porte fermée incontournable | Prendre la clé puis franchir la porte. |
| Porte déjà ouverte          | Ignorer la clé inutile.                |
| Détour plus court           | Choisir le détour plutôt que la clé.   |

Fournir uniquement le but final dans chaque situation. Varier les
positions pour empêcher une stratégie fondée sur un repère fixe.
Vérifier que la génération produit les quatre cas voulus et que leurs
chemins de référence correspondent bien aux règles.

## Le jalon à valider

Livrer un agent qui réussit ces situations sur des cartes nouvelles,
avec le même mécanisme de décision. Comparer à la carte seule et à une
variante privée de prédictions apprises. Si les performances sont
identiques, ne pas conclure que le modèle du monde explique la réussite.

# 3 Apprendre une règle nouvelle

## Utiliser la glace pour développer la méthode

La glace est déjà connue du développeur et a déjà fait l’objet de
diagnostics. Elle constitue donc une épreuve de développement, pas une
nouveauté indépendante pour le prochain verdict. Pour commencer, fixer
une seule modification des transitions et garder les autres règles
constantes.

Le nouveau canal de perception éventuellement ajouté doit être déclaré.
Son ajout manuel ne doit pas être présenté comme la découverte autonome
d’un nouveau sens. Choisir explicitement quels paramètres et quelles
mémoires peuvent évoluer pendant l’adaptation.

## Adapter le composant responsable des actions

La carte actuelle lit les canaux d’origine et propage des valeurs par
déplacements locaux. Ajouter la glace au seul encodeur latent ne lui
permet pas automatiquement de représenter une glissade. Deux options
sont possibles : étendre la carte aux nouvelles transitions, ou
conserver la carte comme compétence de navigation et laisser un modèle
de transitions appris guider les cas nouveaux.

Choisir une option sur les données de développement. Ne pas figer toute
l’évaluation des événements si cela empêche de reconnaître les nouveaux
états : distinguer les préférences stables, comme éviter la lave, des
prédictions apprises de leurs conséquences.

## Comparer trois conditions au même point de départ

| **Condition**                         | **Interprétation**                                          |
|---------------------------------------|-------------------------------------------------------------|
| Copie figée                           | Mesure la compétence initiale sans mise à jour.             |
| Rejeu ancien uniquement               | Mesure le bénéfice du calcul supplémentaire sur les acquis. |
| Expériences nouvelles et rejeu ancien | Mesure le bénéfice des nouvelles interactions.              |

Utiliser les mêmes poids de départ et des cartes d’évaluation communes.
Égaliser le budget de mises à jour entre les deux conditions
apprenantes. Évaluer sans mise à jour pendant la mesure, sur des cartes
distinctes de celles utilisées pour apprendre.

## Mesurer une progression utile

- Mesurer la réussite, les morts et le nombre d’actions après 0, 10, 25,
  50 et 100 épisodes d’adaptation ; ajuster ces budgets uniquement
  pendant le développement.

- Mesurer aussi la précision des transitions rares, mais conserver la
  réussite comportementale comme résultat principal.

- Réévaluer les anciennes tâches avant et après adaptation pour mesurer
  la perte de compétences.

Le jalon est atteint lorsque les nouvelles expériences améliorent les
actions davantage que les deux témoins, avec une conservation suffisante
des acquis. Si seules les prédictions progressent, examiner leur
influence sur la décision avant d’ajouter des modules.

# 4 Préparer une évaluation indépendante

## Séparer entraînement développement et test

L’entraînement sert à apprendre les paramètres. Le développement sert à
choisir les réglages, les budgets et les seuils. Le test réservé sert
uniquement à évaluer la version figée. Utiliser des graines distinctes
et, pour certaines épreuves, des familles de cartes distinctes : changer
seulement la graine ne garantit pas une nouveauté structurelle.

Faire préparer si possible les cas réservés par une personne qui ne
règle pas l’agent. Annoncer le périmètre admissible, sans révéler les
instances. Pour C, réserver une autre règle de transition compatible
avec les entrées et actions annoncées. Ne pas demander au système de
découvrir une information qu’il ne peut ni observer ni inférer.

## Fixer les critères avant de voir les résultats

Les seuils suivants sont des propositions de travail, pas des normes de
l’IAG. Les valider sur le développement, puis les figer avec la taille
des échantillons et la méthode statistique.

| **Mesure**         | **Seuil proposé**                                                                                        |
|--------------------|----------------------------------------------------------------------------------------------------------|
| A et B             | Au moins 80 % de réussite dans chaque famille ou situation.                                              |
| C après adaptation | Gain d’au moins 15 points par rapport au départ et avantage sur chacun des deux témoins.                 |
| Conservation       | Baisse maximale de 5 points sur les anciennes familles.                                                  |
| Répétabilité       | Au moins 5 entraînements indépendants et 200 cartes par condition et par graine, si le budget le permet. |

Présenter les résultats par graine et par famille, avec des intervalles
à 95 %. Pour les comparaisons, apparier les cartes et les graines. Ne
pas traiter toutes les cartes issues d’un même entraînement comme des
modèles indépendants. Une estimation hiérarchique peut prendre en compte
ces deux sources de variation ; cinq graines donnent encore une
précision limitée.

Prévoir une issue « résultat incertain » lorsque les intervalles ne
permettent pas de trancher. Les seules moyennes ponctuelles ne suffisent
pas à affirmer une supériorité ou une conservation fiable.

## Préserver les traces et éviter la fuite du test

- Figer le code, les poids, les réglages, les budgets et le protocole ;
  conserver leurs empreintes et la version de l’environnement.

- Réinitialiser l’état et la mémoire entre épisodes selon une règle
  annoncée. Pour C, préciser exactement ce qui persiste pendant
  l’adaptation.

- Enregistrer les actions, événements, résultats, temps de calcul,
  graines et identifiants des poids ; publier aussi les échecs.

- Si le test inspire une correction, le reclasser en développement et
  préparer de nouvelles épreuves réservées.

# 5 Organiser les modifications et les livrables

Les chemins ci-dessous correspondent à l’archive examinée. Les nouveaux
noms proposés sont des suggestions ; aucune de ces modifications n’a été
réalisée par ce document.

| **Zone du dépôt**                | **Travail à réaliser**                                                              |
|----------------------------------|-------------------------------------------------------------------------------------|
| evaluation/                      | Créer une batterie distincte ; conserver le protocole et les résultats historiques. |
| mini_iag/tasks/ et task_agent.py | Accepter un but final sans séquence intermédiaire imposée.                          |
| modules/world_model.py           | Améliorer et diagnostiquer les transitions pertinentes sur plusieurs pas.           |
| planning/latent_planner.py       | Remplacer l’énumération exhaustive pour les horizons longs ; journaliser les plans. |
| modules/mental_map.py et life/   | Relier l’adaptation aux décisions et prendre en compte les nouvelles transitions.   |
| final_agent.py                   | Construire les témoins figés et les références entraînables correctement.           |
| diagnostics/                     | Ajouter les comparaisons sans carte, sans prédiction et sans adaptation.            |

# 6 Exécuter les lots et présenter le résultat

Lot 1 — Référence reproductible : récupérer les poids ou documenter leur
réentraînement, conserver la version initiale, enregistrer les résultats
détaillés et définir les témoins.

Lot 2 — But final : construire les quatre situations de prérequis et
vérifier leur solvabilité avec un oracle réservé au diagnostic.

Lot 3 — Modèle et recherche : améliorer les transitions critiques,
développer la recherche et établir quelles briques expliquent les
performances.

Lot 4 — Adaptation : entraîner le composant décisionnel sur la glace,
comparer les trois conditions et mesurer la conservation des acquis.

Lot 5 — Verdict réservé : figer la version, exécuter A B C, analyser les
incertitudes et rédiger une conclusion limitée aux capacités observées.

## Ce qui peut attendre

La conversation, une perception en images brutes, une mémoire
autobiographique et un workspace plus élaboré ne sont pas nécessaires à
cette première démonstration. Un second environnement sera utile ensuite
pour étudier le transfert entre domaines. Il faudra montrer que les
acquis du premier accélèrent l’apprentissage du second.

## La formulation du résultat

Si A B C réussissent : « Dans les familles de mondes évaluées, cet agent
généralise à des configurations inédites, découvre des prérequis non
fournis et adapte son comportement à une règle nouvelle grâce à ses
interactions. » En cas de réussite partielle, nommer seulement les
capacités validées et conserver les autres comme objectifs.

Sources du diagnostic : README.md, evaluation/resultat_final.json,
evaluation/tasks.py, mini_iag/agent.py, task_agent.py, final_agent.py,
modules/mental_map.py, event_detector.py, life/continual_learner.py et
diagnostics 11 et 12 de l’archive fournie. Les recommandations et seuils
de ce document sont des propositions de conception et doivent être
validés expérimentalement.
