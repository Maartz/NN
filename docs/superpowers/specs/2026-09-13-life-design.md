# Une première simulation de vie

Le cadre a été approuvé dans la conversation : une créature dans une grille,
avec nourriture et énergie, un comportement programmé observable, puis un
réseau et une population qui évolue. Le but est de comprendre chaque étape.

## Choix

Une simulation Erlang avec un replay HTML autonome offre un résultat visible
sans serveur applicatif. Un affichage terminal serait plus rapide à construire
mais moins lisible ; un serveur interactif ajouterait un protocole inutile à ce
premier exercice. Le navigateur affiche des épisodes réellement calculés par
Erlang, et ne prétend pas entraîner le réseau en direct.

## Monde

Grille 12 × 12, départ au centre, 14 nourritures distinctes. Énergie initiale
40, capacité 60, coût de chaque action 1, gain de nourriture 18. Quatre actions
cardinales ; un mur bloque le mouvement mais coûte de l'énergie. Un épisode
finit à zéro énergie, après 160 actions, ou quand toute la nourriture a été
mangée. Pas de réapparition de nourriture, d'obstacles intérieurs ni de combats.

Sept perceptions : direction normalisée vers la nourriture la plus proche
(dx, dy), énergie / capacité, quatre indicateurs de mur. La créature dispose
donc d'une direction globale vers la nourriture, ce qui est explicitement
visible. Le comportement programmé réduit la distance de Manhattan.

## Réseau et évolution

Un petit réseau 7 → 6 → 4, tanh, calculé par des fonctions Erlang. L'action est
la sortie maximale (ordre de départage : nord, est, sud, ouest). Les génomes
sont des listes de poids avec biais. Architecture fixe : seules les valeurs
évoluent. Chaque évaluation de candidat s'exécute dans un processus surveillé.

Population de 32, 40 générations, 3 mondes d'entraînement identiques pour tous
les candidats, seed explicite. Score = 100 × nourritures + actions survécues.
Conservation des quatre meilleurs, reproduction par mutations de leurs poids.
La sélection ne voit jamais les trois mondes de validation. Les performances
affichées comparent sur ces mondes le programme, le meilleur réseau initial,
et le champion final. Une seed fixe garantit une expérience reproductible sur
le même runtime, pas une réussite universelle.

## Lecture visuelle

Une grille centrale, une créature orientée, sa trace et ses nourritures. Trois
modes, trois mondes de validation, lecture/pause, pas suivant, curseur temporel,
vitesse. Panneau perceptions → décision → conséquence et courbe du score
moyen d'entraînement du champion par génération. Tous les nombres proviennent
des épisodes exportés. L'état initial et l'état final sont inspectables.

Interface française, fond bleu pâle #edf3f8, encre #20364c, bleu #326dc9,
nourriture orange #dd792d, grille blanche. Titres Georgia sobres, corps système,
nombres monospace. La signature est le trajet visible dans la grille, pas une
page promotionnelle. Interface clavier, responsive, sans dépendance distante.

## Modules et vérification

life_world : règles pures et épisodes déterministes.
life_brain : réseau et mutation des poids.
life_evolution : sélection et évaluations surveillées.
life_demo : export du jeu d'épisodes et assemblage du HTML.

Tests : consommation, murs, énergie, terminaison, déterminisme, sorties réseau,
sélection élitiste, reproductibilité, erreurs explicites des workers. Vérifier
les épisodes affichés, les contrôles navigateur, puis toute la suite XOR.
Documenter les commandes et un parcours de lecture court dans docs/LIFE.md.
Les modules XOR et Mnesia existants restent disponibles.
