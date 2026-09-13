# Petite vie : le fil à suivre

Pour travailler hors ligne : **[atelier ExoSelf dans le train](TRAIN.md)**,
avec commandes copiables, redémarrages et exercice d'instrumentation.

L'objectif est de voir une créature agir, puis de comprendre comment ses
poids influencent ses décisions. Toute la simulation et l'évolution tournent
en Erlang. La page HTML rejoue les épisodes enregistrés.

## Lancer

Depuis la racine du projet, avec Erlang/OTP 26+ et rebar3 :

```sh
./scripts/life.sh
open _build/life/index.html
```

La page est autonome : elle fonctionne aussi sans serveur et sans accès réseau.
Le script compile, entraîne une population et ExoSelf, puis génère quatre fichiers :

- `_build/life/index.html` : la vue interactive avec ses données intégrées ;
- `_build/life/experiment.json` : les réseaux, les deux courbes et les douze épisodes ;
- `_build/life/champion.term` : le champion de la population, lisible avec `file:consult/1` ;
- `_build/life/exoself.ets` : le génotype entraîné par ExoSelf.

Les relances remplacent ces quatre fichiers. Pour garder une expérience :

```sh
./scripts/life.sh /tmp/mon-experiment/index.html
```

## Regarder avant de lire le code

1. La page démarre sur **ExoSelf**. Choisir **Programmé**, cliquer **Un pas**. Sur le premier monde, le
   premier mouvement mange une nourriture : énergie 40 → 57.
2. Lancer la lecture. Le programme ramasse les 14 nourritures : les règles
   du monde permettent donc de réussir sans réseau.
3. Choisir **Réseau initial**. Inspecter les quatre scores à droite : le plus
   élevé choisit le prochain mouvement. Ce ne sont pas des probabilités.
4. Comparer **Population** et **ExoSelf** sur les trois mondes. Sur le troisième,
   ils mangent respectivement cinq et sept nourritures, contre une au départ.
5. Revenir sur un pas avec le curseur. Les perceptions et les sorties décrivent
   la décision qui va être prise ; le message du bas décrit le pas précédent.
   À la fin, aucune nouvelle décision n'est affichée.

## Le code, dans cet ordre

### 1. `src/life_world.erl` : que peut faire une créature ?

`new(Seed)` construit une map contenant position, nourriture, énergie et temps.
`step(World, Action)` retourne une nouvelle map. Les quatre actions sont
`north`, `east`, `south`, `west`. Une action coûte 1 énergie ; manger rapporte
18, avec un plafond de 60. Un mur bloque le déplacement mais coûte quand même
1. La nourriture ne réapparaît pas. Un épisode finit à zéro énergie, après 160
actions, ou après avoir tout mangé.

`sense(World)` retourne sept nombres :

```text
[dx / 11, dy / 11, énergie / 60, mur_nord, mur_est, mur_sud, mur_ouest]
```

`dx` et `dy` pointent vers la nourriture la plus proche selon la distance de
Manhattan. Les coordonnées augmentent vers la droite et vers le bas. Les
indicateurs de mur valent 0 ou 1. C'est volontairement un capteur très informatif :
la créature connaît une direction globale, ce n'est pas une vision locale.

`scripted_action(World)` choisit simplement une direction vers cette cible.
`episode(Controller, Seed)` conserve toutes les images de l'épisode ;
`evaluate(Controller, Seed)` calcule les mêmes résultats sans stocker les images.

### 2. `src/life_brain.erl` : comment le réseau choisit-il ?

```text
7 perceptions → 6 neurones cachés → 4 sorties → direction du score maximal
```

Chaque neurone calcule `tanh(somme(poids × entrées) + biais)`. Un réseau est une
map avec deux listes de lignes : `hidden` et `output`. Dans chaque ligne, le
dernier nombre est le biais. Cela fait 76 nombres au total. En cas d'égalité,
les directions sont départagées dans l'ordre nord, est, sud, ouest.

Ce module calcule le réseau par appels de fonctions pour la référence
**Population**. Le mode **ExoSelf** utilise les vrais processus `sensor`,
`neuron`, `actuator` et `cortex` du projet, avec la même architecture et les mêmes
poids de départ. Les tests vérifient que les deux représentations produisent les
mêmes actions et états ; les sorties numériques concordent à 10⁻¹² près.

### 3. `src/life_evolution.erl` : la référence par population

1. Créer 32 réseaux aléatoires.
2. Évaluer chacun sur les trois mêmes mondes d'entraînement : seeds 11, 22, 33.
3. Classer sur la moyenne du score `100 × nourritures mangées + pas survécus`.
4. Garder les quatre meilleurs sans modification.
5. Créer les 28 autres à partir de ces parents : chaque poids a 15 % de chance
   de varier jusqu'à ±0,7 ; un coefficient est toujours choisi pour mutation.
   Les coefficients restent dans [-6, 6].
6. Répéter pendant 40 générations.

Il n'y a ni gradient, ni croisement, ni création de neurones. La conservation
des parents garantit que le meilleur score d'entraînement ne diminue pas.
Elle ne garantit aucune amélioration sur les mondes de validation.

Chaque candidat est évalué dans un processus surveillé. Ses trois épisodes
sont calculés successivement dans ce processus. Les réponses sont rassemblées
dans l'ordre de la population : l'ordonnancement ne change pas la sélection.
Une erreur de worker interrompt explicitement l'expérience ; on n'invente pas
un score à la place. Chaque réponse a une limite d'attente de cinq secondes.

### 4. `src/life_exoself.erl` et `src/life_scape.erl` : brancher ExoSelf

`life_exoself:save/2` traduit les 76 poids en génotype ETS. ExoSelf charge ce
fichier et crée les processus habituels. La morphologie `life_mimic` relie un
capteur à sept valeurs et un actionneur à quatre sorties au scape `life_sim`.

```text
cortex → sensor → neurones cachés → neurones de sortie → actuator
           ↑                                               ↓
           └──────── life_scape → life_world ──────────────┘
```

Le scape applique exactement les mêmes règles du monde. Après les trois mondes
11, 22, 33, il renvoie le score moyen ; le cortex termine alors l'évaluation.
ExoSelf garde ses mécanismes de sauvegarde, restauration et perturbation des
poids. On n'a pas réécrit son algorithme d'apprentissage. La topologie reste fixe.

`life_exoself:episode/2` rejoue le génotype avec les vrais acteurs, sans mutation,
et enregistre leurs sorties et les états du monde pour le navigateur.

### Un budget explicite

Les deux méthodes partagent la sélection initiale parmi 32 réseaux et partent
du même meilleur réseau. Ce coût de 32 évaluations est compté pour chacune.

- **Population** : 32 évaluations initiales + 40 × 32 = **1 312**.
- **ExoSelf** : 32 évaluations initiales + 1 280 évaluations de réglage = **1 312**.

Chaque évaluation parcourt trois mondes : **3 936 épisodes par méthode**.
La première évaluation du réglage ExoSelf réévalue le réseau de départ et compte
bien dans les 1 280. Pour cette comparaison, `max_attempts` est porté à 1 280 afin
que l'arrêt après 50 échecs consécutifs ne réduise pas son budget. La limite
habituelle reste disponible en dehors de cette expérience.

C'est une égalité du nombre d'évaluations, pas du temps de calcul ni du nombre
de messages. Les épisodes de validation sont exécutés après l'entraînement et
ne participent ni à la sélection ni à ce budget.

### 5. `src/life_demo.erl` : que voit le navigateur ?

Après la sélection, le programme, le meilleur réseau initial et les deux champions
jouent sur trois mondes inédits : seeds 101, 202, 303. Les épisodes et leurs
résultats sont intégrés au HTML. Le navigateur affiche les états reçus et les
scores de sortie ; il ne recalcule pas les décisions du réseau.

Le génotype ExoSelf sauvegardé peut être rejoué dans `rebar3 shell` :

```erlang
life_exoself:episode("_build/life/exoself.ets", 303).
```

Pour construire et entraîner directement un nouveau réseau avec les modules
du livre, sans lancer la comparaison :

```erlang
genotype:construct("food_network", life_mimic, [6], #{seed => 42}).
life_exoself:train("food_network",
                   #{seed => 42, evaluation_limit => 1000, max_attempts => 1000}).
life_exoself:episode("food_network", 303).
```

`train/2` attend le résultat de `exoself:map/2` et rassemble sa progression.
L'entraînement met à jour le fichier avec les meilleurs poids.

Le champion de la population peut aussi être évalué :

```erlang
{ok, [Saved]} = file:consult("_build/life/champion.term").
Brain = maps:get(champion, Saved).
life_world:evaluate(Brain, 303).
```

Pour une autre expérience, sans changer les fichiers sources :

```erlang
life_demo:export("/tmp/life-seed-7/index.html",
                 #{seed => 7, population => 32, generations => 40}).
```

Une seed reproduit une expérience sur le même runtime. Le panneau de résultats
est calculé à partir des données exportées, y compris si la nouvelle expérience
ne progresse pas. Les mondes de validation ne participent pas à la sélection.

## Résultat observé avec la seed 42

| Monde de validation | Programme | Réseau initial | Population | ExoSelf |
|---|---:|---:|---:|---:|
| 101 | 14 | 1 | 1 | 2 |
| 202 | 14 | 2 | 2 | 5 |
| 303 | 14 | 1 | 5 | 7 |
| Moyenne | 14 | 1,33 | 2,67 | 4,67 |

Ce sont des nourritures mangées sur 14. Le score d'entraînement part de 195,33
et atteint 586 pour la population, contre 1 024,67 pour ExoSelf. ExoSelf fait
mieux sur cette expérience, mais reste loin de la règle programmée. Une seule
seed d'entraînement et trois mondes de validation ne suffisent pas pour désigner
une méthode généralement supérieure.

Le score récompense la nourriture et la durée de vie, pas le chemin le plus
court. Une action contre un mur donne aussi un pas survécu et consomme de
l'énergie. Observer ces cas aide à comprendre ce que l'objectif récompense.
Une prochaine expérience pourrait changer une seule chose, par exemple les
perceptions ou la diversité des mondes d'entraînement, en conservant une
nouvelle série de mondes pour l'évaluation finale.

## Vérifier

### Capacité du réseau et qualité de l'apprentissage

Un test supplémentaire fixe les poids à la main pour reproduire la règle
programmée. Avec l'architecture actuelle 7 → 6 → 4 et `tanh`, les vrais acteurs
ExoSelf mangent **14/14 sur les 100 mondes 2001 à 2100**, avec les mêmes actions
que la règle. Ce réseau n'a pas été appris et ne participe pas à la comparaison
des méthodes. Il prouve que cette architecture peut représenter une solution ;
il ne prouve pas que l'optimiseur sait la trouver depuis des poids aléatoires.

Le réseau est le modèle ; l'entraînement est le processus d'ajustement de ses
poids. ExoSelf utilise une recherche aléatoire qui conserve une modification
seulement si le score augmente strictement. La référence Population sélectionne
et mute plusieurs candidats. Ces deux méthodes entraînent un réseau sans
calculer de gradient. La rétropropagation serait une autre manière de calculer
les informations nécessaires à un ajustement des poids, pour une perte dérivable.

La suite logique est de mesurer les tentatives inchangées, les égalités de score,
les améliorations et les régressions ; puis de comparer plusieurs démarrages sur
un ensemble de validation plus large. Toute modification de la perturbation doit
être comparée à budget égal. Les mondes utilisés pour choisir ces modifications
deviennent de la validation ; un autre ensemble devra servir au test final.

```sh
rebar3 eunit
```

Les tests couvrent les règles du monde, la terminaison, le déterminisme, le
réseau, les mutations, la sélection élitiste, les erreurs de workers et
l'export/rechargement. Ils vérifient aussi la concordance des acteurs avec le
calcul direct, la persistance du génotype ExoSelf, le budget commun et l'entrée
par `genotype:construct/4`. La suite XOR reste incluse. Les erreurs affichées
par les tests d'injection de panne XOR et scape sont attendues.

Validation : **23 tests réussis**. Les douze fins d’épisode ont été contrôlées
dans le navigateur et concordent avec le tableau. Le premier pas ExoSelf et le
retour au début fonctionnent ; aucune erreur console observée.
