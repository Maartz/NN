# Atelier hors ligne : comprendre et expérimenter avec ExoSelf

But : comparer **une longue recherche** et **plusieurs départs indépendants**.
Garder le réseau, `tanh`, le score et les perturbations actuels pour isoler
l'effet des redémarrages. Le protocole ci-dessous utilise le vrai ExoSelf.

Les commentaires `ATELIER` dans les sources indiquent où lire et instrumenter.
Les compteurs détaillés sont un exercice à faire, pas une fonctionnalité déjà
implémentée. Les commandes de comparaison fonctionnent avec le code actuel.

## 1. Avant de partir

Dans un terminal, à la racine du dépôt :

```sh
cd /Users/maartz/Documents/erlang-projects/NN
rebar3 compile
rebar3 eunit
```

La suite contient 23 tests. Les messages d'erreur des tests qui provoquent
volontairement des pannes sont attendus ; regarder le bilan final.
Compiler une fois avec la connexion disponible prépare aussi les dépendances
de rebar3. Le shell ci-dessous utilise ensuite directement les modules locaux.

```sh
erl -pa _build/default/lib/nn/ebin
```

Toutes les prochaines commandes `erlang` vont dans CE shell. Chaque expression
se termine par un point. Les variables Erlang ne se réaffectent pas : si tu
recopies un bloc qui lie déjà `Dir` ou `Smoke`, utilise d'autres noms ou quitte
avec `q().` et ouvre un nouveau shell. Les fichiers sauvegardés restent présents.

La page `_build/life/index.html` s'ouvre directement sans réseau ni serveur.
Elle rejoue l'ancienne comparaison Population/ExoSelf ; l'atelier ci-dessous
enregistre une autre expérience et ne remplace pas les données de cette page.

## 2. Retrouver les pièces

Lire dans cet ordre, en cherchant `ATELIER` dans l'éditeur :

| Fichier | Question à laquelle il répond |
|---|---|
| `src/life_exoself.erl` : `train/2` | Comment lancer, attendre et sauvegarder un entraînement ? |
| `src/exoself.erl` : `loop/13` | Quand garder ou refuser un essai ? Quand s'arrêter ? |
| `src/neuron.erl` : `perturb_IPIdPs/1` | Quels coefficients changent, et de combien ? |
| `src/life_scape.erl` : `loop/7` | Quand les trois épisodes deviennent-ils une évaluation ? |
| `src/life_brain.erl` : `action/1` | Pourquoi des poids différents peuvent-ils donner la même action ? |

**Modèle** = réseau qui transforme les perceptions en scores de direction.
**Entraînement** = recherche des poids. **Évaluation** = mesure d'un jeu de poids
sur tous les mondes d'entraînement. **Épisode** = un monde parcouru.
**Cycle** = un pas de la créature. Les poids restent fixes pendant l'évaluation.

Deux tirages se succèdent : ExoSelf choisit des neurones, puis chaque neurone
choisit des coefficients. Les deux tirages peuvent ne sélectionner aucun élément.
Une modification des poids peut aussi ne changer aucune décision.

## 3. Première manipulation : entraîner, sauvegarder, rejouer

Copier ce bloc dans le shell Erlang :

```erlang
Dir = filename:join("_build", "atelier-" ++ integer_to_list(erlang:system_time(microsecond))).
File = filename:join(Dir, "premier.ets").
ok = filelib:ensure_dir(File).
genotype:construct(File, life_mimic, [6], #{seed => 42}).
R = life_exoself:train(File, #{seed => 1042, evaluation_limit => 100, max_attempts => 100}).
maps:with([score, evaluations, cycles], R).
Episode = life_exoself:episode(File, 3001).
maps:with([eaten, steps, status], Episode).
```

- La graine de `construct/4` choisit les **poids initiaux**.
- La graine de `train/2` choisit les **perturbations**.
- La graine de `episode/2` choisit le **monde**.
- `[6]` est la couche cachée ; `life_mimic` fournit les 7 entrées et 4 sorties.
- `max_attempts => 100` évite que 50 échecs consécutifs arrêtent cet essai avant
  ses 100 évaluations. La première mesure compte dans le budget.
- Le fichier contient les meilleurs poids à la fin. Rappeler `train/2` dessus
  reprend depuis ces poids. Pour repartir de zéro, construire un nouveau fichier.

`score` est le score d'entraînement, pas le nombre de nourritures mangées.
Le score est la moyenne de `100 * eaten + steps` sur les mondes 11, 22, 33.
Il récompense aussi les pas, y compris ceux contre un mur : ne pas le confondre
avec la qualité du déplacement. Regarder `eaten` et `status` en validation.

## 4. Comparaison reproductible : un long essai contre plusieurs courts

Ce bloc définit une fonction dans le shell. Il ne lance pas encore la recherche.
Chaque appel crée un dossier distinct ; les fichiers de la démo sont préservés.
Les mondes 3001 à 3100 servent à la validation. Ils ne choisissent jamais le
gagnant des redémarrages. Celui-ci est choisi uniquement sur son score train.

```erlang
Compare = fun(BaseSeed, TotalBudget, Restarts) ->
    true = TotalBudget >= Restarts andalso Restarts > 0 andalso TotalBudget rem Restarts =:= 0,
    Folder = filename:join("_build", "restarts-" ++ integer_to_list(erlang:system_time(microsecond))),
    ok = filelib:ensure_dir(filename:join(Folder, "results.term")),
    TrainSeeds = [11,22,33],
    ValidationSeeds = lists:seq(3001,3100),
    Run = fun(Label, InitSeed, Budget) ->
        Path = filename:join(Folder, Label ++ ".ets"),
        genotype:construct(Path, life_mimic, [6], #{seed => InitSeed}),
        Result = life_exoself:train(Path,
            #{seed => InitSeed + 100000, evaluation_limit => Budget,
              max_attempts => Budget,
              scape_options => #{life_sim => #{seeds => TrainSeeds}}}),
        Budget = maps:get(evaluations, Result),
        Result#{file => Path, initial_seed => InitSeed}
    end,
    Long = Run("long", BaseSeed, TotalBudget),
    ShortRuns = [Run("short-" ++ integer_to_list(I), BaseSeed + I, TotalBudget div Restarts)
                 || I <- lists:seq(0, Restarts-1)],
    TotalBudget = lists:sum([maps:get(evaluations, X) || X <- ShortRuns]),
    BestShort = lists:foldl(fun(Candidate, Best) ->
        case maps:get(score, Candidate) > maps:get(score, Best) of
            true -> Candidate;
            false -> Best
        end
    end, hd(ShortRuns), tl(ShortRuns)),
    Measure = fun(Path) ->
        Episodes = [maps:without([frames], life_exoself:episode(Path, S)) || S <- ValidationSeeds],
        Foods = [maps:get(eaten, E) || E <- Episodes],
        #{worlds => length(Episodes), mean_food => lists:sum(Foods)/length(Foods),
          minimum_food => lists:min(Foods),
          cleared => length([ok || #{status := cleared} <- Episodes])}
    end,
    Summary = #{base_seed => BaseSeed, evaluations_per_method => TotalBudget,
                long => Measure(maps:get(file, Long)),
                restarts => Measure(maps:get(file, BestShort))},
    Saved = #{summary => Summary, training_seeds => TrainSeeds,
              validation_seeds => ValidationSeeds, long => Long,
              short_runs => ShortRuns, best_short_file => maps:get(file, BestShort)},
    Output = filename:join(Folder, "results.term"),
    ok = file:write_file(Output, io_lib:format("~p.~n", [Saved])),
    Summary#{results_file => Output}
end.
```

Le long essai et le premier court partent des mêmes poids et de la même graine
de perturbation. Les autres courts ont chacun un départ différent. Le budget
inclut la première mesure de chaque réseau ; aucun tri initial gratuit.
À 10 000 évaluations par méthode et 3 mondes, chacune consomme 30 000 épisodes
d'entraînement. La validation est en plus. Le temps de calcul peut différer.

D'abord un **petit contrôle du protocole**, qui ne mesure pas la performance :

```erlang
Smoke = Compare(42, 100, 10).
```

Ensuite la vraie expérience, sur trois graines de départ :

```erlang
Results = [Compare(S, 10000, 10) || S <- [42,123,777]].
```

Cela lance 60 000 évaluations au total. Le shell attend pendant le calcul.
Tu peux commencer par `Compare(42, 10000, 10).` pour une seule comparaison.
Les trois répétitions sont un premier diagnostic, pas une conclusion générale.

Chaque dossier contient les 11 génotypes et `results.term` avec les courbes et
les paramètres. Pour relire le petit essai :

```erlang
{ok, [SavedSmoke]} = file:consult(maps:get(results_file, Smoke)).
maps:get(summary, SavedSmoke).
```

Regarder `mean_food`, `minimum_food` et surtout `cleared` sur 100 mondes.
Si les courts gagnent régulièrement, explorer les redémarrages devient une piste.
Si le long gagne, prolonger une trajectoire paraît utile dans ce protocole.
Si les deux restent faibles, il faudra examiner les mutations et le retour du
score. Ne changer qu'un facteur à la fois pour pouvoir expliquer le résultat.
Les mondes de validation utilisés pour ces décisions ne sont plus un test final :
réserver par exemple 4001 à 4100 pour après le choix des réglages.

## 5. Exercice : rendre les essais refusés visibles

Dans `src/exoself.erl`, juste avant `case Fitness > HighestFitness`, ajouter :

```erlang
Outcome = case EvalAcc of
    0 -> initial;
    _ when Fitness > HighestFitness -> improved;
    _ when Fitness == HighestFitness -> equal;
    _ -> worse
end,
```

Puis enrichir la map du message `progress` existant avec :

```erlang
#{evaluations => U_Evals, score => U_HighestFitness,
  candidate_score => Fitness, outcome => Outcome}
```

Garder le tuple du message `{self(), progress, Map}`. `life_exoself:await/3`
conserve déjà la map entière, donc les nouveaux champs seront dans `history`.
Ces fragments s'insèrent dans la fonction : ne pas les coller au shell seuls.
Ne pas modifier la sélection, la restauration ou les tirages aléatoires ici.

Après la modification, quitter le shell, exécuter `rebar3 eunit` puis
`rebar3 compile`, rouvrir `erl -pa _build/default/lib/nn/ebin` et refaire le
petit exemple de la section 3. Cela évite de tester un ancien module en mémoire.
On peut ensuite compter les résultats :

```erlang
History = maps:get(history, R).
Counts = maps:from_list([{Kind, length([ok || #{outcome := K} <- History, K =:= Kind])}
                        || Kind <- [initial, improved, equal, worse]]).
true = lists:sum(maps:values(Counts)) =:= maps:get(evaluations, R).
1 = maps:get(initial, Counts).
```

Pour cet exemple : 1 mesure initiale + 99 essais classés = 100 évaluations.
Ajouter ces invariants au test d'entraînement dans `test/life_exoself_tests.erl`.
À graine identique, l'instrumentation doit conserver les mêmes poids finaux et
la même courbe de meilleurs scores que le code avant modification.

**Attention au sens des compteurs :** `equal` signifie score égal. Pour compter
les poids inchangés, il faut comparer avant/après mutation dans `neuron.erl`,
puis transmettre l'information à ExoSelf en conservant son protocole d'ACK.
Pour compter les comportements inchangés, il faut comparer les actions sur les
mêmes mondes. Ce sont trois mesures différentes. Commencer par le score.

## 6. Si quelque chose bloque

- `undef` : mauvais répertoire, compilation absente ou ancien module chargé.
- `badmatch` après avoir recopié un bloc : variable déjà liée dans le shell.
- Moins d'évaluations que prévu : contrôler `max_attempts` et `fitness_target`.
- Courbe plate : l'historique actuel montre le meilleur score, pas tous les essais.
- Un fichier entraîné n'est pas du texte : `.ets` se lit avec `life_exoself:load/1` ;
  `results.term` se lit avec `file:consult/1`.
- Le replay ne change pas après l'expérience : normal, ce protocole ne régénère
  pas la page. Les résultats à lire ici sont ceux du shell et de `results.term`.

Le contrôle `handwired_capacity_test_` dans `test/life_exoself_tests.erl` garde
une solution à poids manuels qui réussit sur 100 mondes. Il valide la capacité
du réseau, pas l'apprentissage, et ne doit pas servir de départ à cette expérience.
