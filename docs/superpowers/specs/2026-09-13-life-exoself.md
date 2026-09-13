# Relier Petite vie à ExoSelf

Demande approuvée : utiliser ExoSelf avec les mêmes règles, perceptions et mondes,
enregistrer les épisodes réels et comparer à budget d'évaluations égal.

- Ajouter la morphologie life_mimic (7 entrées, 4 sorties) et un scape adaptant
  life_world au protocole sense/action. Une évaluation parcourt les seeds
  d'entraînement 11/22/33, retourne leur score moyen, puis remet les mondes à zéro.
- Utiliser les vrais processus sensor/neuron/actuator/cortex/exoself. Conserver
  la perturbation et l'acceptation ExoSelf existantes. Les options du scape sont
  facultatives pour préserver le circuit XOR.
- Convertir le réseau 7→6→4 initial de la démo en génotype ETS sans changer ses
  poids ou l'ordre des sorties N/E/S/W. Enregistrer les épisodes de validation
  depuis le scape, à partir du génotype sauvegardé, sans réentraînement.
- Sélection initiale partagée : 32 évaluations. La population utilise encore
  40×32 évaluations. ExoSelf reçoit 1280 évaluations, première réévaluation
  comprise ; max_attempts est relevé au budget pour ne pas interrompre ce test
  comparatif après 50 échecs. Total attribué à chaque méthode : 1312 évaluations
  de trois épisodes chacune. Les différences de durée et de pas sont conservées.
- Ajouter un mode ExoSelf au replay existant, afficher budgets et courbes sur
  l'axe des évaluations, conserver la comparaison avec la population simplifiée.
  Aucune modification de récompense, de capteurs ou de topologie.
- Tests : parité des poids, observations, sorties et actions entre calcul pur
  et acteurs ; moyenne multi-mondes, compteur de cycles, reset entre évaluations,
  limites, déterminisme, sauvegarde/rechargement, erreur de scape et suite XOR.
