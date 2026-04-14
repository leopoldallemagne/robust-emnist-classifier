# LEPL1507_Groupe_2


## Structure du projet

Le dossier `src/` contient les fichiers principaux du projet :

### `classifier_S4_group_02.py`

Ce fichier contient le modèle de classification principal utilisé par le projet.

- Définit la classe `ClassifierS4Group02` basée sur un réseau convolutif simple.
- Utilise deux blocs convolutionnels avec batch normalization, ReLU et max pooling.
- Ajoute une partie fully connected avec des couches linéaires, ReLU et dropout.
- Fourni une méthode `forward` qui retourne des probabilités normalisées par softmax.
- Fourni une méthode `forward_logits` pour l'entraînement avec `CrossEntropyLoss`.
- Le script principal charge le dataset `EMNIST`, entraînement le modèle, affiche la progression, et sauvegarde un modèle scripté en `classifier_S4_group_02.pt`.

### `TrainingWithMetrics.py`

Ce fichier permet d'exécuter des sessions d'entraînement sur des métriques.

- Définit `training(init_metrics, subject, metrics_file, oracle_file, dir)` pour lancer des expériences modulaires.
- Charge soit le dataset EMNIST pur, soit un dataset mixte comprenant des exemples adversariaux si `dataset_file` n'est pas `EMNIST`.
- Entraîne `ClassifierS4Group02` et sauvegarde à la fois les métriques d'entraînement/test et les oracles PyTorch (`.pt`).
- Produit des fichiers CSV de métriques et des fichiers `.pt` scripts qui peuvent être utilisés comme oracles.

### `TrainingWithSchedulers.py`

Ce fichier effectue des expériences axées sur le réglage du scheduler et du taux d'apprentissage.

- Définit `run_experiment(config, save_path)` pour lancer un'entraînement unique selon une configuration.
- Teste différents learning rates et différents schedulers (`None`, `StepLR`, `Plateau`).
- Sauvegarde les métriques par expérience dans `results/*.csv`.
- Permet de comparer l'impact des hyperparamètres et du scheduler sur l'accuracy de test.

### `src/graphs_plot/`

Le dossier `src/graphs_plot/`, fichiers pour générer des graphes

- `MetricsGraph.py`
  - Charge un fichier CSV de métriques d'entraînement et trace les courbes de loss et d'accuracy pour chaque epoch.
  - Affiche aussi les hyperparamètres (`learning_rate`, `batch_size`, `weight_decay`, `dropout`) dans la légende.


- `scheduler_graph.py`
  - Parcourt tous les CSV dans `results/` et compare les courbes `test_acc` par epoch pour différents schedulers.


- `batch_size_accuracy.py`
  - Trace la précision (`test_acc`) et le temps total en fonction de la taille de batch.




