# LEPL1507_Groupe_2

## Structure du projet

Le dépôt contient les fichiers principaux du projet :

### `classifier_S4_group_02.py`

Ce fichier contient le modèle de classification principal utilisé par le projet.

- Définit la classe `ClassifierS4Group02` basée sur un réseau convolutif simple.
- Utilise deux blocs convolutionnels avec batch normalization, ReLU et max pooling.
- Ajoute une partie fully connected avec des couches linéaires, ReLU et dropout.
- Fournit une méthode `forward` qui retourne des probabilités normalisées par softmax.
- Fournit une méthode `forward_logits` pour l'entraînement avec `CrossEntropyLoss`.
- Le script principal charge le dataset `EMNIST`, entraîne le modèle, affiche la progression, et sauvegarde un modèle scripté en `classifier_S4_group_02.pt`.

### `classifier_MLP.py`

Ce fichier contient notre classifieur perceptron multicouche (MLP).

- Définit la classe `ClassifierS4Group02` sous forme de réseau fully connected.
- Aplati les images EMNIST (28x28) en vecteurs de taille 784 avant traitement.
- Utilise trois couches linéaires (784->512->512->47), avec batch normalization, ReLU et dropout (p=0.1) sur les couches cachées.
- Fournit une méthode `forward` qui gère automatiquement plusieurs formats d'entrée ((28,28), (B,28,28), (B,1,28,28)) et retourne des probabilités via softmax.
- Le script principal entraîne le modèle sur EMNIST balanced avec Adam, NLLLoss (appliquée sur log-probabilités), et un scheduler StepLR.
- Évalue le modèle sur l'ensemble de validation à chaque époque, sauvegarde le meilleur checkpoint, puis exporte un modèle TorchScript en `classifier_S4_group_02_MLP_512_512.pt`.

### `TrainingWithMetrics.py`

Ce fichier permet d'exécuter des sessions d'entraînement et d'enregistrer des métriques.

- Définit `training(init_metrics, subject, metrics_file, oracle_file, dir)` pour lancer des expériences modulaires.
- Charge soit le dataset EMNIST pur, soit un dataset mixte comprenant des exemples adversariaux si `dataset_file` n'est pas `EMNIST`.
- Entraîne `ClassifierS4Group02` et sauvegarde à la fois les métriques d'entraînement/test et les oracles PyTorch (`.pt`).
- Produit des fichiers CSV de métriques et des fichiers `.pt` scriptés qui peuvent être utilisés comme oracles.

### `TrainingWithSchedulers.py`

Ce fichier effectue des expériences axées sur le réglage du scheduler et du taux d'apprentissage.

- Définit `run_experiment(config, save_path)` pour lancer un entraînement unique selon une configuration.
- Teste différents learning rates et différents schedulers (`None`, `StepLR`, `ReduceLROnPlateau`).
- Sauvegarde les métriques par expérience dans `results/*.csv`.
- Permet de comparer l'impact des hyperparamètres et du scheduler sur l'accuracy de test.

### `attacker_group_02.py`

Ce fichier implémente les méthodes d'attaque adversariale utilisées pour générer des perturbations sur les images EMNIST.

- Contient la fonction principale `attack(x, f_string, eps)` qui génère une perturbation `delta` sous contrainte L2 (||delta||_2 <= eps) et renvoie l'image adversariale.
- Implémente plusieurs variantes (modes) et redémarrages (restarts), des pertes dédiées (par ex. DLR), ainsi que des stratégies d'optimisation avec momentum, normalisation et projection sur la contrainte.
- Comprend des utilitaires pour préparer l'entrée, charger le modèle, et visualiser la perturbation (heatmap) ainsi que des fonctions d'évaluation (ex. `plot_accuracy_vs_epsilon`).
- Contient aussi du code de test/visualisation (commenté) servant à des essais rapides.

### `graphs_plot/`

Le dossier `graphs_plot/` contient des scripts pour générer des graphes :

- `MetricsGraph.py`
  - Charge un fichier CSV de métriques d'entraînement et trace les courbes de loss et d'accuracy pour chaque epoch.
  - Affiche aussi les hyperparamètres (`learning_rate`, `batch_size`, `weight_decay`, `dropout`) dans la légende.

- `scheduler_graph.py`
  - Parcourt tous les CSV dans `results/` et compare les courbes `test_acc` par epoch pour différents schedulers.

- `batch_size_accuracy.py`
  - Trace la précision (`test_acc`) et le temps total en fonction de la taille de batch.
