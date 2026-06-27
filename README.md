# Projet de fin de module — Deep Learning

**EMSI Casablanca** · École Marocaine des Sciences de l'Ingénieur
Filière : AI & Data Science · Module : Deep Learning · Année universitaire 2025–2026
Auteure : **Kenza Ezzahed** (Groupe 1, site Moulay Youssef)

---

## Idée générale du projet

Ce repository contient le travail réalisé pour le **projet de fin de module Deep Learning**, conçu comme un examen final intégrateur individuel. L'objectif est de démontrer la capacité à **adapter le choix d'une architecture de deep learning à la nature des données traitées**, à travers trois cadres complémentaires :

1. **Données tabulaires** → Perceptron multicouche (MLP)
2. **Données images** → Réseau de neurones convolutionnel (CNN)
3. **Données séquentielles** → Architectures récurrentes (RNN / LSTM / GRU) et Seq2Seq

Chaque partie suit la même démarche : étude théorique des concepts, implémentation sous PyTorch, expérimentation comparative, analyse critique des résultats, et réponse à une question de synthèse sur un dataset réel. Le projet se conclut par une discussion scientifique transversale reliant les trois architectures selon la géométrie, la dépendance locale, la temporalité et la représentation des données.

## Structure du repository

```
.
├── Rapport.pdf     # Rapport scientifique du projet (rédigé à partir des 3 notebooks)
├── partie1_mlp.ipynb                  # Partie I — MLP sur données tabulaires
├── partie2_CNN_CIFAR10.ipynb          # Partie II — CNN sur données images
└── partie3_rnn_seq2seq.ipynb          # Partie III — RNN/LSTM/GRU et Seq2Seq sur données séquentielles
```

### `Rapport.pdf`
Rapport scientifique complet du projet, rédigé à partir du contenu et des résultats des trois notebooks ci-dessous : introduction, méthodologie, implémentation, résultats, interprétations, limites, conclusion et discussion scientifique transversale reliant les trois architectures.

### `partie1_mlp.ipynb` — MLP et ingénierie PyTorch
- **Dataset** : Breast Cancer Wisconsin (classification binaire, 30 features tabulaires)
- **Contenu** : prétraitement (normalisation, split train/val/test), implémentation d'un MLP en `nn.Sequential` et en classe personnalisée (`nn.Module`), inspection des paramètres (`named_parameters`, `state_dict`), comparaison de trois stratégies d'initialisation (gaussienne, constante, Xavier), entraînement avec sauvegarde/rechargement du meilleur modèle, évaluation (accuracy, precision, recall, F1, matrice de confusion), et réponse à la question de synthèse de la Partie I.

### `partie2_CNN_CIFAR10.ipynb` — CNN et vision par ordinateur
- **Dataset** : CIFAR-10 (10 classes, images 32×32 RGB)
- **Contenu** : justification théorique du passage du MLP au CNN (localité, partage des poids, hiérarchie des représentations), calculs manuels des dimensions de sortie (convolution/pooling), implémentation manuelle de la corrélation croisée 2D et du max/average pooling comparée aux couches PyTorch natives, architecture CNN inspirée de LeNet, étude expérimentale sur le pooling, le nombre de filtres et la convolution 1×1, comparaison MLP vs CNN sur le même dataset, visualisation de cartes de caractéristiques (feature maps), et réponse à la question de synthèse de la Partie II.

### `partie3_rnn_seq2seq.ipynb` — RNN, LSTM, GRU et Seq2Seq
- **Dataset** : Multi30k (traduction automatique anglais → allemand, ~29 000 paires de phrases)
- **Contenu** : fondements théoriques des modèles de langage (factorisation par règle de chaîne, perplexité, BPTT, gradient clipping), préparation des données (tokenisation, vocabulaires, tokens spéciaux, padding), implémentation et comparaison de RNN simple, LSTM et GRU dans une architecture encodeur-décodeur avec teacher forcing, illustration expérimentale de l'effet du gradient clipping, deux stratégies de décodage (glouton et beam search), évaluation via perplexité et score BLEU, et réponse à la question de synthèse de la Partie III.

## Exécution

Les notebooks ont été exécutés sur **Google Colab**, ce qui fournit gratuitement un environnement Python avec PyTorch préinstallé et un accès GPU pour accélérer l'entraînement (notamment pour les Parties II et III). Chaque notebook est autonome et peut être ouvert directement dans Colab (ou tout autre environnement Jupyter disposant de PyTorch, torchvision, scikit-learn, numpy, matplotlib et seaborn). Le device (CPU/GPU) est détecté automatiquement dans le code.
