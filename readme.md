# Projet de Fouille

## Auteur
- **LU Yuxuan**
- **ESPINAL Miguelangel**

## Description du Classifieur
Le classifieur implémenté utilise l'approche **PLMFT** (Pretrained Language Model Fine-Tuning) basée sur le modèle **CamemBERT**, spécifiquement conçu pour le traitement du français.

L'architecture comprend quatre classificateurs indépendants, un pour chaque aspect (Prix, Cuisine, Service, Ambiance), permettant une analyse fine des opinions.

### Architecture et Paramètres
- **Modèle de base** : CamemBERT (almanach/camembert-base)
- **Taille maximale de séquence** : 256 tokens
- **Taille de batch** : 32
- **Taux d'apprentissage** : 2e-5
- **Optimiseur** : AdamW avec gradient clipping
- **Early stopping** : patience de 3 epochs
- **Nombre maximum d'epochs** : 5

Chaque classificateur est entraîné séparément pour prédire l'une des quatre classes (Positive, Négative, Neutre, NE) pour son aspect respectif. Le modèle utilise une architecture de classification standard avec une couche de classification au-dessus des embeddings contextuels de CamemBERT.

## Exactitude sur les données de validation
- **n=1**
- **Validation accuracies**:
 - **Prix**: 86.54%
 - **Cuisine**: 87.48%
 - **Service**: 87.17%
 - **Ambiance**: 82.00%
- **Average accuracy**: 85.80%

### n=2
- **Validation accuracies**:
  - **Prix**: 86.07%
  - **Cuisine**: 87.01%
  - **Service**: 87.48%
  - **Ambiance**: 80.75%
- **Average accuracy**: 85.33%

### n=3
- **Validation accuracies**:
  - **Prix**: 86.23%
  - **Cuisine**: 86.85%
  - **Service**: 87.79%
  - **Ambiance**: 80.75%
- **Average accuracy**: 85.41%

### n=4
- **Validation accuracies**:
  - **Prix**: 85.76%
  - **Cuisine**: 87.64%
  - **Service**: 87.01%
  - **Ambiance**: 80.59%
- **Average accuracy**: 85.25%

### n=5
- **Validation accuracies**:
  - **Prix**: 86.23%
  - **Cuisine**: 87.79%
  - **Service**: 87.17%
  - **Ambiance**: 81.22%
- **Average accuracy**: 85.60%

## Résultats globaux
- **ALL RUNS ACC**: [84.95, 84.51, 83.87, 83.87, 84.06]
- **AVG MACRO ACC**: 84.25