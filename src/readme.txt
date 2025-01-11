Projet de Fouille d'opinions dans les commentaires - README.txt

1. Auteur
---------
[Votre Nom Complet]

2. Description du Classifieur
----------------------------
Le classifieur implémenté utilise l'approche PLMFT (Pretrained Language Model Fine-Tuning) basée sur le modèle CamemBERT, spécifiquement conçu pour le traitement du français. 

L'architecture comprend quatre classificateurs indépendants, un pour chaque aspect (Prix, Cuisine, Service, Ambiance), permettant une analyse fine des opinions.

Architecture et Paramètres :
- Modèle de base : CamemBERT (almanach/camembert-base)
- Taille maximale de séquence : 256 tokens
- Taille de batch : 32
- Taux d'apprentissage : 2e-5
- Optimiseur : AdamW avec gradient clipping
- Early stopping avec patience de 3 epochs
- Nombre maximum d'epochs : 5

Chaque classificateur est entraîné séparément pour prédire l'une des quatre classes (Positive, Négative, Neutre, NE) pour son aspect respectif. Le modèle utilise une architecture de classification standard avec une couche de classification au-dessus des embeddings contextuels de CamemBERT.

3. Exactitude sur les données de validation
-----------------------------------------
Prix:      XX.XX%
Cuisine:   XX.XX%
Service:   XX.XX%
Ambiance:  XX.XX%
Moyenne:   XX.XX%