# Détermination du genre des films
## Projet Machine Learning - Benjamin ROUL

Auteur : François LEBRUMENT et Emma WEISS

# **Introduction :**
Notre projet consiste à classifier le genre des films à partir de plusieurs informations relatives au film.
A partir d'une base de données récupérée sur Kaggle. Cette base provient d'un site de critique de film. Elle contient pour chaque film des données relatives au film comme la date de sortie, le réalisateur, le casting, la description et le genre. De plus, pour chacun des films, nous avons à disposition la moyenne des notes des critiques, des internautes et des notes séparées par thème l'humour, le rythme, la tension ou l'érotisme.
La base initiale contient environ 38 000 films répertoriées pour 19 variables.

Notre objectif est de prédire le genre du film grâce à l'exploitation d'un maximum d'information à notre disposition. L'idée derrière ce projet, c'est par des méthodes de Machine Learning, de faciliter l'affectation d'un genre du film pour des entreprises répertoriant des films (Allociné, Netflix, AmazonPrime...). Nous nous sommes limité à la détermination de 2 genres : la comédie et le drame.
Nous avons utilisé les méthodes de Machine Learing suivante que nous avons optimisé : Random Forest, SVM et Réseau de neurones.







I. Importation et nettoyage des données
A. Source et format de la base
1. Provenance de la base
Ce projet se base sur la base Kaggle de STEFANO LEONE nommée FilmTV movies dataset (https://www.kaggle.com/datasets/stefanoleone992/filmtv-movies-dataset)

2. Description des variables (Emma et François sépraration par variables) 

B. Nettoyage des données (Emma)
1. Traitement des NA
2. Répartition du genre du film
3. Traitement des données textuelles (François)

C. Création d'indicateurs (François)
1. Indicateur du continent de production du film
2. Indicateur de sentiments de la description
3. Indicateur d'expérience précédente du casting et du réalisateur

D. Traitement des valeurs atypiques (Déjà faite, à améliorer)
1. Potentiel valeurs atypiques
2. Modification de la base
3. Choix des variables finales

II. Statistiques descriptives
A. Statistiques univariées (Emma)
1. Variable à expliquer
2. Variables explicatives

B. Statistiques bivariées (Emma)
1. Représentation graphique
2. Matrice des corrélations

C. Traitement des variables corrélées (François)
1. ACP
2. Sélection des variables


III. Modélisation
A. Modification de la base
1. Découpage du dataset
2. Standardisation

B. Modèles Random Forest (François)
1. Arbre de décision
2. Random Forest Classique
3. Bagging
4. Boosting : XGBoost
5. Boosting : LightGBM

C. Modèles SVM (Emma)
1. Régression logistique
2. SVC linéaire
3. SVC kernel linéaire
4. SVC kernel polynomial
5. SVC kernel rbf
6. SGD Classifier

D. Modèles Réseaux de neurones
1.


E. Sélection du meilleur modèle
1. Meilleur modèle Random Forest
2. Meilleur modèle SVM
3. Meilleur modèle Réseau de neurones
4. Meilleur modèle final

Conclusion et discussion
