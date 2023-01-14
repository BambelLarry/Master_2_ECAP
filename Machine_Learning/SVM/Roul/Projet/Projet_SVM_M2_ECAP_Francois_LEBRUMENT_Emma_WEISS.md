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
3. Traitement des données textuelles (François) <br>

Plusieurs variables de notre jeu de données sont sous format texte. C'est le cas du titre (title), du résumé (description), du pays (country), des réalisateurs (directors) et des acteurs (actors). Pour les exploiter, il est nécessaire de séparer les termes qui compose la donnée. Nous avons utilisé la technique de tokenization de la méthode Natural Language Processing (NLP). La tokenization consiste à séparer le texte brut en terme (aussi appelé token). Chaque mot et ponctuation sont séparés pour devenir des élements distincts au sein d'une liste Python. Cela nous permet de travailler sur chaque terme de la liste indépendamment.
La tokenization a été utilisé pour les variables title et description, afin de mettre en place un indicateur de sentiment.

Pour les 3 autres variables, nous ne sommes pas face à des phrases que nous devons séparer, mais à des listes de noms propres. En effet, ces variables peuvent avoir plus de 1 élement (exemple : plus d'un pays de production). Donc même si une majorité des pays et des réalisateurs sont seuls au sein d'une observation, nous devons les convertir en liste Python.
Pour traiter ces variables nous devons séparer les noms propres pour qu'ils deviennent des élements distinct dans une liste Python. Néanmoins, nous ne pouvons pas avoir recours à la tokenization. En effet, ici nous ne souhaitons pas séparer terme par terme, mais noms propres par noms propres. Par exemple, le nom d'un acteur est composé d'un prénom et d'un nom. Cela n'aurait pas de sens de séparer ces 2 termes distinctements.
Dans notre jeu de données, les noms propres de ces variables sont séparés par des virgules. Nous avons découpé chaque élément de la liste via le séparateur ", ".

Les 5 variables textuelles sont maintenant convertis sous forme de liste. Nous pouvons à présent les exploiter pour créer de nouveaux indicateurs.

C. Création d'indicateurs (François)

Nous avons séparé les variables textuelles pour pouvoir les exploiter.
A noter que nous n'avons pas réaliser de nettoyage des données textuelles. Nous n'avons pas retiré de stopwords (mots vides tel que "alors", "le" ou "elle"), retiré la ponctuation, retiré des caractères spéciaux tel que les accents ou des erreurs de saisies comme "©" appraissant plusieurs fois. Nous n'avons pas considéré cela comme pertinent pour les indicateurs que nous souhaitons créer.


1. Indicateur du continent de production du film

Le premier indicateur correspond au continent de production du film basé sur la variable country. Nous pouvons imaginer que certains pays ou continent produisent un genre plus qu'un autre. Par exemple, l'Inde est réputée, via Bollywood, pour créer des films musicaux. Nous supposons que certains pays produisent plus de comédie que de drame et inversement.

Nous avons remarqué que la distribution des films n'étaient pas équlibré. Les Etats-Unis et l'Italie concentrent plus de la moitié des pays producteurs de notre base (59,69 % du dataset au moment du traitement). Nous avons préféré les regrouper par continents.

De plus, certains films ont plusieurs pays producteurs, dont certains de plusieurs continents différents. Pour facilité le traitement, nous avons considéré que ces films étaient "internationaux". Certains films produits par plusieurs pays européens par exemple, sont alors considéré comme internationaux.

![image](https://user-images.githubusercontent.com/117921986/212486192-ee40d430-5449-4d1b-a95d-ef3f63c73eea.png)

Pour le traitement des pays nous avons utilisé les librairies pycountry et pycountry_convert. Le premier nous donne la liste des pays du monde (pas complète). La seconde convertie le nom du pays en continent.

Voici la liste des traitements pour convertir le nom de pays en continent :
1. Nous avons converti les pays multi-nationaux en "internationaux".
2. Nous avons converti les pays existant dans la liste countries de pycountry en continent.
3. Pour les pays n'existant pas dans la liste countries nous les avons en "autre". Cette liste n'est pas complète pour certains pays existant comme "Great Britain", "Russia" ou "Czech Republic". Ces derniers peuvent être converti en continent donc nous avons complété la liste puis converti ces pays.
4. Pour les pays ne pouvant pas être converti nous les avons directement affecté au contient. Il s'agit d'ancien pays comme "Soviet Union", "Czechoslovakia" ou "East Germany". Nous les avons tous affecté en Europe, sauf pour "Soviet Union" que nous avons considéré en Asie.

La distribution des continents est la suivante :

![image](https://user-images.githubusercontent.com/117921986/212487318-debcbeb1-5df3-47ee-afa9-78173324916c.png)

Encore une fois, la distribution des contients est déséquilibrés. Pour palier à ce déséquilibre et ne pas avoir trop de modalité, nous avons regroupé l'ensemble des contient hors Europe et Amérique du Nord. La distribution est à présent la suivante : 

![image](https://user-images.githubusercontent.com/117921986/212487401-5629dac7-2950-434c-a638-b8fa161ae3b7.png)

Pour rendre exploitable la variable du continent dans nos modèles, nous l'avons transformée en 3 variables muettes. Il s'agit de variables binaires. La variable muette "EU" est créée pour les films européens, "NA" pour les films nord américains et "Inter" pour les films internationaux (et des autres continents). Evidemment, chaque film possède un 1 dans une des trois variables et deux 0 dans les deux autres (un film étant produit que sur un continent).

2. Indicateur de sentiments de la description

Nous supposons que la description des films drame sont plus négative que celle des comédies
Le résumé du film ("description") est une forme abrégée du contenu du film. Il nous permet de connaitre dans les grandes lignes ce qui s'y déroule. Il s'agit d'un élément important prédéfinissant en amont le genre du film. C'est la raison pour laquelle nous avons créé un indicateur de sentiments sur cette variable.

Nous avons utilisé l'approche lexical de la méthode NLP pour créer l'indicateur de sentiment. Cette approche se base sur une liste de mots de références créer manuellement. Ici, nous avons utilisé la liste de référence Opinion Lexicon de la librairie nltk.corpus. Il s'agit de deux listes de mots : une composée de mots positifs et l'autres de mots négatifs. Nous comptons simplement le nombre de mots en communs de ces listes avec les termes du résumé du film.

A partir de cette comptabilisation, nous avons créé l'indicateur du ton ("ton_global") représentant le sentiment dégagé (positif ou négatif) suite à la lecture du texte. Ce ton est calculé de la manière suivante :

![image](https://user-images.githubusercontent.com/117921986/212488139-42c475fb-756e-42a4-8d0d-6f3c36299b06.png)
Cet indicateur se situe entre -1 et 1. Moins il est proche de 0, moins le ton est équilibré. Si l'indicateur est négatif, alors le ton est négatif (plus de mots négatifs que positifs) et inversement.


Toutefois, la création de cet indicateur possède quelques limites.
D'abord, il ne prend pas en compte la négation ("not") inversant la polarité d'un terme. Par exemple, "mal" est négatif, mais précédé d'une négation ("pas mal") devient positif. La liste de mots "Lexicoder Sentiment Dictionary" la prend en compte, mais est trop coûteuse pour nous de la mettre en place.
Ensuite, l'indicateur ne remet pas les mots dans leur contexte. En effet, nous nous sommes basé sur les listes Opinions Lexicons définissant la polarité des termes de manières strictes. Il n'y a pas de nuance en fonction du contexte ou du sens du mot. Par exemple, "licencié" possède 2 sens : l'un c'est être titulaire d'une licence universitaire ou sportive (positif), l'autre c'est être renvoyé de son emploi (négatif). Bien que les listes de mots et les résumés soit en anglais, certains peuvent avoir plusieurs sens. Opinion Lexicon est une liste basé sur le sens commun des mots, mais ne créé pas de nuance comme le "SentiWordNet".
Puis, nous ne prenons pas en compte l'orthographe des mots. Surtout que nous n'avons pas réalisé de nettoyage. En effet, faire un nettoyage des résumés prendrait trop de temps. Nous avons alors considéré qu'ils étaient bien écrit et que les mots du textes concordent avec ceux de la liste de mots. Toutefois, Opinion Lexicon propose pour quelques exceptions des orthographes de mots différentes.
Enfin, les mots avec une polarité (positive ou négative) dans un texte sont largement minoritaires. Ils ne consituent même pas 5 % d'un texte. Donc, la notion de ton d'un texte ne représente pas sont intégralité.

Malgré ces limites de notre indicateurs de sentiments basé sur l'Opinion Lexicon nous donne un aperçu global du ton du texte.
Nous supposons que les drames ont principalement des résumés considérés comme négatif et les comédies des résumés positifs.


4. Indicateur d'expérience précédente du casting et du réalisateur
Nous pouvons remarqué que certains acteurs et réalisateurs sont associés à certains types de films. Par exemple, l'acteur Jim Carrey est connu pour ses premiers rôles dans des comédies comme "Ace Ventura", "The Truman Show" ou "The Mask". Grâce à l'expérience du casting et du réalisateur, nous pourrions mieux supposer le genre du film. C'est pour cela que nous avons créé l'indicateur de la part de comédie dans l'expérience du casting sur l'ensemble des films (comédie+tragédie) dans lesquels les acteurs ont joué.

Nous avons créé ces indicateurs séparemment pour les et les réalisateurs, même si le processus est le même. Nous allons présenter l'indicateur pour les acteurs.

Au départ, nous avions comptabilisé le nombre de comédie et de drame dans lesquels chaque acteur avait joué, indépendamment de la date de sortie du film. Puis, nous sommions le nombre de comédie/drame de l'ensemble des acteurs du casting pour créer l'indicateur d'expérience. Néanmoins, cela signifie que pour un film sortie en 1960 par exemple, nous comptabilisions le nombre de comédie et de drame du casting sortie après. De plus, nous nous posions la question si fallait prendre en compte le film de 1960 dans la somme du casting.
Par ailleurs, cet indicateur se basait sur notre variable à expliquer. Comme nous sommes supposer la prédire, nous avions imaginé de créer cet indicateur que sur une base Train. Donc, que pour la base Test, l'indicateur se baserait sur l'échantillon Test.
En outre, nous nous demandions s'il fallait pondéré le nombre de comédie/drame par le nombre d'acteur dans le casting. En effet, un casting nombreux sur un film augmente logiquement l'expérience global.

Cette première approche contenait plusieurs problèmes, questionnements et limites. Suite à des discussion avec @Roulitoo, nous avons modifié notre approche pour revenir à un indicateur plus simple : l'expérience précédente du casting dans les comédies et drames. Cela suppose donc que pour le film sortie en 1960, nous ne prenons en compte que l'expérience des acteurs avant la sortie du dit film.

Pour cela, nous avons créé une base de données des acteurs avec pour chaque observation l'acteur et le film dans lequel il a joué. Nous avons aussi ajouté l'année de sortie du film. Puis, pour chaque couple acteur-film, nous avons compté le nombre de comédie et drame dans lequel l'acteur a joué avant l'année de sortie du film. Le principe fonctionne un peu comme un filtre : d'abord on filtre sur l'acteur, puis on filtre sur la date de sortie du film. Voilà comment cette base des acteurs est représenté pour un acteur, ici "Kim Rossi Stuart".

![image](https://user-images.githubusercontent.com/117921986/212498840-f00d5fec-ab1f-46b6-a144-f1753f4f6aff.png)

Ensuite, nous avons simplement fusionné notre base des acteurs avec celle des films en sommant pour l'ensemble du casting. Nous obtenons pour chaque film le nombre de comédie et drame (séparément) dans lequel a joué le casting avant.

Enfin, nous avons calculé l'indicateur de la part de comédie d'expérience sur l'ensemble des films d'expérience (comédie et drame). Cet indicateur nous permet d'éviter le problème d'inégalité entre des films avec des gros casting et ceux avec des casting peu connu avec peu d'expérience. En effet, cet indicateur varie entre 0 si le casting n'a que des expériences dans les drames et 1 si il n'a que des expériences dans les comédies. Cet indicateur est vue comme une jauge avec pour milieu 0.5. Cela signifie que le casting a autant d'expérience dans les comédies que dans les drames.
Cependant, il peut arriver pour certains films que le casting n'ait aucune expérience. Dans ces cas là, nous les affectons à 0.5, le juste équilibre.

De plus, nous avons ajouté dans la base l'expérience total du casting (somme du nombre de comédie et de drame dans lequel a joué le casting).



Toutefois, bien que nous implémentons ces indicateurs dans la base, il possède tout de même quelques limites :
-Il est probable que certains acteurs aient le même nom et prénom. Ils peuvent fausser l'indicateur, mais nous ne pouvons rien y faire (hormis vérifier l'ensemble des films à la main)
-Comme nous n'avons pas la date précise de sortie du film, seulement l'année. Donc, l'indicateur exclu les films sortie la même année (même s'ils sont sortis à une date antérieur). Cela a probablement peu d'impact sur notre indicateur au final.
-L'expérience du casting se base seulement sur les films présents dans la base et non sur l'ensemble des films de la carrière des acteurs. 
-La non expérience de certain casting fait que 0.5 devient la classe modale de la variable. Cela est plus flagrant sur la distribution pour les producteurs.

Distribution de l'indicateur de la part de comédie dans l'expérience du casting, et l'expérience du casting :


![image](https://user-images.githubusercontent.com/117921986/212499991-064628fd-bc6a-45f8-b7bd-92b565da807e.png)
![image](https://user-images.githubusercontent.com/117921986/212500057-a80b16b0-9e22-47e0-9721-bd964e248434.png)


Le même processus et calcul a été réalisé pour les réalisteurs. Néanmoins, les réalisateurs travaillent sur moins de films en moyenne que les acteurs : 6,69 films par acteur contre 1,05 film par réalisateur. Cela fait que beaucoup de réalisateurs n'ont qu'une seule expérience. Donc, que nous affectons pour une grande partie des réalisateurs la valeur de 0.5. De plus, l'expérience des réalisateurs est assez inégale. Une grande majorité des réalisateurs n'ont pas d'expérience.

Distribution de l'indicateur de la part de comédie dans l'expérience du réalisateur, et l'expérience du réalisateur :

![image](https://user-images.githubusercontent.com/117921986/212500311-74759152-f272-45fa-83dc-eda7afb24361.png)
![image](https://user-images.githubusercontent.com/117921986/212500313-70965514-3396-480e-bf89-a4a20bec375b.png)

Cependant, la distribution de l'indicateur de la part de comédie des réalisateurs est trop particulière. En effet, elle est séparé en 3 grands groupe : les 0, les 0.5 et les 1. De notre point de vue, cette distribution est trop particulière pour qu'on l'ajoute dans nos modèles. Nous avons tout de même essayé de créer 3 variables qualitatives muettes avec pour intervalle [0;0,33], [0,33;0,67] et [0,67;1]. Nous avons testé ces variables muettes dans une sélection de variable, mais ces variables faisait partie des dernières sélectionnées (cf partie).


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
