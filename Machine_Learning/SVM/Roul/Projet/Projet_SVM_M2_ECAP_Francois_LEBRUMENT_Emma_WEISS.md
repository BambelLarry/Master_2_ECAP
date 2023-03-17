# Projet Support Vector Machine et Réseaux de Neurones  
__François Lebrument et Emma Weiss-Blanchard__

__Master 2 Econométrie et Statistiques Appliquées__

_Annotation sur l'écriture inclusive utilisée dans ce rapport :_ Ce projet a été rédigé avec de l’écriture inclusive. Le point médian a été utilisé à plusieurs reprises ainsi que le pronom « iel ».  Ce pronom est employé pour évoquer une personne quel que soit son genre. 

# Sommaire

1. [Introduction](#introduction)
2. [Importation et nettoyage des données](#paragraph1)
    1. [Source et format de la base](#subparagraph1)
    	1. [Provenance de la base](#test1)
    	2. [Description des variables](#test2)
    3. [Nettoyage des données](#subparagraph2)
    	1. [Traitements des valeurs manquantes](#test3)
    	2. [Répartition du genre des films](#test4)
    	3. [Traitement des données textuelles](#test5)
    5. [Création d'indicateurs](#subparagraph3)
    	1. [Indicateur du continent de production du film](#test6)
    	2. [Indicateur de sentiments de la description](#test7)
    	3. [Indicateur d'expérience précédente du casting et du réalisateur](#test8)
    7. [Traitement des valeurs atypiques](#subparagraph4)
    	1. [Détection des outliers](#test9)
    	2. [Vérification des atypicités](#test10)
  
3. [Statistiques descriptives](#paragraph2)
    1. [Statistiques univariées](#subparagraph5)
    	1. [Variable à expliquer](#test11)
    	2. [Variables explicatives](#test12)
    3. [Statistiques bivariées](#subparagraph6)
    	1. [Les variables quantitatives](#test13)
    	2. [Les variables qualitatives](#test14)
    	3. [La variable à expliquer](#test15)
    5. [Traitement des variables corrélées](#subparagraph7) 
    	1. [Analyses en composantes principales](#test16)
    	2. [Sélection des variables](#test17)
    
4. [Modélisation](#paragraph4)
    1. [Modification de la base](#subparagraph8)
    	1. [Découpage du dataset](#test18)
    	2. [Standardisation](#test19)
    	3. [Présentation des indicateurs de performances des modèles](#test20)
    3. [Modèles de Random Forest](#subparagraph9)
    	1. [Différents modèles de Random Forest et choix du modèle final](#test21)
    	2. [Tuning du meilleur modèle](#test22)
    	3. [Interprétation des résultats des modèles optimisés](#test23)
    5. [Modèles de SVM](#subparagraph10)
    	1. [Différents modèles de SVM et choix du modèle final](#test24)
    	2. [Tuning du meilleur modèle](#test25)
    	3. [Interprétation des résultats du modèle optimisé](#test26)
    7. [Modèles de Réseau de neurones](#subparagraph11)
    	1. [Modèle Réseau de Neurones et optimisation](#test27)
    	2. [Interprétation du modèle optimisé](#test28)
    9. [Comparaison des meilleurs modèles et prévisions avec de nouvelles données](#subparagraph12)
    	1. [Comparaison des modèles optimisés](#test29)
    	2. [Prévision sur de nouvelles données](#test30)
    
5. [Conclusion](#paragraph5)

7. [Discussion](#paragraph6)


# Introduction <a name="introduction"></a>

Notre analyse consiste à classifier le genre des films à partir de plusieurs informations relatives au film. Nous avons récupéré une base de  données sur Kaggle  provenant du site italien de recommandation de films : [filmtv.it](https://www.filmtv.it/). Cette base contient pour chaque film des données comme la date de sortie, le réalisateur, le casting, la description et le genre. De plus, pour chacun des films, nous avons à disposition la moyenne des notes des critiques, des internautes et des notes séparées par thème l'humour, le rythme, la tension ou l'érotisme. La base initiale contient environ 40 000 films répertoriés pour 19 variables.	
	
Notre objectif est de prédire le genre du film grâce à l'exploitation d'un maximum d'informations à notre disposition. L’objectif derrière ce projet, c'est de réaliser des prédictions du genre du film par des méthodes de Machine Learning et Deep Learning. Le cas d’usage que pourrait avoir cette analyse serait de faciliter l'affectation d'un genre du film pour des entreprises de service de vidéo à la demande (Allociné, Netflix, AmazonPrime...). Dans notre analyse, nous nous sommes limité.e.s à la détermination de 2 genres : les comédies et les drames. Nous émettons comme hypothèse en amont que cette classification peut se baser sur le casting présent dans le film, le pays de production, le ton du résumé du film et la note des internautes sur l’humour.

**Quelles sont les caractéristiques des films discriminants le mieux le genre du film et quels sont les modèles de prédictions du genre les plus efficaces sur notre base de données ?**
	
Dans un premier temps, nous allons expliquer notre démarche de compréhension et de traitement de la base. Pour le nettoyage de la base, nous sommes passé.e.s par un traitement des valeurs manquantes, une création d’indicateur et une vérification des valeurs atypiques. Dans un deuxième temps, nous avons observé les distributions de nos variables explicatives, ainsi que les liens qu’elles pouvaient avoir avec le genre du film. Dans un troisième temps, nous avons mis en place des modèles de prédiction de type Random Forest, SVM et réseaux de neurones. Nous avons sélectionné les meilleures modèles, puis nous les avons optimisés pour réaliser les meilleurs prédictions du genre.	

# I. Importation et nettoyage des données <a name="paragraph1"></a>
## A. Source et format de la base <a name="subparagraph1"></a>
### 1. Provenance de la base <a name="test1"></a>
	
Nous avons travaillé à partir d’une base de données disponibles sur le site [Kaggle](https://www.kaggle.com/datasets/stefanoleone992/filmtv-movies-dataset?select=filmtv_movies+-+ENG.csv). Cette base de données a été importée depuis le site de filmTV via du web scraping. Elle porte sur les notes données à différents films. L’intérêt de cette base est de permettre de comprendre les notations données aux films. La base initiale comprend 40 047 films différents en version anglaise, et 19 variables.  

### 2. Description des variables <a name="test2"></a>

•	__Identifiants des films (filmtv_id)__ : Cette variable correspond aux identifiants uniques donnés à chaque film.

•	__Title__ : La variable Title nous donne le titre de chaque film dans sa langue maternelle. 

•	__Year__ :  Cette variable nous permet de connaitre l’année de sortie des films.

•	__Genre__ : La variable Genre correspond aux différentes catégories de films que nous avons dans notre base. Nous avons 27 genres de films dans cette base de données.  Dans ce projet, nous nous intéresserons uniquement aux films dramatiques et comiques. 

•	__Duration__ : Duration nous indique la durée des films.

•	__Country__ : Cette variable permet de savoir de quelle origine sont issues les films de notre base de données.

•	__Directors__ : Grâce à la variable Directors, nous savons quels.les réalisateurs ou réalisatrices ont produit les différents films.

•	__Actors__ : Puis, la variables Actors nous permet de connaitre l’ensemble du casting pour chaque film.

•	__Moyenne des votes (avg_vote)__ : La variable avg_note nous permet de connaitre la note moyenne obtenue entre les notes du public et des critiques.

•	__Vote de la critique (critics_vote)__ :  Avec cette variable, nous connaissons uniquement les notes données par les critiques aux films. Ces notes vont de 0 à 10.

•	__Vote du public (public_vote)__ : A l’inverse, cette variable nous indique les notes données par le grand public aux différents films. Ces notes vont de 0 à 10.

•	__Vote Total (total_vote)__ : Cette variable correspond au nombre total des votes exprimés par les critiques et le public.

•	__Description__ : Nous avons ensuite cette variable qui nous permet d’avoir la description globale des films. 

•	__Notes__ : Cette variable est une annotation textuelle qui vise à décrire de façon très générale les films. 
  
•	__Humor / Tension / Rhythm / Effort  / Erotism__ : Ces 5 variables sont des notes attribuées aux films afin de les caractériser (en fonction de leur humour, suspense, érotisme…). Ces notes sont comprises entre 0 et 5. 


## B. Nettoyage des données <a name="subparagraph2"></a>
  
  Notre analyse étant centrée uniquement sur les films humoristique et dramatiques nous avons dans un premier temps supprimé tous les autres films ne faisant pas partie de ce genre. Nous avons à présent une base de données composée de 20 699 films.
  
### 1. Traitement des valeurs manquantes <a name="test3"></a>
  
  Dans un premier temps, nous devons regarder si notre base de données ne comporte pas de valeurs manquantes. 
  
  *Table n°1 : Tableau des valeurs manquantes*
  
| Variables        | Nombre de valeurs manquantes           | Pourcentage de valeurs manquantes  |
| ------------- |:-------------:| :-----:|
| **Filmtv_id**     | 0| 0 % |
| **Title**     | 0| 0 % |
| **Year**     | 0| 0 % |
| **Genre**     | 0| 0 % |
| **Duration**     | 0| 0 % |
| **Country**     | 3| 0.01 % |
| **Directors**     | 15| 0.07 % |
| **Actors**     | 15| 0.07 % |
| **Avg_vote**     | 0| 0 % |
| **Critics_vote**     | 1631| 7.88 % |
| **Public_vote**     | 202| 0.98 % |
| **Total_votes**     | 0| 0 % |
| **Description**     | 761| 3.68 % |
| **Notes**     | 10425| 50.36 % |
| **Humor**     | 0| 0 % |
| **Rythme**     | 0| 0 % |
| **Effort**     | 0| 0 % |
| **Tension**     | 0| 0 % |
| **erotisme**     | 0| 0 % |
  
 *Source : Dossier SVM et Réseau de neurones, François Lebrument et Emma Weiss-Blanchard* 
  
  Ce tableau nous permet de repérer les variables qui possèdent des données manquantes. Nous voyons que la variable ***Note*** comporte 10 425 observations manquantes, ce qui représente 50.36% des films. Nous décidons de supprimer cette variable puisqu’il sera difficile de l’exploiter avec autant d’observations inexistantes. 
Puis, nous voyons que nous avons 6 autres variables comportant des valeurs manquantes mais avec des proportions plus faibles. Afin de conserver ces variables, nous décidons de directement supprimer les films pour lesquels il manque des valeurs. Nous passons ainsi à une base de données avec 18 218 films et 18 variables explicatives.

  
### 2. Répartition du genre du film <a name="test4"></a>
  
  Puisque nous travaillons avec une variable à prédire qualitative de type binaire, nous devons prêter une attention particulière à la répartition de cette dernière. En effet, si la variable à prédire est déséquilibrée, cela pourra avoir un impact direct sur la qualité des futurs modèles. 
  
  *Figure n°1 : Répartition de la variable Genre*
 
  ![image](https://user-images.githubusercontent.com/116641409/213942448-ee14e020-4426-4ec5-8d0a-3c4d9610c3a2.png)

*Source : Dossier SVM et Réseau de neurones, François Lebrument et Emma Weiss-Blanchard*  
	
  La figure numéro 1 nous permet de visualiser la répartition au sein de la variable Genre. Dans notre base de données, nous avons 55.7% de films qui sont dramatiques et 44.3% de films comiques. La répartition n’est pas totalement égalitaire mais elle en est proche. Nous décidons de ne pas appliquer de rééquilibrage et de la garder comme cela pour la suite de notre étude. 
  
### 3. Traitement des données textuelles <a name="test5"></a>
	
Plusieurs variables de notre jeu de données sont sous format texte. C'est le cas du titre (***title***), du résumé (***description***), du pays (***country***), des réalisatrices.eurs (***directors***) et des actrice.eurs (***actors***). Pour les exploiter, il est nécessaire de séparer les termes qui composent la donnée. 	
	
Nous avons utilisé la technique de tokenization de la méthode Natural Language Processing (NLP) pour séparer les termes des textes bruts pour les variables ***title*** et ***description***. Avec chaque terme séparé (ponctuation inclue), nous avons mis en place un indicateur de sentiment basé sur la description (cf partie I.C.2.).

Les 3 autres variables sont des listes de noms propres. En effet, ces variables peuvent avoir plus d’un élément pour chaque observation (exemple, plus d'un pays de production). Nous avons alors séparé les noms propres via le séparateur « ,  » pour que chaque élément soit unique. Par exemple, pour la variable ***actors***, un élément de la liste correspond au nom et prénom de la personne. Nous obtenons donc, pour une observation, la liste des acteurs séparée distinctement.
	
Les 5 variables textuelles traitées sont maintenant stockées sous forme de liste pour faciliter la réalisation de traitement. Nous pouvons à présent les exploiter pour créer de nouveaux indicateurs.	
	
## C. Création d'indicateurs <a name="subparagraph3"></a>
	
Une fois les données textuelles séparées, nous pouvons les exploiter pour créer de nouveaux indicateurs. À noter que nous n'avons pas réalisé de nettoyage des données textuelles. Nous n'avons pas retiré de stopwords, ni retiré la ponctuation ou des erreurs de saisies comme "©" apparaissant plusieurs fois. Nous n'avons pas considéré cela comme pertinent pour les indicateurs que nous souhaitons créer.
	
### 1. Indicateur du continent de production du film <a name="test6"></a>
	
Le premier indicateur correspond au continent de production du film basé sur la variable ***country***. L’hypothèse sous-jacente est que certains pays produisent plus un genre de films que d’autres. Par exemple, l'Inde est réputée, via Bollywood, pour créer des films musicaux. Nous supposons que certains pays produisent plus de comédies que de drames et inversement.	
	
Nous pouvons remarquer que la distribution des pays de production n’est pas équilibrée. Les Etats-Unis et l'Italie concentrent plus de la moitié des pays producteurs de notre base (59,69 % du dataset au moment du traitement). Afin de ne pas créer trop de variables dummies, nous avons choisi de regrouper par continent.
	
Par ailleurs, certains films ont plusieurs pays producteurs, dont certains de plusieurs continents différents. Pour faciliter le traitement, nous avons considéré que ces films sont "internationaux", même si le film est produit dans plusieurs pays d’un même continent.
	
*Tableau N°2 : Distribution des pays producteurs sans regroupement*
		
| Pays ou groupe de pays      | Effectif   | 
| ------------- |:-------------:| 
| **United States**     | 6 343|
| **Italy**     | 4 531|
| **France**     | 1 305|
| **Great Britain**     | 794|
| **Germany**     | 337|
| **...**     | ...|
| **Chile, Argentina, Germany**     | 1|
| **Mexico, France, Sweden**     | 1|
| **Romania, France, Hungary**     | 1|
| **Germany, Czechoslovakia**     | 1|
| **Palestine, Netherlands, Germany, Mexico**     | 1|
	
*Source : Dossier SVM et Réseau de neurones, François Lebrument et Emma Weiss-Blanchard* 	
	
Pour le traitement des pays, nous avons utilisé les librairies *pycountry* et *pycountry_convert*. La première nous donne la liste des pays du monde (mais qui n’est pas complète). La seconde convertie le nom du pays en continent.
	
Voici le processus des traitements pour convertir le nom de pays en continent :	

• __1__ :	Nous avons converti les films multinationaux en "internationaux".
	
• __2__ :Nous avons converti les pays existant dans la liste countries de *pycountry* en continent.
	
• __3__ :	Pour les pays n'existant pas dans la liste pycountry, nous les avons classés en "autre". La liste *pycountry* n'est pas complète pour certains pays existants comme "*Great Britain*", "*Russia*" ou "*CzechRepublic*". Néanmoins, ces derniers peuvent être convertis en continent. Ainsi, nous avons ajouté ces pays manuellement dans la liste pour les convertir en continent.
	
• __4__ :	Pour le reste des pays ne pouvant pas être convertis, nous les avons directement affectés au continent. Par exemple, nous avons traité d'ancien pays comme "*Soviet Union*", "*Czechoslovakia*" ou "*East Germany*". Seul « Soviet Union » a été ajouté en Asie, tous les autres ont été affectés à l’Europe.
	
*Tableau N°3 : Distribution des continents sans regroupement*
	
| Continents      | Effectif   | 
| ------------- |:-------------:| 
| **Europe**     | 7 752 |
| **North America**     | 6 574 |
| **Internationaux**     | 2 789|
| **Asia**     | 829|
| **South America**     | 134 |
| **Australia**     | 119 |
| **Africa**     | 21|	
	
*Source : Dossier SVM et Réseau de neurones, François Lebrument et Emma Weiss-Blanchard*

Encore une fois, la distribution des continents est déséquilibrée. Pour pallier ce problème et pour ne pas avoir trop de modalités (donc de variables dummies), nous avons regroupé arbitrairement certains continents. Tous les continents hors ***Europe*** et ***Amérique du Nord*** ont été regroupés dans ***Internationaux***. 	
	
Pour rendre exploitable la variable du continent dans nos modèles, nous l'avons transformée en 3 dummies variables. Il s'agit de variables binaires. La variable muette ***EU*** est créée pour les films européens, ***NA*** pour les films nord américains et ***Inter*** pour les films internationaux (et des autres continents). Evidemment, chaque film possède un 1 dans une des trois variables et deux 0 dans les deux autres (un film étant produit que sur un continent).
	
### 2. Indicateur de sentiments de la description <a name="test7"></a>
	
Nous supposons que la ***description*** des films peut aider à distinguer  les deux genres de films. Nous émettons l’hypothèse que les résumés des films dramatiques sont plus négatifs que ceux des films comiques. C'est la raison pour laquelle nous avons créé un indicateur de sentiments sur la variable ***description***.
	
Nous avons utilisé l'approche *lexical* de la méthode NLP pour créer l'indicateur de sentiment. Cette approche se base sur une liste de mots de références créées manuellement. Ici, nous avons utilisé la liste de référence *Opinion Lexicon* de la librairie *nltk.corpus*. Il s'agit de deux listes de mots : une composée de mots positifs et l'autre de mots négatifs. Nous comptons simplement le nombre d’occurrence de mots en commun entre ces listes et les mots du résumé des films.	
	
À partir de cette comptabilisation, nous avons créé l'indicateur du ton ***ton_global*** représentant le sentiment dégagé (positif ou négatif) suite à la lecture du texte. Ce ton est calculé de la manière suivante :	 ![image](https://user-images.githubusercontent.com/116641409/215738730-33e8ade3-bb7e-42ac-86ea-377346755649.png)
Cet indicateur se situe entre -1 et 1. Plus il s’éloigne de 0, plus le ton est fort. Si l'indicateur est négatif, alors le ton est négatif (plus de mots négatifs que positifs) et inversement.
	
Toutefois, cet indicateur possède quelques limites. D'abord, il ne prend pas en compte la négation ("not") inversant la polarité d'un terme. La liste de mots "*Lexicoder Sentiment Dictionary*" la prend en compte, mais il est trop coûteux pour nous de la mettre en place. Ensuite, l'indicateur ne remet pas les mots dans leur contexte. En effet, nous nous sommes basé.é.s sur des listes *Opinions Lexicons *définissant la polarité des termes de manières strictes. Il n'y a pas de nuance en fonction du contexte ou du sens du mot. La liste de mots "*SentiWordNet*" prend en compte les nuances en pondérant la négativité et la positivité d’un mot. Puis, nous ne prenons pas en compte l'orthographe des mots, surtout sans avoir réalisé de nettoyage. En effet, faire un nettoyage des résumés nous prendrait trop de temps. Toutefois, *Opinion Lexicon* propose pour quelques exceptions des orthographes de mots différentes, ce qui nous facilite la tâche. Enfin, les mots avec une polarité (positive ou négative) dans un texte sont largement minoritaires. Ils ne constituent qu’environ 5 % d'un texte. Donc, la notion de ton d'un texte ne représente pas son intégralité.
	
Malgré les limites de notre indicateur de sentiments, il nous donne un aperçu global du ton du texte. Nous supposons que les drames ont principalement des résumés considérés comme négatifs et les comédies des résumés positifs.
	
	
### 3. Indicateur d'expérience précédente du casting et du réalisateur ou de la réalisatrice <a name="test8"></a>
	
Nous supposons aussi que certain.e.s ***actors*** et ***directors** sont souvent associé.e.s à certains genres de films. Par exemple, l'acteur Jim Carrey est connu pour ses premiers rôles dans des comédies comme "*Ace Ventura*", "*The Truman Show*" ou encore "*The Mask*". Grâce à l'expérience du casting et du réalisateur/de la réalisatrice, nous pourrions mieux prédire le genre du film. C'est pour cela que nous avons créé l'indicateur de la part de comédie dans l'expérience de l’ensemble du casting sur l'ensemble des films (comédie+tragédie) dans lesquels les acteurs et actrices ont joué.
	

Nous avons créé ces indicateurs séparément pour les ***actors*** et les ***directors***, même si le processus est le même. Nous allons présenter nos approches avec l’exemple de l'indicateur des actrices et des acteurs.	
	
Au départ, nous avions comptabilisé le nombre de comédie et de drame dans lequel chaque acteur et actrice avait joué, indépendamment de la date de sortie du film. Puis, nous sommions le nombre de comédie/drame de l'ensemble des acteurs.trices du casting pour créer l'indicateur d'expérience. Néanmoins, cela signifie que pour un.e acteur.trice ayant joué dans un film sorti en 1960 par exemple, nous comptabilisions le nombre de comédie et de drame du casting sorti après cette année. Comme cet indicateur se basait sur notre variable à expliquer, nous nous demandions si nous devions inclure le dit film. Comme nous sommes supposé.es la prédire, nous avions imaginé de créer l’indicateur d’expérience exclusivement sur une base de train. Puis, pour la base test, l'indicateur se basait sur celui mis en place par la base train. En outre, nous nous demandions s'il fallait pondérer le nombre de comédie/drame par le nombre d'acteurs et d’actrices dans le casting. En effet, un casting nombreux sur un film augmente logiquement l'expérience globale.

Cette première approche contenait plusieurs problèmes, questionnements et limites. Suite à des discussions avec notre professeur, nous avons modifié notre approche pour revenir à un indicateur plus simple : l'expérience précédente du casting dans les comédies et drames. Cela suppose donc que pour les films sortis en 1960, nous ne prenons en compte que l'expérience des acteurs avant la sortie du dit film.	
	
Pour cela, nous avons créé une base de données des acteurs et actrices avec pour chaque observation l'acteur.trice, le film dans lequel iel a joué et l’année de sortie du film. Puis, pour chaque couple acteur.trice-film, nous avons compté le nombre de comédie et de drame pour lequel l'acteur ou l’actrice a joué avant l'année la sortie du film. Le processus fonctionne comme un filtre. Voici ci-dessous un exemple pour comment cette base des acteurs et actrices est représentée pour un.e actrice.eur ici "*Kim Rossi Stuart*".
	
*Tableau N°4 : Films de Kim Rossi Stuart* 
	
| filmtv_id       | Actors   | Genre  | Year  | Nombre de comedie  | Nombre de drame  |
| ------------- |:-------------:| :-----:| :-----:| :-----:| :-----:|
| 3     | Kim Rossi Stuart |1 | 1991 | 0 | 0 |
| 293     | Kim Rossi Stuart |1 | 1992 | 0 | 1 |
| 10 113     | Kim Rossi Stuart |1 | 1994 | 0 | 2 |
| 12 745     | Kim Rossi Stuart |1 | 1994 | 0 | 2 |
| 12 918     | Kim Rossi Stuart |1 | 1995 | 0 | 4 |
| 14 321     | Kim Rossi Stuart |1 | 1995 | 0 | 4 |
| 17 580     | Kim Rossi Stuart |1 | 1998 | 0 | 6 |
| 17 591     | Kim Rossi Stuart |1 | 1998 | 0 | 6 |
| 23 920     | Kim Rossi Stuart |0 | 2002 | 0 | 8 |
| 41 454     | Kim Rossi Stuart |1 | 2004 | 1 | 8 |
| 27 028     | Kim Rossi Stuart |1 | 2004 | 1 | 8 |
| 32 248     | Kim Rossi Stuart |1 | 2005 | 1 | 10 |
| 37 339     | Kim Rossi Stuart |1 | 2007 | 1 | 11 |
| 30 729     | Kim Rossi Stuart |0 | 2008 | 1 | 12 |
| 56 477     | Kim Rossi Stuart |1 | 2013 | 2 | 12 |
| 67 642     | Kim Rossi Stuart |1 | 2015 | 2 | 13 |
| 86 588     | Kim Rossi Stuart |1 | 2016 | 2 | 14 |
| 183 065    | Kim Rossi Stuart |0 | 2020 | 2 | 15 |
| 171 961    | Kim Rossi Stuart |1 | 2020 | 2 | 15 |
| 218 100    | Kim Rossi Stuart |1 | 2022 | 3 | 16 |
	
*Source : Dossier SVM et Réseau de neurones, François Lebrument et Emma Weiss-Blanchard*
	
Ensuite, nous avons simplement fusionné notre base des acteurs.trices avec celle des films en sommant pour l'ensemble du casting. Nous obtenons pour chaque film le nombre de comédie et de drame (séparément) dans lequel a joué le casting avant.
	
Enfin, nous avons calculé l'indicateur de la part de comédie d'expérience sur l'ensemble des films d'expérience (comédie et drame). Cet indicateur nous permet d'éviter le problème d'inégalité entre des films avec des gros castings et ceux avec des castings avec peu d'expérience. En effet, cet indicateur varie entre 0 si le casting n'a que des expériences dans les drames et 1 s'il n'a que des expériences dans les comédies. Cet indicateur est vu comme une jauge avec pour centre 0.5. Cela signifie que le casting a autant d'expérience dans les comédies que dans les drames. Cependant, il peut arriver pour certains films le casting n'ait aucune expérience. Dans ces cas-là, nous les affectons à 0.5, l’équilibre de la jauge. Ainsi, nous avons créé les variables ***per_comedy_casting***  et ***per_comedy_real*** .
	
	
De plus, nous avons ajouté dans la base un indicateur de l'expérience total du casting, nommé ***xp_casting*** et ***xp_real***
	
Toutefois, bien que nous implémentons ces indicateurs dans la base, ils possèdent tout de même quelques limites. En effet, il est probable que certain.e.s acteurs et actrices aient le même nom et prénom. Iels peuvent fausser l'indicateur, mais nous ne pouvons rien y faire (hormis vérifier l'ensemble des acteurs et des actrices à la main). Puis, comme nous n'avons pas la date précise de sortie du film, seulement l'année, l'indicateur exclut les films sortis la même année (même s'ils sont sortis à une date antérieure). Cela a probablement peu d'impact sur notre indicateur au final. Ensuite, l'expérience du casting se base seulement sur les films présents dans la base et non sur l'ensemble des films de la carrière des acteurs.  Enfin, la non expérience de plusieurs castings fait que la classe modale devient 0.5. Cela est plus flagrant sur la distribution pour les ***directors***.
	
Le même processus de création d’indicateur a été réalisé pour la variable ***directors***. Néanmoins, les réalisateurs et les réalisatrices travaillent sur moins de films en moyenne que les acteurs : 6,69 films par acteur.trice contre 1,05 film par réalisateur.trice. Cela fait que beaucoup de réalisateurs et de réalisatrices n'ont qu'une seule expérience. Ainsi, nous affectons pour une grande partie des réalisateurs.trices à la valeur de 0.5. De plus, l'expérience des réalisateurs et des réalisatrices est assez inégale. Une grande majorité des réalisateurs.trices n'ont pas d'expérience.
	
*Figure N°2 : Distribution des variables per_comedy_realetxp_real*	

![image](https://user-images.githubusercontent.com/116641409/215749030-44e019e3-ee4a-4725-bf5e-55c4a64bb717.png)

*Source : Dossier SVM et Réseau de neurones, François Lebrument et Emma Weiss-Blanchard*
	
Nous considérons la distribution de l'indicateur de la part de comédie dans l’expérience des réalisateurs et des réalisatrices comme étant particulière. En effet, elle est séparée en 3 grands groupes : les 0, les 0.5 et les 1. De notre point de vue, cette distribution est trop particulière pour que nous l’ajoutions dans nos modèles comme cela. Nous avons tout de même essayé de discrétiser cet indicateur en 3 dummies variables avec pour intervalle [0;0,33], [0,33;0,67] et [0,67;1]. Néanmoins, en testant ces dummies variables dans une sélection de variable, elles faisaient partie des dernières sélectionnées (cf partie II.C.2).
	
Les 2 indicateurs relatifs aux ***directors*** nous semblent trop particuliers pour les inclure dans les modèles. Toutefois, nous avons tout de même réalisé des analyses de ces variables.
	

## D. Traitement des valeurs atypiques <a name="subparagraph4"></a> 

### 1.	Détection des outliers <a name="test9"></a>
  
  A présent nous avons notre base de données complète avec les indicateurs que nous avons créés. Dans un premier temps, nous allons nous intéresser aux statistiques descriptives simples concernant les variables quantitatives afin de déceler de potentielles anomalies.
  
  *Tableau n°5 : Statistiques descriptives*
  
  ![image](https://user-images.githubusercontent.com/116641409/213942654-76c44b48-55d4-4204-b486-8dea394422d1.png)

  ![image](https://user-images.githubusercontent.com/116641409/213942662-f5df1582-9d4a-4eb4-9733-95e1ba1cfab3.png)
  
  *Source : Dossier SVM et Réseau de neurones, François Lebrument et Emma Weiss-Blanchard*
  
  Dans un premier temps, nous pouvons nous intéresser aux différents indicateurs que nous avons créés précédemment. Nous avons en premier les indicateurs de sentiments : ***pos, neg, ratio_pos_neg et ton_global***. Nous allons conserver uniquement l'indicateur ***ton_global*** qui permet de donner une indication générale sur les sentiments, positifs ou négatifs, qui ressortent sur la description des films. Les 3 autres indicateurs nous permettent de construire ***ton_global***, il n'est pas pertinent de les conserver eux aussi car cela pourrait conduire à de la multicolinéarité entre ces variables. L'indicateur ***ton_global*** est compris entre 1, qui correspond à une description uniquement avec des sentiments positifs, et -1 (-0.131334 pour notre échantillon) qui indique une description de film avec uniquement des sentiments négatifs.
  
  
  Nous avons également les 8 indicateurs liés aux castings du film et aux réalisatrices.eurs que nous avons créés et détaillés précédemment. Comme pour l’indicateur de sentiment, nous n’allons pas conserver l’ensemble de ces indicateurs car cela pourrait introduire de la multicolinéarité dans nos modèles. Nous décidons de conserver dans un premier temps les variables ***per_comedy_casting*** et ***per_comedy_real***, qui sont des indicateurs compris entre -1 et 1, à la différence des variables **nb_drama** et **nb_comedy** qui ont des distributions plus étendues et ainsi plus de risques de contenir des outliers. Puis, nous décidons de conserver les variables ***xp_casting*** et ***xp_real***. La première correspond au nombre de films d’expériences précédentes du casting et la seconde au nombre de films d’expérience du réalisateur ou de la réalisatrice. Le minimum pour ces 2 variables est de 0, ce signifie que dans certains films le casting n’a pas d’expérience. Cependant, nous remarquons pour ces indicateurs qu’il y a un écart important entre la valeur du 3ème quartile et de la valeur maximale. De plus, l’écart type de cette variable est assez élevé. Ainsi, nous suspectons ces variables de contenir des valeurs atypiques. Nous devrons le vérifier ultérieurement.
  
  Maintenant que nous avons sélectionné et analyser les indicateurs que nous garderons par la suite, nous pouvons revenir aux variables initialement fournies par notre base. La première variable du tableau, ***filmtv_id***, qui correspond aux identifiants donnés à chaque film dans la base de données que nous avions récupérée initialement. Elle n'est pas pertinente à conserver car elle ne nous apporte aucune information pour prédire le genre des films. Nous avons donc décidé de la supprimer de notre base. 
  
  Nous accordons une attention particulière aux variables qui ont des distributions de valeurs très grandes, car cela pourrait indiquer la présence potentielle de valeurs atypiques. Pour la variable ***year***, nous constatons que la valeur minimum est assez éloignée de la valeur du 1er quartile. De plus, l'écart-type est relativement élevé. Nous soupçonnons certainement valeurs, notamment dans les premières valeurs, d'être atypiques. A l'inverse, nous remarquons pour la variable ***duration*** qu'il y a un grand écart entre la valeur du 3ème quartile qui est de 110 minutes et celle de la valeur maximale qui est à 924 minutes (un film de 924 minutes correspond à environ 15h30 de visionnage). De plus, nous voyons qu'il y un écart relativement important entre la valeur minimale et le 1er quartile. Nous pensons donc que cette variable contient elle aussi des valeurs atypiques. Cependant, les valeurs maximales et minimales pour cette variable nous ont poussé à faire des recherches sur les films que notre base de données contient. Nous avons réalisé que notre base prenait en compte des mini-séries et la durée correspond à la durée totale de tous les épisodes réunis (ce qui est le cas pour la mini-série *Un matrimonio* par exemple), ce qui explique la durée de certains films soit très élevée ou faible (pour ceux qui correspondent qu'à un seul épisode). Nous avons tout de même des films qui durent très longtemps, comme *Heimat - Eine Chronik in elf Teilen*, qui dure 924 minutes). Il sera donc nécessaire, en plus de vérifier l'atypicité de certaines valeurs, d'imposer des filtres plafond et plancher afin que nous gardions uniquement les films et non les mini-séries.
  
  Nous remarquons qu'il y a aussi des écarts importants entre le 3ème quartile et la valeur maximale pour la variable ***total_votes*** et ***erotims***. Pour la variable ***total_votes***, nous voyons également que nous avons un écart-type élevé et une différence relativement importante entre la médiane et la moyenne. Nous soupçonnons donc ces variables de posséder valeurs atypiques.
  
  Pour confirmer les éléments que nous venons de mettre en avant, nous allons réaliser des histogrammes pour ces variables afin de mieux visualiser la distribution des valeurs.
  
  *Figure n°3 : Histogrammes des variables quantitatives*
  
  ![image](https://user-images.githubusercontent.com/116641409/214380656-346e64ee-d8e2-41f7-b123-f7b088b729ab.png)
  
   *Source : Dossier SVM et Réseau de neurones, François Lebrument et Emma Weiss-Blanchard*
  
  Grâce aux histogrammes réalisés ci-dessus, nous pouvons visualiser plus facilement la distribution de valeurs au sein de nos variables quantitatives. Cela nous permet de confirmer certains points évoqués précédemment. En effet, nous voyons pour la variable ***years*** que nous avons peu de films réalisés dans les premières années de notre échantillon, et une forte concentration des films vers la fin de l'échantillon. Puis, pour la variable ***duration***, nous voyons nettement une symétrie à gauche concernant la distribution dans l'échantillon, ce qui signifie que la majorité des films sont réparties entre 41 minutes et environ 150 minutes. Entre environ 150 et un peu plus de 200 minutes, nous avons un petit nombre de films, et au-delà, nous avons très peu de films jusqu'à la valeur maximale. Nous remarquons la même forme de distribution pour les variables ***erotims***, ***total_votes***, et ***xp_casting*** les majeures parties des valeurs sont réunies à gauche et nous avons peu d'observations vers les valeurs maximales.Nous remarquons également que la variable ***xp_real*** possède à la fois une distribution étendue et beaucoup de effectifs avec la valeur de 0. En effet, il y a 6 856 films avec un.e réalisatrice.eur débutant.e (pas de film à leur actif). De ce fait, les valeurs les plus élevées seront certainement considérées comme atypiques. Cela nous fait douter de la pertinence de cette variable et pour cela, nous décidons de ne pas la garder pour la suite de notre étude.

Nous allons vérifier, que ce soit pour les variables où nous avons relevé des anomalies et les autres, la présence de valeurs atypiques grâce aux boxplots.
  
  *Figure n°4 : Boxplot des variables quantitatives*
  
![image](https://user-images.githubusercontent.com/116641409/215872091-c139a5d3-8fff-4175-8bf9-18d9fb79f3db.png)

  *Source : Dossier SVM et Réseau de neurones, François Lebrument et Emma Weiss-Blanchard*

  Nous voyons pour les variables ***years,  avg_vote,  critics_vote,  total_votes***  et ***erotims*** que nous avons effectivement des valeurs suspectées d'être atypiques à la fin de la distribution de valeurs. Pour la variable ***public_vote***, nous avons une seule valeur potentiellement atypique. Nous allons par la suite appliquer le test ESD qui nous permettra de savoir si ces valeurs suspectées d'être atypiques le sont réellement.
  
  Enfin, concernant la variable ***duration***, nous avons à la fois des valeurs suspectées d'être atypiques vers le haut et le bas. Comme nous l'avons évoqué précédemment, nous devons appliquer un filtre minimum afin de garder que les longs métrages, et non les courts. Pour cela, nous avons regardé la distinction proposée par le Centre national du cinéma et de l'image animée (CNC) entre les longs métrages, et les courts métrages qui ne font pas partie de notre étude. Le CNC considère qu'un long métrage est caractérisé par une durée minimum de 60 minutes. Les courts métrages n'étant pas pris en compte dans notre étude, nous pouvons ainsi mettre un filtre afin de garder que les longs métrages, ce qui équivaut à mettre un filtre avec une valeur minimum de 60 minutes.
  
  A présent que ce filtre est appliqué, nous pouvons de nouveau regarder si nous avons toujours des valeurs suspectées d'être atypiques vers le bas.

  *Figure n°5 : Boxplot de la variable duration avec le filtre planché*
  
![image](https://user-images.githubusercontent.com/116641409/215873863-eb72d03a-d082-4d24-9ce1-4cd104a2777b.png)

  *Source : Dossier SVM et Réseau de neurones, François Lebrument et Emma Weiss-Blanchard*
  
  Le filtre que nous avons imposé permet de conserver uniquement les longs métrages dans notre base de données. Nous avons à présent des valeurs suspectées d'être atypiques que vers le haut. Cela nous permet d'utiliser le test ESD pour pour vérifier si c'est valeurs sont réellement atypiques.
  
  ### 2. Vérification des atypicités <a name="test10"></a>
  
  A présent, nous pouvons passer à la vérification de l’atypicité grâce au test ESD.
  
  *Table n°6 : Vérification des valeurs atypiques*
  
  | Variables        | Nombre de *valeurs* suspectées d’être atypiques   | Nombre *d’observations* réellement atypiques  | La valeur atypique seuil |
| ------------- |:-------------:| :-----:| :-----:|
| **Year**     | 10 | 0  | - |
| **Duration**     | 109 | 215 | 173 |
| **Avg_vote**     | 12 | 0  | - |
| **Critics_vote**     | 6| 0 | - |
| **Public_vote**     | 1 | 0 | - |
| **Total_votes**     | 358 | 518 | 214 |
| **Humor**     | 0 | - | - |
| **Rythme**     | 0 | - | - |
| **Effort**     | 0 | - | - |
| **Tension**     | 0 | - | - |
| **erotism**     | 2 | 48 | 4 |
| **ton_global**     | 0 | - | - |
| **per_comedy_casting**     | 0| - | - |
| **xp_casting**     | 162 | 125 | 181 |
| **per_comedy_real**     | 0 | - | - |
  
   *Source : Dossier SVM et Réseau de neurones, François Lebrument et Emma Weiss-Blanchard*
  
  Le tableau ci-dessus nous indique s’il y a des valeurs atypiques au sein des variables, et le seuil à partir duquel il n’y a plus de valeurs atypiques. Ce seuil à été déterminé grâce au test ESD. Ce tableau nous permet de confirmer que les variables ***duration,  total_votes, erotims*** et ***xp_casting*** ont bien des valeurs atypiques. Les autres variables où nous avions repéré des valeurs potentiellement anormales ne se sont pas avérées atypiques. Pour la variable ***duration***, le test nous indique qu'au-delà de 173 minutes, les observations sont atypiques. Ainsi, nous appliquerons un filtre permettant de garder les films avec une durée inférieure à 173 minutes, ce qui nous fera perdre 215 films. Puis, pour la variable ***total_votes***, au-delà de 214 votes pour un film, l'observation est considérée comme atypique dans notre échantillon. Nous imposerons un filtre pour conserver uniquement les films avec un nombre de votes inférieur à 214, et nous perdrons 519 observations. Puis, nous constatons pour la variable ***erotims*** que la valeur de 4 est considérée comme atypique. Nous appliquerons donc un filtre nous permettant de garder uniquement les notes d'érotisme inférieures à un 4, ainsi nous perdrons 48 individus. Enfin, pour la variable ***xp_casting***, nous allons imposer des seuils à 181 afin de ne pas garder les valeurs atypiques.   

Ainsi, nous arrivons à une base finale composée de 17 227 films et 19 variables.

  
# II. Statistiques descriptives <a name="paragraph2"></a>
## A. Statistiques univariées <a name="subparagraph5"></a> 
  
  A présent que nous avons nettoyé notre base de données, nous pouvons passer son analyse en commençant par des statistiques univariées. 
  
### 1. Variable à expliquer <a name="test11"></a>
  
  Nous commençons par regarder de nouveau la répartition de notre variable à prédire afin de savoir si la suppression des valeurs atypiques a modifié la distribution des valeurs au sein des 2 catégories.
  
  *Figure n°6 : Répartition de la variable Genre*

 ![image](https://user-images.githubusercontent.com/116641409/215760951-b19fcb02-0fe5-42dd-a695-80531a03b984.png)

 *Source : Dossier SVM et Réseau de neurones, François Lebrument et Emma Weiss-Blanchard*

  Nous constatons que la répartition de notre variable à prédire s’est légèrement améliorée. Nous n’allons pas appliquer de rééquilibrage et garder la variable Genre comme cela dans les modèles.
  
### 2. Variables explicatives <a name="test12"></a>
  
  Ensuite, nous pouvons passer à l’analyse de nos variables explicatives en commençant par les variables quantitatives. 
  
  *Tableau n°7 : Statistiques descriptives*
  
  ![image](https://user-images.githubusercontent.com/116641409/214397254-e5bae754-965c-4e50-b825-168625414b8a.png)

  *Source : Dossier SVM et Réseau de neurones, François Lebrument et Emma Weiss-Blanchard*

  Nous observons une amélioration des répartitions des valeurs au sein des variables après la suppression des valeurs atypiques. Pour les variables où nous avons appliqué des filtres, les moyennes et médianes sont plus proches et les écart-types sont plus faibles.
Pour le confirmer de façon plus visuelle, nous pouvons de nouveau réaliser les histogrammes des variables quantitatives. 
  
  *Figure n°7 : Histogrammes des variables explicatives quantitatives*
  
  ![image](https://user-images.githubusercontent.com/116641409/214399570-1487ea6e-4ad1-4565-90eb-6106bf783933.png)
  
  *Source : Dossier SVM et Réseau de neurones, François Lebrument et Emma Weiss-Blanchard*
  
Grâce aux histogrammes au-dessus, nous voyons pour les variables où nous avons supprimé les valeurs atypiques que les distributions des observations au sein des sont plus regroupées. Les modèles que nous appliquerons par la suite ne seront donc pas biaisés par la présence de valeurs atypiques. En revanche, nous constatons que la distribution des valeurs pour l’indicateur ***per_comedy_real*** n’est régulière. En effet, nous voyons que la majeure partie des observations se répartit au niveau des 2 extrémités et du centre. Nous pensons que cette variable nous apportera peu d’informations, ainsi nous décidons de la supprimer de notre base de données.
  
  Nous avons précédemment créé un indicateur qualitatif concernant l’origine géographique des films. Cet indicateur se compose de 3 catégories : l’Europe (EU), l’International (Inter) et North Amérique (NA). Il est intéressant de regarder la répartition de cet indicateur afin de savoir s’il sera pertinent de le garder dans notre base, et de l’exploiter dans  les futurs modèles.
  
  *Figure n°8 : Répartition des continents*
  
![image](https://user-images.githubusercontent.com/116641409/215761210-2c8ed300-1c9f-4fd3-83c5-def6f7f65952.png)
  
  *Source : Dossier SVM et Réseau de neurones, François Lebrument et Emma Weiss-Blanchard*
  
  La catégorie qui contient le plus d’observations est l’Europe avec 42.6% des films de notre base, et celle qui en comporte le moins avec 21.2% est la catégorie des films internationaux. La répartition n’est pas égalitaire entre les 3 catégories mais elle n’en est pas éloignée. Nous décision de la garder comme cela et ainsi de conserver cet indicateur dans notre base de données.

## B. Statistiques bivariées <a name="subparagraph6"></a>
  
  Une fois que nous avons étudié les statistiques univariées, nous pouvons passer aux bivariées afin de compléter notre analyse.
  
### 1. Les variables quantitatives <a name="test13"></a>

  Nous commençons par les variables de type quantitatif. Dans un premier temps, nous faisons des nuages de points entre 2 variables afin de voir la répartition des films en fonction de leur genre. 
  
  *Figure n°9 : Nuage de points entre les variables selon le genre des films*
  
  ![image](https://user-images.githubusercontent.com/116641409/214402053-d169073e-a8e5-4a14-8f7c-04d8e988cc61.png)

  *Source : Dossier SVM et Réseau de neurones, François Lebrument et Emma Weiss-Blanchard*
  
  Les graphiques ci-dessus nous apportent des informations sur les potentiels pouvoirs explicatifs de certaines variables ou encore sur l’existence potentielle de relations entre certaines. Sur le premier graphique entre la variable ***critic_vote*** et ***avg_vote***, nous voyons  distinctement que les différents points, que ce soit pour le genre drama ou comedy, évoluent dans la même direction de façon linéaire. Nous soupçonnons ces deux variables d’être fortement corrélées. Nous pourrons le vérifier ultérieurement lors de la création d’une matrice de corrélation. En ce qui concerne les autres variables, nous n’en voyons pas d’autres qui sembleraient aussi corrélées que ces deux variables. Cependant, nous pouvons voir que certaines associations de variables ont globalement la même répartition entre les 2 styles de films. En effet, avec les nuages de points obtenus avec les variables ***avg_vote*** et ***per_comedy_casting***, ainsi que celui avec ***critics_vote*** et ***per_comedy_casting***, les films comiques sont regroupés plus vers le haut et les films dramatiques vers le bas de façon assez nette. Cela nous questionne sur le fait que les variables critics_vote et avg_vote pourraient potentiellement expliquer le même phénomène. Nous pourrons également le vérifier lors de la création de la matrice des corrélations. Quant-aux autres graphiques, la distinction entre les 2 styles de films est beaucoup moins nette, même si nous apercevons brièvement que les films dramatiques se trouvent plus vers la droite des graphiques lorsque dans la combinaison des variables il y a ***ton_global***. 
  
 Puis, nous pouvons compléter notre analyse en combinant d’autres variables entre elles. Nous avons appliqué différentes combinaisons et nous avons choisi d’en présenter 3 ci-dessous.
  
  *Figure n°10 : Comparaison des 2 genres de films en fonction de différentes variables*
  
  ![image](https://user-images.githubusercontent.com/116641409/214403511-3f810515-9da3-4d7e-9a10-33dc073325e9.png)

  *Source : Dossier SVM et Réseau de neurones, François Lebrument et Emma Weiss-Blanchard*
  
  Grâce à ces graphiques, nous constatons qu’en fonction du genre des films, nous n’avons pas les mêmes relations avec les variables explicatives. Sur le premier graphique, nous avons mis en relation l’année de sortie des films et la note d’humour moyenne obtenue par les films. Nous voyons qu’il y a une nette différence entre les films dramatiques et humoristiques, mais que cet écart semble diminuer à partir des années 1960. En effet, les films humoristiques ont globalement de meilleures notes, ce qui semble logique pour ce type de film. Cette variable permettra potentiellement de bien distinguer les 2 types de films dans les modèles. Puis pour le deuxième graphique, nous constatons que nous avons une évolution assez similaire entre les 2 genres de films mais pas à la même échelle. Plus la note obtenue par le public augmente, plus l’expérience du casting du film augmente dans un premier temps, et baisse drastiquement pour les 2 catégories de films. Nous voyons que les films comiques ont globalement un casting avec plus d’expériences que les films dramatiques. Enfin, grâce au dernier graphique, nous constatons qu’il y a également une différence importante entre le style des films en fonction de la tension et des notes obtenues par le public. En effet, nous voyons que les films dramatiques ont des notes concernant le suspense plus élevées que les films comiques. De façon globale, plus la note de suspense augmente, et plus la note obtenue par le public augmente également, et de manière plus importante pour les films dramatiques. Ainsi, ces différents graphiques nous permettent de mettre en avant les différences entre les 2 genres de films.
  
Comme nous l’avons vu précédemment, il serait intéressant de construire une matrice des corrélations afin de vérifier qu’il n’existe pas de variables fortement liées entre-elles.   
  
  *Figure n°11 : Matrices des corrélations*
  
  ![image](https://user-images.githubusercontent.com/116641409/214405116-e784813a-ee31-43da-bb7b-7c1b1b1864b7.png)

  *Source : Dossier SVM et Réseau de neurones, François Lebrument et Emma Weiss-Blanchard*
  
  Nous avons réalisé une première matrice des corrélations, et afin de pouvoir mieux distinguer les variables fortement liées entre-elles, nous en avons construit une autre qui inclut seulement les corrélations supérieures à 0.50. Nous constatons, comme nous l’avons suspecté précédemment, que les variables ***critics_votes*** et ***avg_notes*** sont fortement corrélées. Elles ont un coefficient de corrélation aux alentours de 0.9, elles sont donc fortement liées. Nous allons devoir en garder qu’une seule afin de ne pas introduire de multicolinéarité dans les modèles. Nous remarquons que ces 2 variables sont elles-mêmes fortement corrélées avec la variable ***public_vote*** avec des coefficients de corrélation d’environ 0.85 (pour avg_vote) et 0.70 (pour critics_vote). Nous constatons d’autres groupes de variables qui sont fortement corrélées, comme la variable ***tension*** qui l’est à la fois avec la variable ***rhythm*** et ***effort***. Nous avons également un fort lien de corrélation entre les variables ***effort*** et ***humor*** de l’ordre de 0.60. Nous savons à présent que nous allons devoir passer par une phase de sélection de variables afin de ne pas introduire de variables corrélées entre elles dans les modèles. 
  
### 2. Les variables qualitatives <a name="test14"></a>
  
  A présent que nous avons étudié nos variables quantitatives, nous pouvons passer à l’étude de notre variable qualitative. La seule variable qualitative que nous avons dans notre base est celle que nous avons créée afin de connaitre l’origine géographique de nos pays. 
  
  *Figure n°12 : Boxplot entre la variable géographique et les variables quantitatives*
  
  ![image](https://user-images.githubusercontent.com/116641409/214413376-7e3f3735-eb2a-48e1-9bf3-734019a42bd2.png)

  *Source : Dossier SVM et Réseau de neurones, François Lebrument et Emma Weiss-Blanchard*
  
  Afin de voir si ce nouvel indicateur nous apporte de l’information, nous avons réalisé des boxplot entre la variable qualitative du lieu géographique de la production du film et les autres variables quantitatives. Nous constatons que selon les variables, il y a des différences de répartition en fonction de l’origine géographique des films. Si nous prenons le graphique avec la variable ***xp_casting***, nous voyons qu’il y a plus de films avec un casting expérimenté en Europe que dans les 2 autres zones. La médiane du nombre d’expérience par film est nettement plus faible pour les pays provenant de l’international. Nous constatons le même phénomène lorsque nous prenons la variable ***per_comedy_casting***. 
  
### 3. La variable à expliquer <a name="test15"></a>
 
  Pour finir l’analyse des statistiques descriptives, nous pouvons nous intéresser  à notre variable à expliquer. 
  
  *Figure n°13 : Boxplot entre le genre des films et les différentes variables explicatives*
  
  ![image](https://user-images.githubusercontent.com/116641409/214417964-02f046d1-443b-4f88-9bd3-fbd8d347b543.png)

   *Source : Dossier SVM et Réseau de neurones, François Lebrument et Emma Weiss-Blanchard*
  
  Nous avons réalisé dans un premier temps des boxplots entre les 2 genres de films et les variables explicatives quantitatives. Ils nous permettent de mettre en avant les différences entre les 2 types de films selon les variables explicatives. Si nous prenons l’indicateur ***ton_global***, nous voyons que les films comiques ont à la fois un premier et troisième quartiles plus élevés ainsi que la médiane, ce qui signifie qu’ils ont une description avec des sentiments exprimés plus positifs que les films dramatiques. A l’inverse, nous remarquons que les films dramatiques possèdent un Q1, M2 et Q3 plus élevés concernant les notes données par le public. 
  
  Afin d’affiner notre analyse, nous pouvons réaliser un violin graphic pour voir les queues de distribution en fonction des genres de films et des variables. Pour réaliser ce graphique, les variables ont été standardisées, donc nous ne pourrons pas interpréter les valeurs de ce graphique.
  
  *Figure n°14 : Violin plot*
  
  ![image](https://user-images.githubusercontent.com/116641409/214418215-1b139355-a970-4813-b0dc-86d45f91c6c0.png)

  *Source : Dossier SVM et Réseau de neurones, François Lebrument et Emma Weiss-Blanchard*
  
   Globalement, nous voyons que les distributions entre les 2 modalités en fonction des variables sont relativement proches. Par exemple, avec la variable ***ton_global***, nous voyons que les observations se regroupent vers les mêmes endroits entre les 2 genres de films, avec des pics aux 2 extrémités. De plus, ce graphique nous permet également de visualiser les queues de distributions des variables. Certaines variables, comme ***duration*** ou ***total_vote***, ont des distributions de valeurs beaucoup étendues avec moins de valeurs vers les extrémités. 
  
  
 Ensuite, nous avons réalisé un test d’égalité de moyenne entre la variable à expliquer et les variables explicatives. Ce test nous permet de vérifier si les 2 genres de films ont des moyennes statistiquement différentes pour chaque variable explicative. Nous posons l’hypothèse nulle selon laquelle les moyennes entre les 2 modalités sont égales, et l’hypothèse contraire où les moyennes sont différentes.
  
  *Table n°8 : Test d’égalité des moyenne*
  
  | Variables        | P-value obtenue   | 
| ------------- |:-------------:| 
| **Year**     | 0.0 |
| **Duration**     | 0.0 |
| **Avg_vote**     | 0.0 |
| **Critics_vote**     | 0.0 |
| **Public_vote**     | 0.0 |
| **Total_votes**     | 0.0 |
| **Humor**     | 0.0 |
| **Rythme**     | 0.0 |
| **Effort**     | 0.0 |
| **Tension**     | 0.0 |
| **erotisme**     | 0.0 |
| **Ton_global**     | 0.0 |
| **Per_comedy_casting**     | 0.0 |
| **Xp_casting**     | 0.0 |
  
  *Source : Dossier SVM et Réseau de neurones, François Lebrument et Emma Weiss-Blanchard*
  
 Le tableau ci-dessus donne les différentes p-values obtenues par chaque variable au test d’égalité des moyennes. Pour chaque variable, nous avons une p-value de 0.0, ce qui signifie que nous refusons l’hypothèse nulle au seuil de risque de 1%, ainsi les moyennes entre les 2 styles de films sont différentes pour chaque variable. Cependant, au vu du grand nombre d’observations présent dans notre base, nous pouvons remettre en question la véracité de ce test. Plus il y a un nombre important d’observations, et plus le test sera souple. Il faut donc prendre les résultats de ce test comme des indications.
  
 Pour finir, nous pouvons regarder la distribution des valeurs entre la variable à expliquer et la variable qualitative de notre base de données.
  
  
  *Figure n°15 : Lieu d’origine des films selon le genre*
  
  ![image](https://user-images.githubusercontent.com/116641409/214420677-ebfc9c33-6638-4fcb-a2d7-be9207ebc4cb.png)
  
  *Source : Dossier SVM et Réseau de neurones, François Lebrument et Emma Weiss-Blanchard*
  
Le graphique nous rappelle que films dramatiques sont légèrement plus représentés que les films comiques au sein de notre base de données, et nous permet également de constater des différences entre les proportions des 2 genres de films selon l’origine géographique. En effet, il y a une plus grande proportion de films qui sont originaires de l’international pour les films dramatiques par rapport aux films humoristiques. En revanche, nous constatons que ce soit pour les films dramatiques ou comiques, les proportions de films provenant d’Europe ou du nord de l’Amérique sont très proches. 

## C. Traitement des variables corrélées <a name="subparagraph7"></a>

Nous avons pu remarquer précédemment des corrélations fortes entre 2 groupes de variables. Afin de s’assurer de ne pas inclure de multi-colinéarités dans nos modèles, nous avons réalisé des sélections de variables et des analyses en composantes principales (ACP).	
	
### 1. Analyses en Composantes Principales <a name="test16"></a>
	
Pour à la fois éviter d’introduire de la multicolinéarité dans les modèles et réduire le nombre de dimensions de notre base, nous décidons de regrouper les variables corrélées entre elles grâce à la méthode de l’analyse en composantes principales. La réalisation d’ACP va nous permettre de créer des variables latentes synthétisant les informations de groupe de variables trop corrélées entre-elles (|corrélation| >0.5).
	
Les ACP ont été réalisées sur les données standardisées. Les ACP ont été créées sur l’échantillon train, puis appliqué sur l’échantillon train et test (pour ne pas inclure d’informations de l’échantillon de test dans la mise en place des ACP).	
	
Voici les résultats de nos 2 ACP, réalisées sur les 2 groupes de variables corrélées.
	
*Tableau n°9 : Regroupement des ACP*	
	
| Variables à regrouper       | Nombre de dimensions conservées et inertie associée  |  Noms des variables latentes  | 
| ------------- |:-------------:| :-------------:|
| **Avg_vote, Critics_vote et Public_vote**     | 1er axe : 0.87 | PC_vote |
| **Tension, Effort et Ehythm**     | 1er axe : 0.70 et 2ème axe : 0.19|	PC1_ter et PC2_ter |
	
*Source : Dossier SVM et Réseau de neurones, François Lebrument et Emma Weiss-Blanchard*
	
Concernant la 2ème ACP, la variable ***humor*** est très fortement corrélée avec ***rhythm***. Nous ne l'avons pas inclus dans l’ACP pour différentes raisons. En premier, les	deux premières composantes de l’ACP avec ***humor*** représentent largement moins d’inertie (pour PC1 et pour PC2) que l’ACP sans ***humor*** . Puis, la sélection des variables indique que ***humor*** est la variable la plus discriminante du genre du film (contrairement aux variables latentes de l’ACP avec humor). Nous avons donc fait le choix de ne pas synthétiser cette variable dans une ACP, malgré sa forte corrélation de 0.54 avec la variable latente ***PC_2_ter***.
		
### 2. Sélection des variables <a name="test17"></a>

Nous avons réalisé des sélections de variables pour observer celles considérées comme les plus discriminantes pour le genre. Les sélections de variables sélectionnent itérativement la variable la plus discriminante pour l’inclure dans le modèle. Lors de nos modélisations, nous avons inclus l’ensemble de nos variables, car nous ignorons le nombre optimal de variables explicatives à inclure dans le modèle (sachant que le nombre optimal n’est pas le même entre les  modèles ce qui complique la tâche).
	
Pour la sélection de variables, nous avons utilisé 5 méthodes pour ensuite réaliser une moyenne des classements pour chaque variable. Les méthodes employées sont : 
	
•	Une Recursive Feature Elimination (RFE) basée sur la régression logisitique,
	
•	Une RFE basée sur un arbre de décision,
	
•	Une Sequential Forward Selection basée sur un Random Forest,
	
•	Une Sequential Backward Selection basé sur un Random Forest,
	
•	Une sélection univarié des variables par la méthode des KBest.

Ces sélections de variables proposent des classements différents. Nous avons moyenné le classement, pour avoir un classement final.

*Tableau n°10 : Classement des variables sélectionnées*

 | Classement final de sélection de variables *avant* les ACP      |    | Classement final de sélection de variables *après* les ACP   |  |
| ------------- |:-------------:| :-----:| :-----:|
| **Humor**     | 0.0 | **Humor**     | 0.0 |
| **Tension**     | 1.6 | **PC1_ter** | 2.0 |
| **per_comedy_casting**     | 3.2 | **per_comedy_casting**   | 2.4 |
| **Effort**     | 3.4| **Xp_casting** | 5.2 |
| **Duration**     | 6.6 | **PC_vote** | 5.2 |
| **Xp_casting**     | 8.2 | **Year**  | 6.4 |
| **Avg_vote**     | 8.4 | **Duration** | 7.0 |
| **Year**     | 8.8 | **PC2_ter** | 7.0 |
| **Inter**     | 9.4 | **ton_global** | 7.2 |
| **Public_vote**     | 9.8 | **Inter** | 8.0 |
| **ton_global**     | 10.0 | **Total_votes** | 9.4 |
| **Critics_vote**     | 10.2 | **EU** | 9.8 |
| **EU**     | 10.6| **NA**  | 10.4 |
| **Erotism**     | 10.8 | **Erotism**  | 11.0 |
| **Total_votes**     | 11.0 | 
| **NA**     | 11.6 |  |  |
| **Rhythm**     | 12.4 |  |  |
	
*Source : Dossier SVM et Réseau de neurones, François Lebrument et Emma Weiss-Blanchard*
	
Nous pouvons remarquer que les variables latentes des ACP sont classées haut dans les sélections de variables. Cela appuie notre utilisation des ACP, car certaines variables comme ***rhythm*** sont moins discriminantes seules que rassemblées en variables latentes. Par ailleurs, la variable ***humor*** est sélectionnée en première position dans chaque sélection de variables. Cela signifie que c’est la variable qui discrimine le mieux le genre du film. D’autres variables comme ***per_comedy_casting, xp_casting*** ou ***duration*** sont elles aussi plutôt discriminantes, en étant dans la partie haute du classement. Néanmoins, les variables du continent de production ***NA*** et ***EU***, ainsi que ***erotism*** et total_votes se trouvent largement dans la partie basse des sélections de variables.
	
Par ailleurs, le classement des sélections de variables nous ont permis de prendre des décisions au début de notre analyse. Par exemple, initialement nous avons essayé de discrétiser les variables **per_comedy_casting** et **per_comedy_real** en 3 classes (cf partie I.B.3). Ces classes discrétisées apparaissaient dans les 5 dernières variables sélectionnées en moyenne (la vérification étant faite indépendamment pour la discrétisation des 2 variables), alors que dans le classement actuel, la variable per_comedy_casting est sélectionnée en 3ème position. Les sélections de variables nous ont confortés dans nos prises de décision pour ces variables.
	
Cependant, nous n’exploitons pas les résultats de la sélection des variables pour les modèles. La première raison est que les modèles Random Forest ont déjà une approche similaire. De plus, cela inclurait trop de complexité pour correctement définir les modèles, notamment avec le tuning des hyperparamètres (par exemple, nous nous posons des questions sur le moment où nous devons réaliser la sélection de variables, si cela doit avoir lieu avant ou après le tuning des hyperparamètres). Puis, ne pas inclure certaines variables pourraientt retirer des liens « cachés » entre certaines variables explicatives, notamment pour les réseaux de neurones qui sont en capacités de détecter des liens subtils entre différentes variables. Enfin, nous ne pouvons pas définir précisément le nombre de variables optimales de variable à inclure. Ainsi,  nous décidons d’inclure nos 14 variables.	
	
# III. Modélisation <a name="paragraph4"></a>
  
  A présent que nous avons notre base de données finale, nous pouvons passer à la création des modèles.
  
## A. Modification de la base <a name="subparagraph8"></a>
	
### 1. Découpage du dataset <a name="test18"></a>
  
Dans un premier temps, nous devons découper notre base de données en deux sous-ensembles (train et test). Cela va nous permettre d’entrainer nos modèles sur le sous-ensemble train, et de tester leurs performances avec le sous-ensemble test. Nous avons décidé d’appliquer une répartition de70% des effectifs pour le train et 30% pour la base test.
  
### 2. Standardisation <a name="test19"></a>
  
Afin que nous puissions réaliser les différents modèles, nous devons standardiser les variables quantitatives présentes dans la base train que nous avons créée précédemment. Nous avons appliqué la même standardisation de la base train, sur la base test. 
Cette procédure nous permet d’améliorer la qualité de nos données en les mettant à la même échelle. Par ailleurs, les corrélations et liens entre les variables restent inchangés.
	
### 3. Présentation des indicateurs de performances des modèles <a name="test20"></a>

Pour mesurer la qualité des modèles, leurs performances et les comparer, nous utilisons les indicateurs suivants :
	
• **Matrice de confusion :** il s’agit d’une matrice de taille 2x2 permettant la comparaison entre les valeurs observées (en ligne) et les valeurs prédites (en colonne). Le nombre d’observations situées dans la case 0-0 et celle dans la case 1-1 sont celles correctement prédites. Un modèle de bonne qualité maximisera le nombre d’observations dans ces cases et minimisera les observations en 0-1 et 1-0, qui correspondent aux erreurs. La matrice de confusion est un bon indicateur, visuel et interprétable.
	
• **Qualité du modèle et taux d’erreur :** il s’agit de deux indicateurs basés sur la matrice de confusion. La qualité du modèle correspond au nombre d’observations correctement prédites, sur le nombre d’observation total. Le taux d’erreur est simplement son complémentaire. Nous cherchons à maximiser à 1 la qualité du modèle, donc à minimiser à 0 le taux d’erreur.
	
• **Recall et Precision :** il s’agit de deux indicateurs basés sur la matrice de confusion. Le Recall permet d’identifier la proportion de résultats positifs réellement identifiés. La Precision permet d’identifier la proportion d’identifications positives correctement prédites. Ces deux indicateurs ont peu d’utilité pour nous, étant donné que nous n’avons pas un genre de film à mieux prédire que l’autre.	
	
• **F1-score :** il s’agit d’une combinaison du Recall et de la Precision. C’est l’indicateur le plus complet sur la qualité de nos modèles. Néanmoins, il est difficilement interprétable. Nous cherchons à le maximiser à 1.	
	
• **AUC (Area Under Curve) :** il s’agit d’un indicateur basé sur la courbe ROC. Cette dernière représente la sensibilité en fonction de 1 – spécificité pour toutes les valeurs seuils possibles. L’AUC correspond à l’aire sous la courbe. Si l’AUC est égale à 0.5, cela signifie que le modèle prédit aléatoirement. Nous cherchons à maximiser à 1 l’AUC.
	
Nous calculons ces indicateurs sur les bases train et test pour chaque modèle. Nous cherchons à réduire l’écart entre les indicateurs des 2 bases. Si les indicateurs de train sont supérieurs à ceux du test et que l’écart est trop élevé, le modèle est en sur-apprentissage. C’est-à-dire, qu’il apprend trop sur l’échantillon d’apprentissage et qu’il généralise mal sur de nouvelles données. Le choix du meilleur modèle est un compromis entre des indicateurs élevés et l’écart le plus faible entre ceux de la base train et test.
	
Maintenant que nous avons présenté les indicateurs que nous allons utiliser, nous pouvons expliquer les modèles utilisés et les interpréter.		
	
## B. Modèles de Random Forest <a name="subparagraph9"></a>
	
### 1. Différents modèles de Random Forest et choix du modèle final <a name="test21"></a>
	
Afin de trouver le modèle le plus adapté à nos données pour trouver les meilleurs résultats, nous avons dans un premier temps mis en place 5 modèles de type Random Forest. Le principe des modèles Random Forest est de segmenter le genre des films par la variable qui discrimine le mieux ces valeurs. Ainsi, par plusieurs itérations de segmentation, plusieurs nœuds terminaux sont créés dans l’objectif d’être le plus pure possible.
	
• **Arbre de décision :** Cette approche est la base des méthodes Random Forest. Cette méthode segmente les valeurs du genre du film par des variables discriminantes. Elle est représentée sous la forme d’un arbre ce qui la rend interprétable. Dans notre cas, nous utilisons un arbre de décision de type classification dont l’objectif est de prédire une catégorie. La prédiction renvoyée est la catégorie ayant la plus grande fréquence dans le nœud terminal dans lequel l’observation est arrivée. Cette méthode a facilement tendance à sur-apprendre. Nous avons donc mis en place des règles d’arrêt pour le rendre plus performant.	
	
• **Random Forest :** Cette approche est une généralisation des arbres de décision avec l’introduction d’une perturbation aléatoire. Plusieurs arbres de décision sont créés en parallèle. La prédiction d’une valeur correspond à la catégorie majoritaire prédite de l’ensemble des arbres.	
	
• **Bagging :**  Cette approche consiste à bootstrapper aléatoirement les données de la base train. Cette méthode à pour objectif de réduire la variance en rendant les arbres plus indépendants entre eux.
	
• **XGBoost :** L’eXtreme Gradient Boosting (XGBoost) fait partie des méthodes de Boosting. Le Boosting s'entraîne sur les sous-échantillons de manière suéquentielle (le Bagging le fait parallèlement). Puis, les individus sont pondérés en fonction de leur erreur de prédiction sur le modèle précédent. Le XGBoost se démarque par une approche plus régularisée dans la fonction de perte pour mieux contrôler le sur-apprentissage.	
• **LightGBM :** Le Light GBM fait aussi partie des méthodes de Boosting. Il se démarque par une meilleure efficacité et une meilleure scalabilité. Néanmoins, il a plus tendance à sur-apprendre, il est donc moins robuste que le XGBoost par exemple.

Ainsi, nous avons réalisé ces 5 modèles différents. Afin de sélectionner celui avec les meilleures performances pour l’optimiser, nous avons appliqué une cross-validation.
	
*Figure n°16 : Evolution de l’accuracy selon chaque modèle Random Forest en fonction de la cross-validation*

![image](https://user-images.githubusercontent.com/116641409/215761614-b98a276b-74ce-436f-8339-03f982d0372a.png)
		
*Source : Dossier SVM et Réseau de neurones, François Lebrument et Emma Weiss-Blanchard*	
Le graphique ci-dessus nous montre les résultats obtenus suite au 5 cross-validation. Nous constatons que le modèle nous donnant les moins bons résultats est l’arbre de décision, nous ne le sélectionnerons pas. Les 4 autres modèles ont des accuracy plus proches. Cependant, nous voyons que le modèle LightGBMobtient globalement de meilleures accuracy (0.876 de moyenne). Ainsi, nous décidons de garder ce modèle et de l’optimiser afin d’améliorer ses performances.
	
De plus, en observant les résultats sans cross-validation, nous remarquons un sur-apprentissage par les méthodes Bagging, XGBoost et LightGBM. Pour les 2 premiers, le F1-score est égal à 1 sur la base train (score parfait), mais généralise mal avec 0.88 sur la base test. Le LightGBM laisse une marge d’erreur avec un F1-score moins élevé de 0.95 sur la base train, mais un écart moins élevé avec le F1-score de la base test (0.88). Pour ce qui est du Random Forest, le F1-score de la base train et test sont de 0.87, ce qui signifie que ce modèle ne sur-apprend pas. Nous gardons également ce modèle pour l’optimisation des hyperparamètres.	
	
	
### 2. Optimisation du meilleur modèle <a name="test22"></a>
	
Maintenant que nous avons sélectionné les 2 modèles finaux des Random Forest, nous allons optimiser leurs hyperparamètres. Pour cela, nous allons utiliser un Random Search. Nous avons défini un intervalle de valeurs à tester pour chaque hyperparamètre, et l’algorithme testera des combinaisons aléatoires d’hyperparamètres. Les hyperparamètres communs à nos deux modèles sont la profondeur de l’arbre (max_depth) et le nombre d’arbre créé (n_estimators). Le LightGBM possède aussi comme hyperparamètre le nombre de modèles de correction successive créée.	

*Tableau n°11 : Récapitulatif des valeurs pour les hyperparamètres*

| Hyperparamètres      | Valeurs testées   | Valeurs finales pour le Random Forest |  Valeurs finales pour le LightGBM |
| ------------- |:-------------:| :-----:| :-----:|
| **max_depth**     | 2 : 50] | 37  | 4  |
| **n_estimators**     | [20 : 500] | 409 | 117  |
| **min_samples_split**     | [0.05, 0.5]  | 270  | -  |
| **class_weigth**     | [None, balanced] | None | -  |
| **n_jobs**     | [1 : 12] | 14.23013875104115 | 7  |
	
*Source : Dossier SVM et Réseau de neurones, François Lebrument et Emma Weiss-Blanchard*
	
Le tableau ci-dessus nous indique en premier les hyperparamètres testés, et les ceux obtenus pour les 2 modèles. Entre ceux communs aux 2 modèles, nous voyons que les hyperparamètres pour le random forest sont plus élevés. 	
	
	
### 3. Interprétation des résultats des modèles optimisés <a name="test23"></a>

À présent, nous allons présenter les indicateurs de performance de la base test de nos modèles Random Forest optimisés. Nous préférons observer la base test, puisque c’est sur celle-ci que nous réalisons les prédictions. Toutefois, nous prêtons une attention particulière à l’écart entre les indicateurs du train et du test pour s’assurer que nous ayons un sur-apprentissage le plus faible possible. 
	
*Tableau n°12 : Indicateur de performances de la base test des modèles Random Forest*

|  Indicateurs obtenus sur l'échantillon test | Random Forest  | LightGBM |
| ------------- |:-------------:| :-------------:|
| **Taux d’erreur**     | 0.15 | 0.12 |
| **F1 Score**     | 0.85 | 0.88 |
| **Ecart F1 score entre train et test**     | 0 | 0.01 |
| **AUC**     | 0.93 |	0.95 |

*Source : Dossier SVM et Réseau de neurones, François Lebrument et Emma Weiss-Blanchard*
	
Ce tableau des indicateurs de performance montre que le LightGBM optimisé est légèrement meilleur au Random Forest optimisé. C’est en cohérence avec les méthodes employées. En effet, le LightGBM est supposé être une amélioration du Random Forest. De plus, nous constatons que ces 2 modèles ne réalisent pas de sur-apprentissage car l’écart entre les F1 score entre la base train et test être proche voire égal à 0.
	
*Figure n°17: Importance des variables selon le modèle LightGBM*
	
![image](https://user-images.githubusercontent.com/116641409/215761850-75f9f29b-802f-4dc0-87fc-ce5756634cbc.png)
	
*Source : Dossier SVM et Réseau de neurones, François Lebrument et Emma Weiss-Blanchard*
	
Ce graphique présente l’importance des variables pour prédire le genre des films. La variable ***humor*** est celle qui discrétise le mieux le genre. Les variables ***per_comedy_casting, PC_ter, PC_vote*** et ***year*** sont elles aussi fortement présentes dans les arbres de décision du modèle LightGBM. Les variables du continent (***EU, NA*** et ***Inter***) de production du film sont largement les variables les moins importantes dans le modèle LightGBM. 
	
*Figure n°18 : Importance des variables selon le modèle LightGBM*
	
![image](https://user-images.githubusercontent.com/116641409/215633416-3ec9ec31-8c0c-4de6-8a43-3084e4932d06.png)	
	
*Source : Dossier SVM et Réseau de neurones, François Lebrument et Emma Weiss-Blanchard*
	
Ce graphique ci-dessus représente la courbe d’apprentissage du modèle basé sur la qualité du modèle, en fonction du pourcentage de découpage de l’échantillon train et test. Nous pouvons remarquer que plus l’échantillon train possède de données, plus le modèle réduit le sur-apprentissage. Néanmoins, les performances des prédictions de l’échantillon test augmentent que faiblement lorsque le modèle est plus entraîné. Ce qui signifie qu’il semble préférable d’avoir un modèle entraîné sur au moins 70 % des observations. La qualité des prédictions sur la base test restera assez similaire.	

## C. Modèles SVM <a name="subparagraph10"></a>
	
### 1. Différents modèles de SVM et choix du modèle final <a name="test24"></a>
  
Afin de trouver le modèle le plus adapté à nos données pour trouver les meilleurs résultats, nous avons également voulu  appliquer 5 modèles de Support Vector Machine et un modèle de régression logistique. Le principe des modèles de SMV est de maximiser l’écart entre les points extrêmes (appelés marges) entre les différents groupes. 

  •	__Régression logistique__ : L’objectif de ce dossier étant de prédire le genre des films (dramatiques ou comiques), nous avons décidé de réaliser un modèle à régression logistique qui permet de répondre à des problématiques de classification binaire. 
  
   •	__SVC linéaire__ : Puis, nous avons réalisé un SVC linéaire. Ce modèle s’utilise notamment lorsque les 2 groupes sont linéairement séparables. Il obtient rarement de meilleurs résultats que les autres types de modèles, car les données sont ne sont généralement pas linéairement séparables, mais nous avons tout de même voulu tester son efficacité sur nos données. 
  
  •	__SGD Classifier__ : Enfin, nous avons réalisé un modèle de type SGD Classifier. Il se différencie du modèle SMV linéaire par son optimiseur qui utilise une descente de gradient. Si nous le choisissons comme modèle final, nous aurons 3 hyperparamètres à optimiser. Nous aurons la fonction de perte, le maximum d’itération et le taux d’apprentissage.
  
  •	__SVC avec un kernel linéaire__ : Ensuite, nous avons voulu comparer les résultats obtenus par le SVC linear avec ceux obtenus par un SVC avec un kernel linéaire. Les 2 méthodes sont légèrement différentes car elles n’utilisent pas les mêmes fonctions de coût pour l’optimisation et qu’elles ont des approches différentes pour les problèmes multi-classes. En effet, les modèles SVC linear utilisent la méthode du One-vs-all et les modèles SVC avec un kernel linéaire utilisent la méthode du One-vs-one. 
  
  •	__SVC kernel polynomial__ : Nous avons également testé des modèles qui s’adaptent mieux à des données qui ne se séparent pas linéairement. Nous avons en premier testé un SVC avec un noyau polynomial. Dans un premier temps, nous garderons les hyperparamètres définis par défaut. Cependant, si ce modèle est retenu pour être le modèle final, nous devrons optimiser ses hyperparamètres. Nous aurons le degré polynomial (hyperparamètres d), les marges (hyperparamètres C) ainsi que le coefficient 0 (hyperparamètres r) à optimiser.
  
 •	__SVC kernel rbf__ : Nous avons ensuite appliqué un SVC avec un noyau Radial Basis Function (rbf). Ce type de noyau permet de réaliser des calculs de similarité.  Si le modèle final est celui-ci, nous aurons 2 hyperparamètres à optimiser, la marge (hyperparamètre c) et la fonction de distribution (hyperparamètre gamma).
  
  Ainsi, nous avons réalisé ces 6 modèles différents. Afin de sélectionner celui avec les meilleures performances pour l’optimiser, nous avons appliqué une cross-validation.
  
*Figure n°19 : Evolution de l’accuracy selon chaque modèle en fonction de la cross-validation*
  
![image](https://user-images.githubusercontent.com/116641409/215762103-1a501586-9cf4-426d-9c7c-e4eceadfd2c2.png)
  
   *Source : Dossier SVM et Réseau de neurones, François Lebrument et Emma Weiss-Blanchard*
  
  Le graphique ci-dessus nous montre les résultats obtenus suite au 5 cross-validation. Nous constatons que le modèle nous donnant les moins bons résultats est le SGD Classifier, nous ne le sélectionnerons pas. Les 5 autres modèles ont des accuracy plus mélangées. Cependant, nous voyons que le modèle SVC avec un noyau de type Radial Basis Function obtient globalement de meilleures accuracy. Ainsi, nous décidons de garder ce modèle et de l’optimiser afin d’améliorer ses performances.
  
### 2. Tuning du meilleur modèle <a name="test25"></a>
  
  Maintenant que nous avons sélectionné le modèle final pour les SVM, nous allons optimiser ses hyperparamètres. Pour cela, nous allons utiliser un Random Grid Search. Nous allons définir un intervalle de valeurs à tester pour chaque hyperparamètre, et l’algorithme testera des combinaisons aléatoires d’hyperparamètres. Nous avons mis un intervalle de 0.001 à 0.01 pour l’hyperparamètre gamma et un intervalle de 0.5 à 20 pour celui des marges.
  
  *Tableau n°13 : Récapitulatif des valeurs pour les hyperparamètres*
  
| hyperparamètres        | valeurs obtenues  | 
| ------------- |:-------------:| 
| **Gamma**     | 0.008356509792325835 |
| **Marge**     | 14.23013875104115 |
  
  *Source : Dossier SVM et Réseau de neurones, François Lebrument et Emma Weiss-Blanchard*
  
 Ainsi, les valeurs qui optimisent le modèle SVC avec un noyau rfb sont les suivantes.
	
Afin de pouvoir vérifier si nous obtenons bien de meilleurs résultats avec notre modèle optimisé, nous appliquons de nouveau une cross-validation entre notre modèle sans et avec tuning. 

  *Figure n°20 : Evolution de l’accuracy pour le modèle avec et sans tuning*
  
![image](https://user-images.githubusercontent.com/116641409/215762193-193d69fe-2347-462d-b2de-fc21e038203d.png)
  
   *Source : Dossier SVM et Réseau de neurones, François Lebrument et Emma Weiss-Blanchard*
  
  Grâce au graphique numéro 19, nous voyons que notre modèle avec les hyperparamètres obtient de meilleures performances. Cela nous conforte dans le choix de ce modèle.

### 3. Interprétation du modèle SVM optimisé <a name="test26"></a>

Nous prenons les valeurs trouvées pour les hyperparamètres afin de réaliser le nouveau modèle optimisé. Nous obtenons les résultats suivants. 

*Tableau n°14 : Résultats du SVC avec un noyau RBF tuné*
	
|  | Indicateurs obtenus sur l'échantillon test | 
| ------------- |:-------------:| 
| **Taux d’erreur**     | 0.12 |
| **F1 Score**     | 0.88 |
| **Ecart F1 score entre train et test**     | 0.00 |
| **AUC**     | 0.9496 |

*Source : Dossier SVM et Réseau de neurones, François Lebrument et Emma Weiss-Blanchard*

Nous voyons que ce modèle optimisé obtient un taux d’erreur à 0.12 et un F1 score de 0.88 pour l’échantillon test. De plus, l’écart entre les F1 scores de l’échantillon train et test est de seulement 0.00, ce qui nous fait supposer qu’il n’y a pas de sur-apprentissage de la part du modèle sur l'échantillon train.

Afin de compléter notre analyse sur le modèle optimisé obtenu, nous réalisons la courbe d’apprentissage. 

*Figure n°21 : Learning curve du SVC rbf tuné*

![image](https://user-images.githubusercontent.com/116641409/215608525-3028c735-d356-4eee-ada4-54fdd98cedd3.png)	
	
*Source : Dossier SVM et Réseau de neurones, François Lebrument et Emma Weiss-Blanchard*
	
Nous pouvons remarquer que plus l’échantillon train possède de données, plus le modèle réduit le sur-apprentissage. Néanmoins, les performances des prédictions de l’échantillon test augmentent que très faiblement lorsque le modèle est plus entraîné. Ce qui signifie qu’il semble préférable d’avoir un modèle entraîné sur au moins 50% à 70 % des observations. 	
	
## D. Modèles Réseaux de neurones <a name="subparagraph11"></a>
	
### 1. Modèle et tuning <a name="test27"></a>
	
En dernier, nous avons construit un modèle de réseau de neurones avec une architecture feedforward. Ce type de modèle possède une architecture plus complexe que les modèles appliqués précédemment car ils ont plusieurs hyperpamètres. Pour cette raison, nous avons décidé d’appliquer directement un algorithme de Grid Search afin de réaliser un modèle de réseau de neurones optimisé. Nous avons voulu optimiser 4 hyperparamètres. 
	
*Tableau n°15 : Hyperparamètres de réseau de neurones*

| Hyperparamètres      | Valeurs testées   | Valeurs finales|
| ------------- |:-------------:| :-----:|
| **Nombre de couches cachées**     | [1, 2, 3] | 3  |
| **Nombre de neurones**     | [50, 75, 100, 150, 200] | 100 |
| **Batch size**     | [50 : 400]  | 270  |
| **Epochs**     | [10, 20, 30, 40, 50] | 20 |
	
*Source : Dossier SVM et Réseau de neurones, François Lebrument et Emma Weiss-Blanchard*
  
Le tableau 15 nous montre les hyperparamètres que nous avons décidé de tester et les résultats obtenus par l'algorithme. Ainsi, nous avons un modèle final composé de 3 couches avec 100 neurones. Nous avons également obtenu un batch size à 270 et un nombre d’époques égal à 20. 

### 2. Interprétation du modèle optimisé <a name="test28"></a>

Nous prenons les valeurs trouvées pour les hyperparamètres afin de réaliser le réseau de neurones optimisés. Nous obtenons les résultats suivant. 	

*Tableau n°16 : Résultats du Réseau de neurones avec un Grid Search*
	
|  | Indicateurs obtenus sur l'échantillon test | 
| ------------- |:-------------:| 
| **Taux d’erreur**     | 0.12 |
| **F1 Score**     | 0.88 |
| **Ecart F1 score entre train et test**     | 0.01 |
| **AUC**     | 0.9451 |
	
*Source : Dossier SVM et Réseau de neurones, François Lebrument et Emma Weiss-Blanchard*
	
Nous constatons que ce modèle optimisé obtient un taux d’erreur de seulement 0.12 et un F1 score de 0.88 pour l’échantillon test. Il arrive à relativement bien prédire une majorité des films de la base. De plus, l’écart entre les F1 scores de l’échantillon train et test est de seulement 0.01, ainsi nous supposons que ce modèle ne comporte pas de sur-ajustement. 

## E. Comparaison des meilleurs modèles et prévisions avec de nouvelles données <a name="subparagraph12"></a>
	
### 1. Comparaions des 3 modèles optimisés <a name="test29"></a>
	
Maintenant que nous avons optimisé les modèles que nous avons sélectionnés, nous pouvons comparer leurs performances.	
	
*Tableau n°17 : Récapitulatif des résultats des modèles optimisés*	

| Hyperparamètres      | Random Forest   | LightGBM | SVM rbf | Réseau de neurones |
| ------------- |:-------------:| :-----:|:-----:|:-----:|
| **Taux d’erreur**     | 0.15 | 0.12  | 0.12  | 0.12  |
| **F1 Score**     | 0.85 | 0.88 | 0.88 | 0.88 |
| **Ecart F1 score entre train et test**     | 0 | 0.01  | 0  | 0.01  |
| **AUC**     | 0.93 | 0.95 | 0.9496 | 0.9451 |
	
*Source : Dossier SVM et Réseau de neurones, François Lebrument et Emma Weiss-Blanchard*
	
Le modèle nous donnant les moins bons résultats est le Random Forest qui a le taux d’erreur le plus élevé et de plus faibles résultats concernant le F1-score ainsi que l’AUC. En revanche, il est plus difficile de déterminer le modèle qui nous donne les meilleures performances. En effet, entre le LightGBM, SVM et RN, les taux d’erreur et les F1-score sont similaires, tous les 4 ont du mal à prédire 12% des observations de la base test. De plus, il y a très peu de différences entre les écarts de F1 score et les AUC. Ces 3 modèles sont donc similaires, nous ne pourrons pas définir lequel est le meilleur dans le cadre la prédiction des genres de films. Pour en sélectionner un, il faudrait appliquer ces modèles à de nouvelles données afin de voir si un se démarque. 	
	
### 2. Prévisions sur de nouvelles données <a name="test30"></a>
	
Pour aller plus loin dans notre analyse, nous avons souhaité vérifier la reproductibilité de notre code en ajoutant manuellement de nouvelles données. L’idée est de pouvoir ajouter de nouvelles données dans la base, sans avoir besoin de modifier le fichier csv. Ici, les données sont ajoutées manuellement, mais nous pourrions imaginer une méthode que scraping qui récupère de nouveaux films sur le site, qui tente de prédire le genre et qui les ajoute dans la base d’entrainement. L’intérêt pour nous est de vérifier que notre traitement des données et la prédiction des modèles fonctionne même sur de nouvelles données qui ne sont pas dans la base d’origine.
	
D’abord, nous sommes allé.e.s récupérer les données de 2 comédies et 2 drames sur le site [filmtv.it](https://www.filmtv.it/). Comme la base de données d’origine est récente (elle se termine au 24 novembre 2022), il est assez difficile de trouver des films n’étant pas dans la base de données. En effet, les seuls éventuellement disponibles sont les films récents, mais qui n’ont pas forcément l’entièrement des données que nous souhaitons, notamment pour les variables de votes du public qui sont assez peu complète. Ainsi, nous avons ajouté les films suivants, le 13 janvier 2023 : *Eo*, *I migliorigiorni*, *Il grande giorno* et *The Fabelmans*.
	
Ensuite, à partir des données rentrées, nous créons nos indicateurs, nettoyons les données et standardisons à partir de la standardisation de la base train. Pour le moment, les seuls éléments de nettoyage qui ne sont pas mis en place sont le traitement des NA, le traitement des valeurs atypiques et la transformation du genre en variable binaire (0 et 1).
	
	
Enfin, nous réalisons les prédictions à partir des modèles entraînés sur la base train. Seul le modèle Random Forest prédit correctement le genre des 4 films. Les 3 autres modèles réalisent une erreur sur le genre du film *The Fabelmans* prédit comme une comédie. Due au faible nombre d’observations, la qualité de ces modèles est alors de 0.75. Avec plus d’observations, les indicateurs de performance convergeront vers leur vraie valeur.
	
Ce dernier processus a été ajouté pour tenter d’automatiser la prédiction des modèles sur de nouvelles données récupérées. Même s’il reste quelques éléments du processus à améliorer, comme la récupération des données ou le nettoyage des variables, nos modèles prédisent correctement le genre. Un autre défaut de ce processus est qu’il faut récupérer les données sur le site [filmtv.it](https://www.filmtv.it/) pour avoir accès aux notes de ***rhythm, tension, erotism, …*** qui n’existent pas sur d’autres sites de notation de film.	

# Conclusion <a name="paragraph5"></a>

Dans ce projet, nous avons cherché à prédire le genre des films en se concentrant sur les comédies et les drames. Nous avons utilisé une base données fournie par le site Kaggle venant elle-même du site filmTV. 

Dans un premier temps, nous avons réalisé une analyse complète et un traitement de la base. En effet, après avoir nettoyé la base de données des valeurs manquantes qu’elle contenait, nous avons créé 3 indicateurs textuels grâce à la méthode  Natural Language Processing (NLP).  Le premier indicateur concerne l’origine géographique des films. Nous avons récupéré les noms des pays dont les films sont originaires afin de les regrouper en 3 régions distinctes, l’Europe (***EU***), l’Amérique du nord (***NA***), et les films ayant plusieurs nationalités ou venant d’un autre continent dans la catégorie international (***Inter***). A partir de la description écrite des films, nous avons pu créer un indicateur de sentiment afin de connaitre le ton global des films. Enfin, nous avons créé 2 indicateurs regroupant l’expérience du casting des films et l’expérience des réalisatrices.eurs. Cependant, nous avons gardé uniquement l’indicateur concernant l’expérience des castings, et non celui sur  les réalisatrices et réalisateurs qui nous semblait moins pertinent. Une fois nos indicateurs ajoutés à notre base, nous avons pu chercher à savoir si elle contenait des valeurs atypiques grâce à l’étude des boxplots et l’application du test ESD. Afin de supprimer les valeurs considérées comme atypiques, nous avons appliqué différents filtres sur certaines de nos variables. Ainsi, nous sommes arrivé.es à une base finale composée de 17 227 films et 19 variables. 

Dans un deuxième temps, nous avons réalisé une analyse descriptive sur notre échantillon. La réalisation de statistiques univariées nous ont permis de mieux comprendre les variables présentes dans la base de données, et de supprimer celles qui ne nous semblaient finalement pas pertinentes à conserver. Puis, nous avons également appliqué des statistiques bivariées afin d’analyser les liens entre les variables explicatives. Nous avons constaté que certaines variables de notre base étaient fortement liées. Pour cette raison, nous avons cherché à les regrouper en utilisant la méthode des analyses en composantes principales (ACP). Ainsi, nous avons créé 3 variables latentes (***PC_vote***, ***PC1_ter*** et ***PC2_ter***). Ensuite, nous avons appliqué une sélection de variables fondées sur 5 méthodes pour classer les variables selon leur importance. Cependant, nous avons décidé de conserver l’intégralité des variables afin de limiter la perte d’informations.

Après la réalisation de ces analyses et traitements, nous avons consacré la troisième partie de ce projet à la conception de modèles de Machine et Deep Learning dans l’optique de prédire le genre des films. Après avoir découpé notre dataset en un échantillon train et test notre dataset, et les avoir standardisés, nous avons créé différents modèles fondés à la fois sur des arbres de décision, du support vector marchine (SVM) et des réseaux de neurones. À partir de ces modèles, nous avons pu déterminer le genre des films présent dans la base test. Pour chaque catégorie de modèles, nous avons sélectionné celui (voire ceux pour les random forest) qui donnait les meilleures performances dans l’objectif de l’optimiser grâce à un Grid Search ou un Random Search. Ainsi, nous avons optimisé un modèle Random Forest, LightGBM, SVM avec un kernel RBF et un réseau de neurones. Le modèle Random Forest est celui qui arrive le moins bien à prédire le genre des films sur notre base test. Pour les 3 autres modèles, il est difficile de définir celui qui arrive le mieux à prédire le style de films de notre base car ils ont des performances très similaires. Ces 3 modèles optimisés ont un taux d’erreur de 0.12 et un F1-score égal à 0.88. 


# Discussion <a name="paragraph6"></a>

Cependant, nous pouvons noter différentes limites à notre étude, notamment dans la partie modélisation. En effet, lors de l’application des Random Search et Grid Search pour optimiser les modèles sélectionnés, nous avons-nous même sélectionné des intervalles de valeurs à tester pour les hyperparamètres. Pour éviter que les algorithmes d’optimisation tournent durant plusieurs heures, nous avons mis des intervalles de valeurs assez réduits. Hors, nous aurions peut-être trouvé de meilleurs modèles en mettant des plages de valeurs différentes pour les hyperparamètres. 

De plus, nous avons obtenu 3 modèles finaux optimisés avec des performances quasiment égales. Ces 3 modèles ont du mal à prédire 12% des observations notre base test. Pour améliorer cela, il serait pertinent d’ajouter de nouvelles observations dans la base d’entrainement afin que les modèles apprennent à prédire ces observations. Nous pourrions également ajouter d’autres variables afin d’améliorer les prédictions. 

Enfin, durant ce projet nous avons centré notre analyse sur seulement 2 genres de films à prédire, comique et dramatique. La base initiale contenait d’autres genres de films. Il pourrait être donc intéressant d’appliquer et d’adapter nos modèles dans le but de prédire les différents de style de films, et ainsi réaliser des prédictions multi-classes. 

Pour conclure ce projet, nous tenons à dire qu’il nous a donné l’opportunité d’utiliser des méthodes que nous n’avions pas appliquées avant (du type NLP et certains modèles de machine learning). Il nous a également permis d’approfondir nos connaissances sur les différents modèles de SVM. De plus, nous avons pu travailler sur une thématique qui nous plaisait, ce qui nous a permis de bien comprendre les données de notre base et d’aller plus loin dans nos analyses. 
(Merci Roulitoo ;) 😉)

