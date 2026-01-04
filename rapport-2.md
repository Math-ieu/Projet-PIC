# RAPPORT TECHNIQUE : DÉTECTION D'ANOMALIES INDUSTRIELLES  
## Catégorie : CABLE – Dataset MVTec AD  
### Implémentation PyTorch et évaluation multi-niveaux  

---

## TABLE DES MATIÈRES  

1. [Introduction Générale](#introduction-generale)  
2. [Méthodologie et Préparation des Données](#methodologie)  
3. [Architectures des Modèles et Processus d'Entraînement](#architecture)  
4. [Résultats Expérimentaux et Analyse des Performances](#resultats)  
5. [Analyse Comparative et Sélection du Modèle Optimal](#analyse)  
6. [Conclusion Générale et Perspectives](#conclusion)  
7. [Mise en Production](#production)  
8. [Références](#references)  
9. [Annexes Techniques](#annexes)  

---

## 1. INTRODUCTION GÉNÉRALE  

### 1.1 Contexte et Motivation  

La détection automatique d'anomalies dans les environnements industriels représente un enjeu technologique majeur pour l'industrie 4.0. Dans le secteur de la fabrication de câbles, les défauts peuvent survenir à différentes étapes de production et compromettre la fiabilité des produits finaux. Les méthodes d'inspection visuelle traditionnelles, bien qu'encore largement utilisées, présentent des limitations significatives en termes de coût, de reproductibilité et de capacité à traiter de grands volumes de production.  

Ce projet s'inscrit dans cette problématique industrielle en proposant une solution automatisée basée sur l'apprentissage profond pour la détection, la classification et la localisation d'anomalies sur des câbles manufacturés. L'objectif principal est de développer un système capable d'identifier avec précision différents types de défauts tout en minimisant les fausses alarmes, un paramètre critique dans les environnements de production où les arrêts de ligne ont un coût économique important.  

### 1.2 Objectifs Spécifiques  

Le projet vise à atteindre trois objectifs opérationnels complémentaires :  

1. **Détection binaire** : Distinguer automatiquement les images de câbles normaux des images présentant des anomalies, avec un taux de détection élevé et un faible taux de fausses alarmes.  
2. **Classification multi-classe** : Identifier spécifiquement le type de défaut parmi huit catégories prédéfinies, permettant une analyse détaillée des problèmes de production.  
3. **Localisation pixel-level** : Segmenter précisément les régions défectueuses dans les images, facilitant ainsi le diagnostic et l'analyse des causes racines.  

### 1.3 Structure du Rapport  

Ce rapport présente une approche complète allant de la préparation des données à l'évaluation de différentes architectures de deep learning. La méthodologie adoptée repose sur une évaluation rigoureuse de quatre modèles distincts, chacun représentant une approche différente du problème de détection d'anomalies. Les résultats sont analysés à trois niveaux de complexité, permettant une compréhension approfondie des forces et limitations de chaque approche.  

---

## 2. MÉTHODOLOGIE ET PRÉPARATION DES DONNÉES  

### 2.1 Introduction à la Méthodologie  

La méthodologie adoptée dans ce projet repose sur une approche systématique intégrant la préparation des données, l'implémentation de plusieurs architectures de deep learning, et une évaluation multi-niveaux rigoureuse. Cette section décrit les étapes de préparation des données qui constituent la base de l'entraînement et de l'évaluation des modèles.  

### 2.2 Présentation du Dataset MVTec AD - Catégorie Cable  

Le dataset MVTec AD est reconnu comme un benchmark standard pour la détection d'anomalies industrielles. La catégorie "cable" présente des caractéristiques particulièrement intéressantes pour notre étude. Elle comprend des images haute résolution de câbles avec huit types de défauts distincts : déformation des fils, inversion de câbles, combinaisons de défauts, coupures d'isolation interne et externe, câbles ou fils manquants, et perforation d'isolation. Chaque image défectueuse est accompagnée d'un masque de segmentation pixel-level précis, permettant une évaluation complète des capacités de localisation des modèles.  

La structure originale du dataset sépare strictement les images normales (pour l'entraînement) des images anormales (pour le test). Cette organisation, bien qu'adaptée aux méthodes non supervisées, nécessite une adaptation pour les approches supervisées comme celles employées dans ce projet.  

### 2.3 Réorganisation pour l'Apprentissage Supervisé  

Pour permettre un apprentissage supervisé efficace, nous avons restructuré le dataset en fusionnant les ensembles d'entraînement et de test, puis en appliquant un split stratifié. Cette approche garantit que chaque sous-ensemble (entraînement, validation, test) contient une proportion représentative de toutes les classes, y compris les différentes catégories d'anomalies. Le split final retenu est de 60% pour l'entraînement, 20% pour la validation, et 20% pour le test, avec préservation de la distribution des classes dans chaque sous-ensemble.  

Cette restructuration est essentielle car elle permet aux modèles d'apprendre directement à partir d'exemples d'anomalies durant l'entraînement, plutôt que de se baser uniquement sur des représentations du "normal" comme c'est le cas dans les approches non supervisées.  

### 2.4 Stratégie d'Augmentation de Données Ciblée  

Le dataset présente un déséquilibre marqué entre la classe "normale" et les différentes classes d'anomalies, certaines anomalies étant particulièrement rares. Pour remédier à ce problème, nous avons implémenté une stratégie d'augmentation de données ciblée qui génère des variations artificielles des images des classes minoritaires.  

Le principe de cette stratégie est simple mais efficace : plus une classe est rare, plus elle reçoit d'augmentations. Les transformations appliquées incluent des modifications géométriques (rotation, translation, mise à l'échelle) et photométriques (ajustement de luminosité, contraste, saturation), ainsi que l'ajout de bruit et d'effets de flou. L'intensité de ces transformations est modulée en fonction de la rareté de la classe, avec une probabilité plus élevée d'application pour les classes les plus rares.  

Cette approche a permis de multiplier par 3,2 la taille du dataset d'entraînement (passant de 224 à 719 images) tout en équilibrant la répartition entre classes, créant ainsi des conditions optimales pour l'apprentissage des modèles.  

### 2.5 Pondération des Classes dans la Fonction de Perte  

En complément de l'augmentation de données, nous avons introduit des poids de classes dans les fonctions de perte. Ces poids sont inversement proportionnels à la fréquence de chaque classe, ce qui signifie que les erreurs commises sur les classes rares sont plus sévèrement pénalisées que celles sur les classes fréquentes.  

Techniquement, le poids \( w_c \) pour une classe \( c \) est calculé comme suit :  
\[ w_c = \frac{N_{\text{total}}}{N_{\text{classes}} \times N_c} \]  
où \( N_{\text{total}} \) est le nombre total d'échantillons, \( N_{\text{classes}} \) le nombre de classes, et \( N_c \) le nombre d'échantillons dans la classe \( c \).  

Cette approche assure que les modèles accordent une attention suffisante aux classes minoritaires pendant l'entraînement, améliorant ainsi leur capacité à détecter des anomalies rares mais critiques.  

### 2.6 Pipeline de Prétraitement  

Le pipeline de prétraitement comprend deux versions distinctes : une version riche pour l'entraînement (avec augmentation) et une version minimale pour la validation et le test. Les images sont redimensionnées à 224×224 pixels pour uniformiser les dimensions d'entrée, puis normalisées selon les statistiques d'ImageNet (moyenne = [0.485, 0.456, 0.406], écart-type = [0.229, 0.224, 0.225]). Cette normalisation est cruciale pour la stabilité de l'entraînement, particulièrement lors de l'utilisation de modèles pré-entraînés.  

### 2.7 Conclusion de la Section Méthodologie  

La méthodologie de préparation des données décrite dans cette section constitue un cadre solide pour l'entraînement et l'évaluation des modèles. L'approche combinant réorganisation stratifiée, augmentation ciblée et pondération des classes permet de surmonter les défis liés au déséquilibre des données et à la rareté de certaines anomalies. Ces préparations créent les conditions nécessaires pour comparer équitablement différentes architectures et évaluer objectivement leurs performances sur les trois niveaux de tâches définis.  

---

## 3. ARCHITECTURES DES MODÈLES ET PROCESSUS D'ENTRAÎNEMENT  

### 3.1 Introduction aux Architectures  

Cette section présente les quatre architectures de deep learning implémentées et évaluées dans ce projet. Chaque modèle représente une approche différente du problème de détection d'anomalies, allant d'un CNN personnalisé conçu spécifiquement pour cette tâche à des architectures pré-entraînées optimisées par transfer learning, en passant par une approche non supervisée basée sur un autoencodeur. Pour chaque modèle, nous décrivons son principe de fonctionnement, son adaptation au contexte industriel, et son architecture technique détaillée.  

### 3.2 CNN Custom (Architecture Baseline)  

#### 3.2.1 Présentation et Justification  

Le CNN Custom constitue notre architecture baseline, développée spécifiquement pour la détection d'anomalies sur des câbles industriels. Contrairement aux modèles pré-entraînés, cette architecture est conçue à partir de zéro, ce qui permet une adaptation optimale aux caractéristiques spécifiques de notre dataset. L'utilisation d'un modèle personnalisé présente plusieurs avantages : elle permet un contrôle total sur l'architecture, une optimisation fine des hyperparamètres pour notre tâche spécifique, et évite le biais potentiel introduit par un pré-entraînement sur des données génériques comme ImageNet.  

Ce modèle suit une approche multi-tâches, avec deux têtes de sortie distinctes : une pour la classification binaire (normal/anormal) et une pour la classification multi-classe (type de défaut). Cette architecture permet au modèle d'apprendre des représentations partagées pour les deux tâches, améliorant ainsi l'efficacité de l'apprentissage.  

#### 3.2.2 Architecture Technique  

L'architecture du CNN Custom est structurée en cinq blocs convolutionnels profonds, chacun comprenant des couches de convolution, de normalisation par lots (BatchNorm), et d'activation ReLU, suivis d'un pooling maximum. La profondeur progressive des blocs (64, 128, 256, 512, 512 filtres) permet au modèle d'apprendre des caractéristiques hiérarchiques, des motifs simples aux représentations complexes.  

Après les couches convolutionnelles, les features sont aplaties et passent à travers trois couches entièrement connectées avant d'être dirigées vers les deux têtes de sortie. La tête binaire utilise une activation sigmoïde pour produire une probabilité d'anomalie, tandis que la tête multi-classe utilise une activation softmax pour attribuer des probabilités aux neuf classes (huit défauts plus la classe normale).  

#### 3.2.3 Processus d'Entraînement  

L'entraînement du CNN Custom a été réalisé sur 5 époques seulement, une contrainte qui a influencé nos choix d'hyperparamètres et de régularisation. La fonction de perte totale combine deux composantes : une perte binaire (BCEWithLogitsLoss) pondérée à 60% et une perte multi-classe (CrossEntropyLoss avec poids de classes) pondérée à 40%. Cette pondération reflète la priorité accordée à la détection binaire tout en maintenant une capacité de classification fine.  

L'optimiseur Adam a été utilisé avec un taux d'apprentissage de 0,001 et un weight decay de 1e-4 pour la régularisation L2. Malgré le nombre limité d'époques, le modèle a montré une convergence rapide grâce à son architecture adaptée et aux techniques d'augmentation de données décrites précédemment.  

### 3.3 ResNet50 avec Fine-Tuning  

#### 3.3.1 Présentation et Justification  

ResNet50 est une architecture de réseau résiduel profond qui a révolutionné le domaine de la vision par ordinateur en permettant l'entraînement de réseaux extrêmement profonds grâce à ses connexions résiduelles. Dans notre projet, nous utilisons ResNet50 pré-entraîné sur ImageNet, ce qui nous permet de bénéficier de représentations riches apprises sur un large corpus d'images.  

Le choix de ResNet50 s'appuie sur plusieurs considérations. Premièrement, sa profondeur (50 couches) lui permet de capturer des caractéristiques hiérarchiques complexes particulièrement utiles pour distinguer des anomalies subtiles. Deuxièmement, son architecture résiduelle facilite l'entraînement en atténuant le problème du gradient disparaissant. Troisièmement, son large usage dans la littérature scientifique nous permet de comparer nos résultats avec des travaux antérieurs.  

#### 3.3.2 Adaptation et Architecture  

Pour adapter ResNet50 à notre tâche spécifique, nous avons remplacé sa couche de classification finale par un classifieur personnalisé comprenant deux têtes, similaire à l'approche utilisée pour le CNN Custom. Cependant, contrairement au CNN entièrement entraîné à partir de zéro, nous avons appliqué une stratégie de fine-tuning sélective : les premières couches du réseau (apprenant des caractéristiques génériques comme les bords et les textures) ont été gelées, tandis que les couches supérieures (apprenant des caractéristiques spécifiques à la tâche) ont été dégelées pour l'entraînement.  

Cette approche de transfer learning permet de conserver les représentations générales apprises sur ImageNet tout en adaptant le modèle aux spécificités de notre domaine. L'architecture finale comprend le backbone ResNet50 suivi d'un Global Average Pooling, de deux couches entièrement connectées avec dropout, et des deux têtes de classification.  

#### 3.3.3 Processus d'Entraînement  

L'entraînement de ResNet50 a également été limité à 5 époques, nécessitant une stratégie de fine-tuning agressive. Nous avons utilisé un taux d'apprentissage différencié : 1e-4 pour les couches dégelées du backbone et 1e-3 pour les nouvelles couches ajoutées. Cette approche permet aux nouvelles couches de s'adapter rapidement tout en ajustant progressivement les couches pré-entraînées.  

La fonction de perte combine la BCEWithLogitsLoss pour la classification binaire et la CrossEntropyLoss pondérée pour la classification multi-classe, avec les mêmes pondérations que pour le CNN Custom. L'optimiseur Adam a été utilisé avec les mêmes paramètres de base, mais avec un gradient clipping à 1,0 pour assurer la stabilité pendant le fine-tuning.  

### 3.4 EfficientNet-B0  

#### 3.4.1 Présentation et Justification  

EfficientNet-B0 représente l'état de l'art en matière d'efficacité architecturale pour les réseaux de neurones convolutifs. Développé par Google Research, EfficientNet utilise un principe de scaling composé qui optimise simultanément la profondeur, la largeur et la résolution du réseau. Ce modèle offre un excellent compromis entre précision et efficacité computationnelle, ce qui le rend particulièrement adapté aux applications industrielles où les ressources peuvent être limitées.  

Dans notre contexte, le choix d'EfficientNet-B0 est motivé par plusieurs facteurs. D'abord, son efficacité computationnelle permet un entraînement plus rapide et une inférence en temps réel, cruciale pour les applications industrielles. Ensuite, son architecture optimisée par recherche neuronale capture efficacement les caractéristiques pertinentes avec moins de paramètres que les modèles traditionnels. Enfin, son pré-entraînement sur ImageNet fournit une base solide pour le transfer learning.  

#### 3.4.2 Adaptation et Architecture  

Pour adapter EfficientNet-B0 à notre tâche, nous avons suivi une approche similaire à celle utilisée pour ResNet50, avec quelques optimisations spécifiques. Le backbone EfficientNet-B0 pré-entraîné a été complété par un classifieur personnalisé à deux têtes. Cependant, nous avons exploité les particularités d'EfficientNet, notamment ses blocs MBConv qui combinent des convolutions depthwise et pointwise pour une efficacité accrue.  

L'architecture finale comprend le backbone EfficientNet-B0 suivi d'un Global Average Pooling, d'une couche de squeeze-and-excitation pour l'attention sur les canaux, de deux couches entièrement connectées avec activation Swish (optimisée pour les réseaux efficients), et des deux têtes de classification. Cette architecture tire parti des optimisations d'EfficientNet tout en étant spécifiquement adaptée à notre tâche de détection d'anomalies.  

#### 3.4.3 Processus d'Entraînement  

Malgré la limitation à 5 époques, EfficientNet-B0 a montré une convergence particulièrement rapide grâce à son architecture optimisée et à son pré-entraînement efficace. Nous avons utilisé une stratégie de fine-tuning similaire à celle de ResNet50, avec des taux d'apprentissage différenciés pour les différentes parties du réseau.  

La fonction de perte reste identique aux autres modèles (combinaison pondérée de BCEWithLogitsLoss et CrossEntropyLoss pondérée), mais nous avons ajouté une régularisation supplémentaire sous forme de label smoothing à 0,1 pour améliorer la généralisation malgré le nombre limité d'époques. L'optimiseur AdamW (une version améliorée d'Adam avec décroissance de poids découplée) a été utilisé pour une meilleure régularisation.  

### 3.5 Autoencodeur Non Supervisé  

#### 3.5.1 Présentation et Justification  

L'autoencodeur non supervisé représente une approche fondamentalement différente des trois modèles précédents. Contrairement aux modèles supervisés qui apprennent à partir d'exemples étiquetés d'anomalies, l'autoencodeur est entraîné uniquement sur des images normales et détecte les anomalies comme des déviations par rapport à cette norme apprise. Cette approche présente plusieurs avantages théoriques : elle ne nécessite pas d'exemples d'anomalies pour l'entraînement, elle peut potentiellement détecter des types d'anomalies jamais vus auparavant, et elle fournit naturellement des heatmaps de localisation via l'erreur de reconstruction.  

Dans le contexte industriel, cette approche est particulièrement intéressante pour les scénarios où les anomalies sont rares ou difficiles à collecter. Elle correspond également à une philosophie différente de détection d'anomalies : plutôt que d'apprendre à reconnaître des défauts spécifiques, le modèle apprend à reconnaître ce qui est normal et signale tout écart significatif.  

#### 3.5.2 Architecture et Principe de Fonctionnement  

L'autoencodeur implémenté suit une architecture encodeur-décodeur symétrique. L'encodeur comprime l'image d'entrée en une représentation latente de faible dimension via une série de couches convolutionnelles avec stride, capturant ainsi l'essence de "la normalité". Le décodeur tente ensuite de reconstruire l'image originale à partir de cette représentation latente.  

Le principe de détection est élégant dans sa simplicité : après entraînement sur des images normales, l'autoencodeur est capable de reconstruire fidèlement de nouvelles images normales, mais échoue à reconstruire correctement des images anormales. L'erreur de reconstruction (mesurée par la différence pixel à pixel entre l'input et l'output) sert alors de score d'anomalie : une erreur élevée indique une anomalie, une erreur faible indique une image normale.  

#### 3.5.3 Processus d'Entraînement  

L'entraînement de l'autoencodeur diffère fondamentalement de celui des modèles supervisés. Il est entraîné uniquement sur des images normales, avec pour objectif de minimiser l'erreur de reconstruction (mesurée par la MSE - Mean Squared Error). Nous avons également ajouté une composante de perceptual loss, calculée comme la différence entre les features extraites par un VGG16 pré-entraîné des images originales et reconstruites, ce qui améliore la qualité perceptuelle des reconstructions.  

Malgré la limitation à 5 époques, l'autoencodeur a montré une capacité à apprendre des représentations compactes de la normalité. L'optimiseur Adam a été utilisé avec un taux d'apprentissage de 0,001, et un early stopping basé sur l'erreur de reconstruction sur l'ensemble de validation a permis d'éviter le surapprentissage.  

### 3.6 Conclusion de la Section Architectures  

Les quatre architectures présentées dans cette section représentent différentes approches philosophiques et techniques du problème de détection d'anomalies. Le CNN Custom offre une solution sur mesure, ResNet50 apporte la puissance des réseaux profonds pré-entraînés, EfficientNet-B0 optimise l'efficacité computationnelle, et l'autoencodeur explore la voie non supervisée. Chaque modèle a été entraîné dans des conditions identiques (5 époques, mêmes données, mêmes métriques d'évaluation) permettant une comparaison équitable. La section suivante présente les résultats détaillés de cette évaluation comparative.  

---

## 4. RÉSULTATS EXPÉRIMENTAUX ET ANALYSE DES PERFORMANCES  

### 4.1 Introduction aux Résultats  

Cette section présente les résultats expérimentaux obtenus par les quatre modèles sur les trois niveaux de tâches définis. L'analyse est structurée pour permettre une compréhension complète des performances de chaque modèle, de leurs forces respectives, et des limitations observées. Les résultats sont présentés avec une attention particulière aux métriques pertinentes pour le contexte industriel, notamment le rappel (pour éviter les défauts non détectés) et la précision (pour minimiser les fausses alarmes).  

### 4.2 Protocole d'Évaluation  

L'évaluation des modèles suit un protocole rigoureux et reproductible. Chaque modèle a été évalué sur le même ensemble de test, jamais vu pendant l'entraînement. Les prédictions ont été réalisées en mode inference (sans gradient), et les métriques ont été calculées sur l'ensemble complet du test. Pour assurer la robustesse des résultats, certaines expériences ont été répétées avec différentes initialisations aléatoires, bien que la contrainte de 5 époques ait limité la variabilité due à l'initialisation.  

Les métriques sont calculées à trois niveaux : classification binaire, classification multi-classe, et segmentation pixel-level. Cette approche multi-niveaux permet d'évaluer non seulement la capacité des modèles à détecter la présence d'anomalies, mais aussi à les caractériser précisément et à les localiser dans l'image.  

### 4.3 Résultats de Classification Binaire  

#### 4.3.1 Performances Globales  

La classification binaire constitue la tâche fondamentale de détection d'anomalies. Les résultats montrent des différences significatives entre les modèles :  

- **EfficientNet-B0** obtient les meilleures performances avec une AUC-ROC de 0,997, démontrant une capacité presque parfaite à séparer les images normales des images anormales. Son F1-Score de 0,91 indique un excellent équilibre entre précision et rappel.  
- **ResNet50** montre un rappel élevé (0,89) mais une précision modérée (0,41), suggérant une tendance à produire des fausses alarmes.  
- **CNN Custom** présente des performances modestes mais stables sur toutes les métriques.  
- **Autoencodeur** obtient les résultats les plus faibles en classification binaire, ce qui est attendu étant donné son approche non supervisée.  

#### 4.3.2 Analyse des Courbes ROC  

Les courbes ROC (Receiver Operating Characteristic) fournissent une visualisation complète du compromis entre le taux de vrais positifs et le taux de fausses alarmes pour différents seuils de décision. La courbe d'EfficientNet-B0 se rapproche du coin supérieur gauche, indiquant des performances presque idéales. ResNet50 montre également une bonne courbe mais avec une zone sous la courbe légèrement inférieure.  

Le point optimal sur la courbe ROC (maximisant la différence entre le taux de vrais positifs et le taux de faux positifs) a été utilisé pour déterminer le seuil de décision optimal pour chaque modèle. Pour EfficientNet-B0, ce seuil est de 0,45, produisant une précision de 0,98 et un rappel de 0,85.  

#### 4.3.3 Implications Industrielles  

Dans le contexte industriel, le choix du seuil de décision représente un compromis crucial entre deux types d'erreurs coûteuses : les défauts non détectés (faux négatifs) qui peuvent entraîner des produits défectueux livrés aux clients, et les fausses alarmes (faux positifs) qui peuvent causer des arrêts de production inutiles. Les résultats suggèrent qu'EfficientNet-B0 offre le meilleur compromis avec un taux de fausses alarmes inférieur à 5% tout en détectant plus de 85% des anomalies.  

### 4.4 Résultats de Classification Multi-Classe  

#### 4.4.1 Performances par Classe  

La classification multi-classe évalue la capacité des modèles à identifier spécifiquement le type de défaut. EfficientNet-B0 obtient à nouveau les meilleures performances avec une AUC-ROC macro de 0,93 et une exactitude globale de 0,91. Cependant, l'analyse par classe révèle des différences significatives :  

- Les défauts structurels comme `missing_cable` et `cut_outer_insulation` sont bien reconnus (F1 > 0,85).  
- Les défauts subtils comme `poke_insulation` et les défauts complexes comme `combined` présentent des performances plus faibles (F1 < 0,70).  

Ces différences reflètent la difficulté intrinsèque de certaines catégories d'anomalies. Les défauts subtils peuvent être difficiles à distinguer des variations normales, tandis que les défauts combinés présentent une grande variabilité qui complique l'apprentissage, particulièrement avec seulement 5 époques d'entraînement.  

#### 4.4.2 Matrice de Confusion  

L'analyse de la matrice de confusion d'EfficientNet-B0 révèle des patterns d'erreur intéressants. La classe `normal` est parfaitement reconnue (aucun faux positif), ce qui est crucial pour minimiser les fausses alarmes. Les principales confusions se produisent entre des classes similaires comme `bent_wire` et `cable_swap`, ou entre différentes formes de coupures d'isolation.  

Ces confusions suggèrent que certaines caractéristiques visuelles sont partagées entre différents types de défauts, rendant leur distinction difficile. Cela pourrait être amélioré par un entraînement plus long ou par l'ajout de contraintes spécifiques dans la fonction de perte pour maximiser la séparabilité entre classes similaires.  

### 4.5 Résultats de Segmentation Pixel-Level  

#### 4.5.1 Métriques de Segmentation  

La segmentation pixel-level évalue la capacité des modèles à localiser précisément les anomalies dans les images. Les résultats montrent un pattern intéressant :  

- **Autoencodeur** obtient le meilleur PRO (Per-Region Overlap) de 0,894, démontrant une excellente capacité à localiser les anomalies grâce à son mécanisme d'erreur de reconstruction pixel-level.  
- **EfficientNet-B0** obtient le meilleur PRO parmi les modèles supervisés (0,315), mais cette performance reste modeste comparée à l'autoencodeur.  
- **CNN Custom** montre les performances les plus faibles en segmentation, reflétant peut-être une architecture insuffisamment profonde pour cette tâche complexe.  

#### 4.5.2 Analyse des Heatmaps  

L'analyse visuelle des heatmaps générées par les différents modèles révèle des différences qualitatives importantes. L'autoencodeur produit des heatmaps précises et bien localisées, avec des zones d'activation correspondant exactement aux régions défectueuses. EfficientNet-B0 produit des heatmaps plus diffuses mais généralement centrées sur les régions pertinentes. ResNet50 montre des activations plus étendues, parfois couvrant des régions normales adjacentes aux anomalies.  

Ces différences reflètent les mécanismes fondamentalement différents utilisés par les modèles pour la localisation. L'autoencodeur utilise directement l'erreur de reconstruction, qui est intrinsèquement locale. Les modèles supervisés utilisent généralement des techniques comme Grad-CAM qui propagent l'information de classification vers les couches convolutionnelles, un processus qui peut être moins précis spatialement.  

### 4.6 Analyse de la Convergence  

#### 4.6.1 Courbes d'Apprentissage  

Malgré la limitation à 5 époques, les courbes d'apprentissage montrent une convergence rapide pour tous les modèles. EfficientNet-B0 converge particulièrement rapidement, atteignant une perte de validation stable dès la troisième époque. ResNet50 montre une convergence plus lente mais régulière, tandis que le CNN Custom présente certaines oscillations suggérant un besoin potentiel d'un taux d'apprentissage plus faible.  

L'autoencodeur montre une convergence rapide de l'erreur de reconstruction sur les images normales, mais cette erreur reste élevée sur les images anormales même après 5 époques, ce qui est souhaitable pour la détection.  

#### 4.6.2 Impact du Nombre Limit d'Époques  

La contrainte de 5 époques a certainement influencé les performances des modèles. Les modèles pré-entraînés (ResNet50 et EfficientNet-B0) ont bénéficié de leur initialisation à partir de poids pré-entraînés, leur permettant d'atteindre de bonnes performances rapidement. Le CNN Custom, entraîné à partir de zéro, a probablement été le plus affecté par cette limitation.  

Cette contrainte reflète néanmoins une réalité industrielle où le temps d'entraînement peut être limité, particulièrement lors du déploiement initial ou des mises à jour fréquentes du modèle. La capacité d'EfficientNet-B0 à converger rapidement est donc un atout important pour les applications pratiques.  

### 4.7 Conclusion de la Section Résultats  

Les résultats expérimentaux présentés dans cette section démontrent que différentes architectures présentent des forces complémentaires sur les différentes tâches de détection d'anomalies. EfficientNet-B0 excelle en classification, particulièrement en classification binaire où il atteint des performances quasi-parfaites. L'autoencodeur, malgré ses faibles performances en classification, montre une capacité exceptionnelle en localisation pixel-level.  

Ces résultats suggèrent qu'une approche hybride, combinant les forces de différents modèles, pourrait offrir la solution optimale. La section suivante présente une analyse comparative approfondie et propose une méthodologie pour sélectionner le modèle le plus adapté aux contraintes industrielles spécifiques.  

---

## 5. ANALYSE COMPARATIVE ET SÉLECTION DU MODÈLE OPTIMAL  

### 5.1 Introduction à l'Analyse Comparative  

Cette section présente une analyse comparative systématique des quatre modèles évalués, visant à identifier le modèle optimal pour le déploiement industriel. L'analyse ne se limite pas à une simple comparaison des métriques individuelles, mais intègre une évaluation multidimensionnelle prenant en compte les performances sur les trois niveaux de tâches, l'efficacité computationnelle, et les considérations pratiques de déploiement.  

### 5.2 Score Composite pour l'Évaluation Multi-Niveaux  

#### 5.2.1 Définition du Score Composite  

Pour permettre une comparaison objective intégrant les trois niveaux de performance, nous avons défini un score composite pondéré selon l'importance industrielle de chaque métrique. Les pondérations ont été déterminées en collaboration avec des experts du domaine et reflètent les priorités typiques des applications industrielles :  

- **AUC-ROC binaire (30%)** : Métrique standard pour la détection d'anomalies, prioritaire car elle mesure la capacité fondamentale à distinguer le normal de l'anormal.  
- **F1-Score binaire (20%)** : Important pour l'équilibre entre précision et rappel, crucial pour minimiser à la fois les défauts non détectés et les fausses alarmes.  
- **Rappel binaire (15%)** : Critique industriellement car les défauts non détectés ont généralement un coût plus élevé que les fausses alarmes.  
- **PRO (25%)** : Métrique officielle MVTec AD pour la segmentation, importante pour la localisation précise et l'analyse des causes.  
- **AUC-ROC multi-classe (10%)** : Moins prioritaire que la détection binaire mais utile pour la classification fine des défauts.  

#### 5.2.2 Calcul des Scores  

Le score composite \( S \) pour chaque modèle est calculé comme suit :  
\[ S = 0.30 \times \text{AUC}_{\text{binaire}} + 0.20 \times \text{F1}_{\text{binaire}} + 0.15 \times \text{Rappel}_{\text{binaire}} + 0.25 \times \text{PRO} + 0.10 \times \text{AUC}_{\text{multi}} \]  

Les scores obtenus sont :  
- **EfficientNet-B0** : 0,785  
- **ResNet50** : 0,678  
- **CNN Custom** : 0,534  
- **Autoencodeur** : 0,497  

#### 5.2.3 Interprétation des Scores  

Le score d'EfficientNet-B0 (0,785) le place nettement en tête, reflétant ses excellentes performances sur la plupart des métriques, particulièrement l'AUC-ROC binaire et le F1-Score. ResNet50 obtient un bon score (0,678) mais est pénalisé par sa faible précision et son PRO modéré. Les scores plus faibles du CNN Custom et de l'autoencodeur reflètent leurs limitations respectives : architecture insuffisamment profonde pour le premier, approche non supervisée peu adaptée à la classification pour le second.  

### 5.3 Analyse des Compromis Performance/Coût  

#### 5.3.1 Efficacité Computationnelle  

L'efficacité computationnelle est un facteur crucial pour le déploiement industriel, particulièrement pour les applications en temps réel ou sur des matériels embarqués. L'analyse comparative révèle des différences significatives :  

- **EfficientNet-B0** offre le meilleur rapport performance/complexité avec seulement 5,3 millions de paramètres et 0,39 GFLOPs, tout en obtenant les meilleures performances.  
- **ResNet50** est nettement plus lourd (25,6 millions de paramètres, 4,1 GFLOPs) pour des performances inférieures.  
- **CNN Custom** est relativement léger (15,2 millions de paramètres) mais ses performances limitées réduisent son intérêt pratique.  
- **Autoencodeur** présente une complexité modérée (18,7 millions de paramètres) mais des temps d'inférence légèrement plus longs dus à la nécessité de reconstruire l'image complète.  

#### 5.3.2 Vitesse d'Inférence  

La vitesse d'inférence, mesurée en images par seconde (FPS), est critique pour les applications de production en ligne :  

- **CNN Custom** : 125 FPS (le plus rapide)  
- **EfficientNet-B0** : 83 FPS (bon compromis)  
- **Autoencodeur** : 67 FPS  
- **ResNet50** : 55 FPS (le plus lent)  

Tous les modèles dépassent largement les 30 FPS généralement considérés comme suffisants pour le traitement en temps réel, mais EfficientNet-B0 offre le meilleur compromis entre vitesse et précision.  

### 5.4 Analyse par Type de Défaut  

#### 5.4.1 Performances Différenciées  

L'analyse détaillée des performances par type de défaut révèle que certains modèles sont particulièrement adaptés à certaines catégories d'anomalies :  

- **EfficientNet-B0** excelle sur les défauts structurels bien définis (`missing_cable`, `cut_outer_insulation`) avec des F1-Scores > 0,85.  
- **ResNet50** montre un bon rappel sur les défauts subtils (`poke_insulation`) grâce à sa profondeur qui lui permet de capturer des caractéristiques fines.  
- **Autoencodeur** localise particulièrement bien les défauts de texture et les petites anomalies grâce à son mécanisme d'erreur de reconstruction pixel-level.  

#### 5.4.2 Implications pour une Approche Hybride  

Ces différences suggèrent qu'aucun modèle n'est optimal pour tous les types de défauts. Une approche hybride, utilisant différents modèles pour différentes catégories d'anomalies, pourrait potentiellement surpasser tout modèle individuel. Par exemple, on pourrait utiliser EfficientNet-B0 pour la détection initiale et la classification grossière, puis un autoencodeur pour la localisation précise ou pour les cas ambigus.  

### 5.5 Impact de la Limitation à 5 Époques  

#### 5.5.1 Conséquences sur les Performances  

La limitation à 5 époques d'entraînement a certainement affecté les performances des modèles, particulièrement :  

- **CNN Custom** : Probablement sous-entraîné, nécessiterait plus d'époques pour converger pleinement.  
- **ResNet50** : Le fine-tuning des couches profondes nécessite généralement plus d'époques pour être optimal.  
- **EfficientNet-B0** : A probablement le moins souffert de cette limitation grâce à son architecture optimisée et à son pré-entraînement efficace.  
- **Autoencodeur** : L'apprentissage de la "normalité" peut nécessiter de nombreuses époques, particulièrement pour des anomalies subtiles.  

#### 5.5.2 Considérations pour le Déploiement  

Dans un contexte de déploiement industriel, la capacité à atteindre de bonnes performances avec peu d'époques est un atout important, car elle permet :  
- Des cycles de développement plus rapides  
- Une adaptation rapide aux changements de production  
- Des économies significatives en coûts computationnels  
- Une intégration plus facile dans des pipelines CI/CD  

EfficientNet-B0 démontre particulièrement bien cette capacité, atteignant d'excellentes performances en seulement 5 époques.  

### 5.6 Sélection du Modèle Optimal  

#### 5.6.1 Critères de Sélection  

La sélection du modèle optimal pour le déploiement industriel repose sur plusieurs critères :  

1. **Performances globales** : Mesurées par le score composite et les métriques critiques (rappel, précision).  
2. **Efficacité computationnelle** : Nombre de paramètres, GFLOPs, mémoire requise.  
3. **Vitesse d'inférence** : Capacité à traiter les images en temps réel.  
4. **Robustesse** : Stabilité des performances sur différentes catégories de défauts.  
5. **Facilité de déploiement** : Compatibilité avec les infrastructures existantes, besoins en matériel.  

#### 5.6.2 Modèle Recommandé : EfficientNet-B0  

Sur la base de l'analyse comparative, **EfficientNet-B0** est recommandé comme modèle optimal pour le déploiement industriel. Cette recommandation s'appuie sur :  

- **Performances exceptionnelles** : Meilleur score composite (0,785), AUC-ROC binaire quasi-parfaite (0,997), excellent équilibre précision/rappel (F1=0,91).  
- **Efficacité computationnelle** : 5,3 millions de paramètres seulement (5× moins que ResNet50), 0,39 GFLOPs (10× moins que ResNet50).  
- **Vitesse d'inférence** : 83 FPS, compatible avec les exigences du temps réel industriel.  
- **Convergence rapide** : Bonnes performances atteintes en seulement 5 époques.  
- **Large adoption** : Architecture bien documentée et largement utilisée, facilitant la maintenance et l'évolution.  

#### 5.6.3 Approches Complémentaires  

Bien qu'EfficientNet-B0 soit recommandé comme modèle principal, certaines approches complémentaires méritent d'être considérées :  

- **Autoencodeur pour la localisation** : Pour les applications où la localisation précise est critique, l'autoencodeur pourrait être utilisé en complément pour générer des heatmaps précises.  
- **Ensemble de modèles** : Pour des applications à très haut niveau de sécurité, un ensemble combinant EfficientNet-B0 et ResNet50 pourrait offrir une robustesse accrue.  
- **Modèles spécialisés** : Pour des types de défauts spécifiques particulièrement difficiles, des modèles spécialisés pourraient être développés.  

### 5.7 Conclusion de la Section Analyse Comparative  

L'analyse comparative approfondie présentée dans cette section démontre clairement la supériorité d'EfficientNet-B0 pour la détection d'anomalies sur des câbles industriels. Son excellent compromis entre performances, efficacité computationnelle et vitesse d'inférence en fait le choix optimal pour un déploiement industriel. Cependant, les forces complémentaires des autres modèles, particulièrement de l'autoencodeur pour la localisation, suggèrent que des architectures hybrides pourraient offrir des avantages supplémentaires pour des applications spécifiques.  

---

## 6. CONCLUSION GÉNÉRALE ET PERSPECTIVES  

### 6.1 Synthèse des Principaux Résultats  

Ce projet a démontré la faisabilité et l'efficacité de différentes approches de deep learning pour la détection d'anomalies sur des câbles industriels. Les principaux résultats peuvent être synthétisés ainsi :  

1. **Performances de détection** : EfficientNet-B0 atteint une AUC-ROC de 0,997 pour la classification binaire, démontrant une capacité quasi-parfaite à distinguer les câbles normaux des câbles défectueux.  
2. **Classification multi-classe** : Avec une exactitude de 0,91 et un F1-Score macro de 0,62, les modèles montrent une capacité prometteuse à identifier spécifiquement le type de défaut.  
3. **Localisation** : L'autoencodeur obtient un PRO de 0,894, montrant l'efficacité des approches non supervisées pour la localisation pixel-level.  
4. **Efficacité computationnelle** : EfficientNet-B0 offre le meilleur rapport performance/coût avec seulement 5,3 millions de paramètres et 0,39 GFLOPs.  

### 6.2 Contributions du Projet  

Ce projet apporte plusieurs contributions significatives :  

1. **Évaluation comparative rigoureuse** : Comparaison systématique de quatre approches différentes sur un benchmark industriel standard.  
2. **Approche multi-niveaux** : Évaluation simultanée sur trois niveaux de complexité (détection, classification, localisation).  
3. **Adaptation aux contraintes industrielles** : Prise en compte de facteurs pratiques comme la vitesse d'inférence, l'efficacité computationnelle, et la convergence rapide.  
4. **Méthodologie reproductible** : Protocole d'évaluation complet et reproductible, avec un score composite pondéré selon les priorités industrielles.  

### 6.3 Limitations et Défis Identifiés  

Plusieurs limitations ont été identifiées durant ce projet :  

1. **Données limitées** : Malgré l'augmentation, certaines classes d'anomalies restent sous-représentées.  
2. **Complexité des défauts combinés** : Les défauts combinant plusieurs types d'anomalies présentent les performances les plus faibles.  
3. **Localisation des modèles supervisés** : Les approches supervisées obtiennent des performances de localisation inférieures aux approches non supervisées.  
4. **Généralisation** : L'évaluation sur une seule catégorie de MVTec AD limite la validation de la généralisation à d'autres types d'objets industriels.  

### 6.4 Perspectives de Recherche et Développement  

#### 6.4.1 Améliorations à Court Terme  

1. **Architectures hybrides** : Combinaison des forces d'EfficientNet-B0 (classification) et de l'autoencodeur (localisation).  
2. **Apprentissage semi-supervisé** : Exploitation des données non étiquetées pour améliorer la robustesse.  
3. **Augmentation de données avancée** : Techniques comme MixUp, CutMix, ou l'utilisation de GANs pour générer des anomalies réalistes.  

#### 6.4.2 Développements à Moyen Terme  

1. **Extension à d'autres catégories** : Application de la méthodologie aux 14 autres catégories de MVTec AD.  
2. **Modèles spécifiques par type de défaut** : Développement de modèles spécialisés pour les catégories d'anomalies les plus difficiles.  
3. **Systèmes adaptatifs** : Capacité d'adaptation continue aux changements dans le processus de production.  

#### 6.4.3 Innovations à Long Terme  

1. **Intégration dans les systèmes de production** : Connexion directe aux systèmes MES (Manufacturing Execution Systems).  
2. **Expliquabilité avancée** : Techniques pour expliquer les décisions du modèle aux opérateurs humains.  
3. **Détection proactive** : Prédiction des anomalies avant qu'elles ne se produisent, basée sur des patterns subtils.  

### 6.5 Recommandations pour le Déploiement Industriel  

Pour un déploiement réussi en environnement industriel, nous recommandons :  

1. **Phase pilote** : Déploiement initial sur une ligne de production limitée pour validation en conditions réelles.  
2. **Monitoring continu** : Mise en place de métriques de suivi des performances en production.  
3. **Processus d'amélioration continue** : Mécanisme pour collecter les cas difficiles et ré-entraîner périodiquement le modèle.  
4. **Formation des opérateurs** : Intégration du système dans le workflow des opérateurs avec une interface adaptée.  

### 6.6 Impact Industriel Potentiel  

L'implémentation d'un système basé sur EfficientNet-B0 pourrait apporter des bénéfices significatifs :  

- **Réduction des coûts** : Diminution jusqu'à 70% des coûts d'inspection manuelle.  
- **Amélioration de la qualité** : Détection plus précoce et plus fiable des défauts.  
- **Augmentation de la productivité** : Traitement plus rapide et possibilité d'inspecter 100% de la production.  
- **Traçabilité améliorée** : Enregistrement automatique et analyse statistique des défauts détectés.  

### 6.7 Conclusion Finale  

Ce projet a démontré que les techniques de deep learning, particulièrement lorsqu'elles sont judicieusement sélectionnées et adaptées, offrent des solutions performantes et pratiques pour la détection d'anomalies industrielles. EfficientNet-B0 émerge comme le modèle optimal, offrant un excellent compromis entre performances, efficacité et facilité de déploiement.  

Les résultats obtenus, bien que prometteurs, ouvrent également la voie à de nombreuses améliorations et extensions futures. La méthodologie développée dans ce projet constitue une base solide pour des applications industrielles concrètes et pour des recherches futures dans le domaine de la détection automatique d'anomalies.  

---

*Fin du rapport technique*