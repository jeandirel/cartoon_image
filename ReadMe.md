Voici le fichier README.md amélioré au format texte :



# Cartoonization GAN Project

Ce projet met en œuvre un réseau génératif adversarial (GAN) basé sur l'architecture Pix2Pix pour transformer des images réalistes en images cartoonisées. Il combine des techniques avancées de vision par ordinateur et de deep learning pour réaliser cette transformation.

Le modèle GAN utilise des paires d'images correspondantes (réalistes et cartoon) pour l'entraînement et génère des images cartoonisées à partir d'images réelles. Des métriques telles que SSIM (Structural Similarity Index) et PSNR (Peak SignaltoNoise Ratio) sont utilisées pour évaluer la qualité des images générées.

## Fonctionnalités
 Transformation d'images réalistes en style cartoon : Utilise un modèle de GAN pour transformer des images réalistes en versions cartoonisées.
 Entraînement du modèle avec des données appariées : Le modèle est entraîné sur un ensemble d'images réelles et cartoonisées correspondantes.
 Validation sur un dataset de test : Après l'entraînement, le modèle est validé à l'aide de métriques standard comme SSIM et PSNR pour évaluer la qualité des images générées.
 Test interactif sur de nouvelles images : Permet à l'utilisateur de tester le modèle sur de nouvelles images et de visualiser les résultats.
 Métriques de qualité d'image : Calcul des scores SSIM et PSNR pour chaque image générée, avec sauvegarde des résultats dans un fichier CSV pour une analyse ultérieure.

## Structure du projet

.
├── final_dataset/          # Dataset d'entraînement et de test
│   ├── train/              # Données d'entraînement
│   │   ├── real/          # Images réalistes
│   │   └── cartoon/       # Images cartoonisées
│   ├── test/               # Données de test
│       ├── real/          # Images réalistes
│       └── cartoon/       # Images cartoonisées
├── models/                 # Modèles sauvegardés
│   └── pix2pix_generator_final.h5  # Modèle entraîné final
├── results/                # Répertoire pour les résultats des tests
│   └── metrics.csv         # Fichier CSV avec les résultats de SSIM et PSNR
├── train_pix2pix.py        # Script principal d'entraînement
├── interactive_test.py     # Script pour tester interactivement sur de nouvelles images
├── validation.py           # Script pour valider le modèle sur un dataset de test
├── README.md               # Documentation du projet


## Installation

### Prérequis
 Python 3.8 ou version ultérieure
 TensorFlow 2.x : Pour l'entraînement et l'inférence du modèle GAN.
 scikitlearn, opencvpython, matplotlib : Bibliothèques pour la manipulation des images, l'évaluation des métriques et la visualisation des résultats.

### Installation des dépendances
Exécutez la commande suivante pour installer toutes les dépendances nécessaires :
bash
pip install tensorflow scikitlearn opencvpython matplotlib


## Entraînement

1. Préparez vos données : Placez vos données d'entraînement et de test dans le répertoire `final_dataset/` en suivant la structure fournie. Assurezvous d'avoir un ensemble d'images réelles et d'images cartoonisées correspondantes pour l'entraînement.
2. Exécutez le script d'entraînement :
   bash
   python train_pix2pix.py
   
   Ce script entraînera le modèle GAN sur les données d'entraînement et sauvegardera le modèle dans le répertoire `models/` une fois l'entraînement terminé.

## Validation

### Validation sur le dataset de test
Pour valider le modèle sur le dataset de test, exécutez :
bash
python validation.py

Ce script génère des images cartoonisées à partir des images réalistes du dataset de test et évalue la qualité des images générées à l'aide des métriques SSIM et PSNR. Les résultats sont sauvegardés dans un fichier CSV (`metrics.csv`) pour une analyse ultérieure.

### Test sur de nouvelles images
Pour tester le modèle sur de nouvelles images, utilisez le script interactif :
bash
python interactive_test.py

Une interface simple vous permettra de sélectionner une image, de la transformer en version cartoon et de visualiser les résultats immédiatement.

## Structure des fichiers

 train_pix2pix.py : Script principal pour entraîner le modèle Pix2Pix. Ce script charge les données, construit le générateur et le discriminateur, et effectue l'entraînement du modèle.
 validation.py : Effectue la validation du modèle sur le dataset de test en calculant les métriques SSIM et PSNR. Il génère également un fichier CSV avec les résultats.
 interactive_test.py : Permet de tester le modèle interactivement sur de nouvelles images. Affiche les résultats à côté des images d'entrée.
 models/pix2pix_generator_final.h5 : Le modèle entraîné final qui peut être utilisé pour générer des images cartoonisées à partir d'images réelles.
 results/metrics.csv : Fichier CSV contenant les résultats des métriques SSIM et PSNR pour chaque image validée.

## Contact

Pour toute question ou assistance, veuillez contacter :
 Jean Direl, Xavier, Hackim, Cheick
 Emails : 
   jeandirel.nzekabeyene@aivancity.education
   hakim.djomo@aivancity.education
   cheickadamyakine.bamba@aivancity.education
   xavier.ondo@aivancity.education



Note : Ce projet est destiné à des fins académiques et non à des usages commerciaux.



### Améliorations et suggestions :
 Le projet peut être étendu en ajoutant des options pour l'entraînement avec des datasets plus grands ou des ajustements d'architecture pour de meilleures performances.
 Des tests supplémentaires peuvent être réalisés en utilisant des images de différents styles pour vérifier la robustesse du modèle.


