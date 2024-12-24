import os
import shutil
import random
from sklearn.model_selection import train_test_split

def split_dataset(input_dir, output_dir, max_images=1000, test_size=0.2):
    """
    Divise les données en ensembles train et test, avec un nombre d'images maximum spécifié.
    
    Cette fonction parcourt le répertoire d'entrée contenant les images organisées par catégories (dossiers),
    limite le nombre total d'images à celui spécifié (max_images), puis divise ces images en ensembles d'entraînement
    et de test selon la proportion spécifiée (test_size). Les images sont ensuite copiées dans les répertoires
    respectifs 'train' et 'test' dans le répertoire de sortie.

    :param input_dir: Répertoire contenant les images classées par catégorie (dossiers).
    :param output_dir: Répertoire où les ensembles 'train' et 'test' seront enregistrés.
    :param max_images: Nombre maximum d'images à sélectionner pour la division (par défaut 1000).
    :param test_size: Proportion des données utilisées pour les tests (par défaut 0.2, soit 20% pour le test).
    """
    # Vérification de l'existence du répertoire de sortie, sinon création
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Liste pour stocker les chemins des images
    all_images = []
    
    # Parcourir les sous-dossiers du répertoire d'entrée (chaque sous-dossier correspond à une catégorie)
    for category in os.listdir(input_dir):
        category_path = os.path.join(input_dir, category)
        # Vérifier si c'est un répertoire (catégorie d'images)
        if os.path.isdir(category_path):
            # Récupérer les chemins de toutes les images dans le sous-dossier
            images = [os.path.join(category, img) for img in os.listdir(category_path)]
            all_images.extend(images)
    
    # Limiter le nombre d'images à celui spécifié par max_images
    selected_images = random.sample(all_images, min(max_images, len(all_images)))
    
    # Diviser les images en deux ensembles (train et test) selon la proportion test_size
    train_images, test_images = train_test_split(selected_images, test_size=test_size, random_state=42)
    
    # Copier les images sélectionnées dans les répertoires de train et test
    for subset, subset_name in [(train_images, 'train'), (test_images, 'test')]:
        for img_path in subset:
            # Extraire la catégorie et le nom de l'image à partir du chemin
            category = os.path.dirname(img_path)  # Catégorie (par exemple 'cats')
            img_name = os.path.basename(img_path)  # Nom de l'image (par exemple 'image1.png')
            # Définir le chemin source et le répertoire de destination
            src_path = os.path.join(input_dir, img_path)
            dest_dir = os.path.join(output_dir, subset_name, category)
            # Créer le répertoire de destination s'il n'existe pas
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            # Copier l'image dans le répertoire de destination
            shutil.copy(src_path, os.path.join(dest_dir, img_name))
            # Afficher un message indiquant l'opération
            print(f"{img_name} -> {subset_name}/{category}")

# Exemple d'utilisation de la fonction
split_dataset("processed_images/", "dataset/", max_images=1000, test_size=0.2)
