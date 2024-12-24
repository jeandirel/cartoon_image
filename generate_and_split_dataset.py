import os
import cv2
import shutil
import random
from sklearn.model_selection import train_test_split

def cartoonize_image(img):
    """
    Transforme une image en style cartoonisé.
    
    Args:
        img (numpy.ndarray): Image d'entrée sous forme de tableau NumPy.
        
    Returns:
        numpy.ndarray: Image transformée avec un effet cartoon.
    """
    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Détecter les contours avec l'algorithme de Canny
    edges = cv2.Canny(gray, 100, 200)
    # Appliquer un filtre bilatéral pour réduire le bruit tout en préservant les bords
    filtered = cv2.bilateralFilter(img, 9, 75, 75)
    # Combiner l'image filtrée avec les contours pour créer l'effet cartoon
    cartoon = cv2.bitwise_and(filtered, filtered, mask=edges)
    return cartoon

def create_paired_dataset(input_dir, paired_output_dir, num_images=None):
    """
    Génère un dataset apparié contenant des images réalistes et leurs versions cartoonisées.
    
    Args:
        input_dir (str): Chemin du dossier contenant les images d'entrée classées par catégories.
        paired_output_dir (str): Chemin du dossier où sauvegarder le dataset apparié.
        num_images (int, optional): Nombre d'images à utiliser. Si None, toutes les images seront utilisées.
    """
    # Création des répertoires pour les images réalistes et cartoonisées
    real_dir = os.path.join(paired_output_dir, "real")
    cartoon_dir = os.path.join(paired_output_dir, "cartoon")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(cartoon_dir, exist_ok=True)
    
    all_images = []
    # Collecter tous les chemins d'images classés par catégories
    for category in os.listdir(input_dir):
        category_path = os.path.join(input_dir, category)
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            all_images.append((category, img_path))
    
    # Si un nombre limite d'images est spécifié, en sélectionner un échantillon aléatoire
    if num_images:
        all_images = random.sample(all_images, min(num_images, len(all_images)))
    
    # Générer et sauvegarder les paires d'images réalistes et cartoonisées
    for category, img_path in all_images:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Erreur : Impossible de lire {img_path}")
            continue
        cartoon_img = cartoonize_image(img)
        
        # Sauvegarder l'image réaliste
        category_real_dir = os.path.join(real_dir, category)
        os.makedirs(category_real_dir, exist_ok=True)
        real_output_path = os.path.join(category_real_dir, os.path.basename(img_path))
        cv2.imwrite(real_output_path, img)
        
        # Sauvegarder l'image cartoonisée
        category_cartoon_dir = os.path.join(cartoon_dir, category)
        os.makedirs(category_cartoon_dir, exist_ok=True)
        cartoon_output_path = os.path.join(category_cartoon_dir, os.path.basename(img_path))
        cv2.imwrite(cartoon_output_path, cartoon_img)

def split_paired_dataset(paired_dir, output_dir, test_size=0.2):
    """
    Divise un dataset apparié en ensembles d'entraînement (train) et de test, 
    en préservant la correspondance entre les images réalistes et cartoonisées.
    
    Args:
        paired_dir (str): Chemin du dataset apparié.
        output_dir (str): Chemin du dossier où sauvegarder les ensembles train/test.
        test_size (float): Proportion du dataset à inclure dans l'ensemble de test.
    """
    # Définition des répertoires pour les images réalistes et cartoonisées
    real_dir = os.path.join(paired_dir, "real")
    cartoon_dir = os.path.join(paired_dir, "cartoon")
    
    # Création des répertoires pour train et test
    train_real_dir = os.path.join(output_dir, "train", "real")
    train_cartoon_dir = os.path.join(output_dir, "train", "cartoon")
    test_real_dir = os.path.join(output_dir, "test", "real")
    test_cartoon_dir = os.path.join(output_dir, "test", "cartoon")
    
    for d in [train_real_dir, train_cartoon_dir, test_real_dir, test_cartoon_dir]:
        os.makedirs(d, exist_ok=True)
    
    # Collecter toutes les images et leurs catégories
    all_images = []
    for category in os.listdir(real_dir):
        category_real_path = os.path.join(real_dir, category)
        for img_name in os.listdir(category_real_path):
            all_images.append((category, img_name))
    
    # Diviser les images en ensembles d'entraînement et de test
    train_images, test_images = train_test_split(all_images, test_size=test_size, random_state=42)
    
    # Copier les fichiers dans les répertoires appropriés
    for subset, subset_name in [(train_images, "train"), (test_images, "test")]:
        for category, img_name in subset:
            # Copier l'image réaliste
            src_real = os.path.join(real_dir, category, img_name)
            dest_real = os.path.join(output_dir, subset_name, "real", category, img_name)
            os.makedirs(os.path.dirname(dest_real), exist_ok=True)
            shutil.copy(src_real, dest_real)
            
            # Copier l'image cartoonisée
            src_cartoon = os.path.join(cartoon_dir, category, img_name)
            dest_cartoon = os.path.join(output_dir, subset_name, "cartoon", category, img_name)
            os.makedirs(os.path.dirname(dest_cartoon), exist_ok=True)
            shutil.copy(src_cartoon, dest_cartoon)

# Étape 1 : Générer les paires d'images réalistes et cartoonisées
create_paired_dataset("processed_images/", "paired_dataset/", num_images=1000)

# Étape 2 : Diviser le dataset apparié en ensembles d'entraînement et de test
split_paired_dataset("paired_dataset/", "final_dataset/", test_size=0.2)
