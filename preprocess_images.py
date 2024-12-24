import os
import cv2
import numpy as np

def preprocess_images(input_dir, output_dir, image_size=(256, 256)):
    """
    Redimensionne et normalise toutes les images dans un répertoire donné.
    
    :param input_dir: Répertoire source contenant les images brutes, organisées par catégories.
    :param output_dir: Répertoire cible pour sauvegarder les images traitées, dans les mêmes catégories.
    :param image_size: Tuple (largeur, hauteur) pour redimensionner les images.
    """
    # Vérifie si le répertoire de sortie existe, sinon le crée
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Parcourt chaque catégorie dans le répertoire source
    for category in os.listdir(input_dir):
        category_path = os.path.join(input_dir, category)
        output_category_path = os.path.join(output_dir, category)
        
        # Vérifie si le chemin de la catégorie est un dossier
        if not os.path.isdir(category_path):
            print(f"Info : {category_path} n'est pas un dossier. Ignoré.")
            continue

        # Crée le répertoire cible correspondant pour cette catégorie
        if not os.path.exists(output_category_path):
            os.makedirs(output_category_path)
        
        # Parcourt toutes les images dans la catégorie
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            try:
                # Lire l'image depuis le chemin spécifié
                img = cv2.imread(img_path)
                
                # Vérifie si l'image a été correctement lue
                if img is None:
                    print(f"Erreur : Impossible de lire {img_path}. Fichier ignoré.")
                    continue
                
                # Redimensionne l'image à la taille spécifiée
                resized_img = cv2.resize(img, image_size)
                
                # Normalise les pixels pour qu'ils soient dans la plage [0, 1]
                normalized_img = resized_img / 255.0
                
                # Reconvertit les pixels normalisés en entier pour stockage
                output_img = (normalized_img * 255).astype(np.uint8)
                
                # Enregistre l'image traitée dans le dossier cible
                cv2.imwrite(os.path.join(output_category_path, img_name), output_img)
            except Exception as e:
                # Capture et affiche toute erreur pendant le traitement
                print(f"Erreur avec le fichier {img_name} : {e}")

# Exemple d'utilisation
if __name__ == "__main__":
    preprocess_images(
        input_dir="raw_images/",          # Répertoire contenant les images d'entrée
        output_dir="processed_images/"    # Répertoire pour sauvegarder les images traitées
    )
