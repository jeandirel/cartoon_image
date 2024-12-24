import os
import tensorflow as tf
from tensorflow.keras.models import load_model

# Fonction de chargement et de prétraitement des images
# Cette fonction lit une image depuis le chemin spécifié, la décode en tant qu'image PNG avec 3 canaux (RGB),
# la redimensionne à une taille de 256x256 pixels, puis normalise les valeurs des pixels entre -1 et 1.
def load_image(image_path):
    # Lire le fichier image à partir du chemin spécifié
    image = tf.io.read_file(image_path)
    # Décoder le fichier PNG en image avec 3 canaux (RGB)
    image = tf.image.decode_png(image, channels=3)
    # Redimensionner l'image à 256x256 pixels
    image = tf.image.resize(image, [256, 256])
    # Normaliser l'image (les pixels sont initialement entre 0 et 255, donc ici ils sont normalisés entre -1 et 1)
    image = (image / 127.5) - 1
    return image

# Chargement du modèle pré-entrainé
# Vérifie si le fichier du modèle existe, et si c'est le cas, il charge le modèle à partir du fichier '.h5'.
model_path = "models/pix2pix_generator_epoch_100.h5"
# Vérification de l'existence du fichier du modèle
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Le modèle '{model_path}' est introuvable.")
# Chargement du modèle avec Keras
model = load_model(model_path)
print("Modèle chargé avec succès.")

# Fonction de chargement du dataset de test
# Cette fonction charge les images réelles et cartonnées à partir de leurs répertoires respectifs,
# les prétraite et les combine en un dataset de test en batch de 16 images.
def load_dataset(real_path, cartoon_path):
    # Charger les images réelles depuis le répertoire 'real_path' en les prétraitant
    real_images = tf.data.Dataset.list_files(os.path.join(real_path, '*.png'), shuffle=False).map(load_image)
    # Charger les images cartonnées depuis le répertoire 'cartoon_path' en les prétraitant
    cartoon_images = tf.data.Dataset.list_files(os.path.join(cartoon_path, '*.png'), shuffle=False).map(load_image)
    # Combiner les datasets des images réelles et cartonnées et les grouper en batchs de 16
    return tf.data.Dataset.zip((real_images, cartoon_images)).batch(16)

# Charger le dataset de test à partir des répertoires spécifiés
test_dataset = load_dataset('final_dataset/test/real', 'final_dataset/test/cartoon')

# Fonction de validation sur le dataset de test
# Cette fonction passe en revue chaque batch d'images réelles, génère les images cartonnées prédites
# par le modèle, puis imprime un message de validation pour chaque batch.
def validate_on_test_set(model, dataset):
    # Itérer sur chaque batch du dataset de test
    for real_images, cartoon_images in dataset:
        # Appliquer le modèle sur les images réelles pour générer les sorties
        outputs = model(real_images)
        print("Validation sur un batch terminé.")

# Appeler la fonction de validation sur le dataset de test
validate_on_test_set(model, test_dataset)

# Fonction de validation sur de nouvelles images
# Cette fonction permet de valider le modèle sur des images spécifiées dans la liste 'image_paths'.
# Si une image n'existe pas, elle affiche une erreur. Si l'image est valide, elle est prétraitée et passée dans le modèle.
def validate_on_new_images(model, image_paths):
    for img_path in image_paths:
        # Vérifier si le chemin de l'image existe
        if not os.path.exists(img_path):
            print(f"Image '{img_path}' introuvable.")
            continue
        try:
            # Charger et prétraiter l'image
            image = load_image(img_path)
            # Ajouter une dimension pour simuler un batch de taille 1 (necessaire pour l'entrée du modèle)
            image = tf.expand_dims(image, axis=0)
            # Appliquer le modèle sur l'image prétraitée
            output = model(image)
            print(f"Image '{img_path}' traitée avec succès.")
        except Exception as e:
            # Gérer les erreurs et afficher un message approprié
            print(f"Erreur pour '{img_path}': {e}")

# Liste des images à valider
new_images = ["Trump1.jpg", "1.jpg"]
# Appeler la fonction de validation sur les nouvelles images
validate_on_new_images(model, new_images)
