import tensorflow as tf
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim  # Importation pour calculer SSIM
from skimage.metrics import peak_signal_noise_ratio as psnr  # Importation pour calculer PSNR
import matplotlib.pyplot as plt  # Pour afficher les images
import csv  # Pour enregistrer les résultats dans un fichier CSV

# Fonction pour charger et prétraiter une image
def load_image(image_path):
    # Lire l'image depuis le fichier
    image = tf.io.read_file(image_path)
    # Décoder l'image PNG avec 3 canaux (RVB)
    image = tf.image.decode_png(image, channels=3)
    # Redimensionner l'image à 256x256 pixels
    image = tf.image.resize(image, [256, 256])
    # Normaliser les valeurs des pixels entre -1 et 1
    image = (image / 127.5) - 1
    return image

# Sauvegarder les métriques SSIM et PSNR dans un fichier CSV
def save_metrics_to_csv(image_name, ssim_score, psnr_score, output_file="results/metrics.csv"):
    header = ["Image Name", "SSIM", "PSNR"]
    file_exists = os.path.exists(output_file)  # Vérifie si le fichier existe déjà
    # Ouvrir le fichier en mode append pour ajouter des lignes
    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Si le fichier n'existe pas, ajouter un en-tête
        if not file_exists:
            writer.writerow(header)
        # Ajouter une nouvelle ligne avec le nom de l'image et les scores SSIM et PSNR
        writer.writerow([image_name, ssim_score, psnr_score])

# Sauvegarder une image générée dans le dossier spécifié
def save_generated_image(image, save_path):
    tf.keras.utils.save_img(save_path, image)  # Sauvegarde de l'image avec Keras
    print(f"Image cartoonisée sauvegardée dans {save_path}.")  # Message de confirmation

# Valider le modèle sur un dataset de test et calculer les métriques
def validate_on_test_set(model, test_dataset):
    total_ssim = 0  # Variable pour stocker la somme des scores SSIM
    total_psnr = 0  # Variable pour stocker la somme des scores PSNR
    num_images = 0  # Compteur d'images traitées

    # Parcourir le dataset de test
    for real_image, cartoon_image in test_dataset:
        # Générer l'image cartoonisée avec le modèle
        generated_image = model(real_image, training=False)

        # Convertir les images en format uint8 pour le calcul des métriques
        real_image_np = ((real_image[0].numpy() + 1) * 127.5).astype('uint8')
        cartoon_image_np = ((cartoon_image[0].numpy() + 1) * 127.5).astype('uint8')
        generated_image_np = ((generated_image[0].numpy() + 1) * 127.5).astype('uint8')

        # Vérifier si l'image est assez grande pour calculer SSIM
        if cartoon_image_np.shape[0] > 6 and cartoon_image_np.shape[1] > 6:
            # Calculer les scores SSIM et PSNR
            ssim_score = ssim(cartoon_image_np, generated_image_np, multichannel=True, win_size=3)
            psnr_score = psnr(cartoon_image_np, generated_image_np)
            total_ssim += ssim_score  # Ajouter le score SSIM à la somme
            total_psnr += psnr_score  # Ajouter le score PSNR à la somme
            num_images += 1  # Augmenter le compteur d'images
        else:
            print(f"Image {num_images + 1} ignorée (taille trop petite pour SSIM).")

    # Calculer la moyenne des scores SSIM et PSNR
    if num_images > 0:
        avg_ssim = total_ssim / num_images
        avg_psnr = total_psnr / num_images
        print("\n--- Résultats de la validation sur le dataset de test ---")
        print(f"Moyenne SSIM : {avg_ssim:.4f}")
        print(f"Moyenne PSNR : {avg_psnr:.4f}")
    else:
        print("Aucune image valide pour calculer les métriques.")

# Valider sur de nouvelles images
def validate_on_new_images(model, image_paths, save_dir="generated_new_images"):
    os.makedirs(save_dir, exist_ok=True)  # Créer le dossier de sauvegarde si nécessaire
    for image_path in image_paths:
        # Charger l'image
        img = load_image(image_path)
        img = tf.expand_dims(img, axis=0)  # Ajouter une dimension de batch

        # Générer l'image cartoonisée
        generated_img = model(img, training=False)
        generated_img = (generated_img[0].numpy() + 1) * 127.5  # Re-normalisation entre 0 et 255
        generated_img = generated_img.astype('uint8')  # Convertir en format d'image

        # Sauvegarder et afficher les images générées
        save_path = os.path.join(save_dir, os.path.basename(image_path))
        save_generated_image(generated_img, save_path)

        # Afficher les résultats
        original_img = plt.imread(image_path)  # Charger l'image originale pour affichage
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.title("Image Réaliste")
        plt.imshow(original_img)  # Afficher l'image originale
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.title("Image Cartoonisée")
        plt.imshow(generated_img)  # Afficher l'image générée
        plt.axis('off')
        plt.show()

# Charger le modèle et effectuer les validations
if __name__ == "__main__":
    model_path = "models/pix2pix_generator_final.h5"  # Chemin vers le modèle pré-entraîné
    real_path = "final_dataset/test/real"  # Dossier contenant les images réelles
    cartoon_path = "final_dataset/test/cartoon"  # Dossier contenant les images cartoonisées

    # Charger le modèle
    model = tf.keras.models.load_model(model_path)
    print("Modèle chargé avec succès !")

    # Charger le dataset de test
    def load_dataset(real_path, cartoon_path):
        real_images = tf.data.Dataset.list_files(os.path.join(real_path, '*.png'), shuffle=False).map(load_image)
        cartoon_images = tf.data.Dataset.list_files(os.path.join(cartoon_path, '*.png'), shuffle=False).map(load_image)
        return tf.data.Dataset.zip((real_images, cartoon_images)).batch(16)  # Batch de 16 images

    test_dataset = load_dataset(real_path, cartoon_path)

    # Valider sur le dataset de test
    validate_on_test_set(model, test_dataset)

    # Valider sur de nouvelles images
    new_images = ["1.png", "stephen.jpg"]  # Noms des nouvelles images à tester
    validate_on_new_images(model, new_images)
