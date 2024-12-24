import os
import numpy as np
import tensorflow as tf
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt

def load_image(image_path):
    """
    Charge une image à partir du chemin donné et effectue le prétraitement.

    Args:
        image_path (str): Chemin de l'image à charger.

    Returns:
        tensorflow.Tensor: Image redimensionnée et normalisée.
    """
    image = tf.io.read_file(image_path)  # Lecture de l'image depuis le fichier
    image = tf.image.decode_png(image, channels=3)  # Décodage de l'image PNG avec 3 canaux
    image = tf.image.resize(image, [256, 256])  # Redimensionnement de l'image à 256x256 pixels
    image = (image / 127.5) - 1  # Normalisation des pixels entre -1 et 1
    return image

def evaluate_metrics(target_img, generated_img):
    """
    Calcule les métriques SSIM (Structural Similarity Index) et PSNR (Peak Signal-to-Noise Ratio) entre deux images.

    Args:
        target_img (numpy.ndarray): Image cible déjà cartoonisée (référence).
        generated_img (numpy.ndarray): Image générée par le modèle.

    Returns:
        tuple: SSIM et PSNR calculés entre les deux images.
    """
    # Dé-normalisation des images de [-1, 1] à [0, 255]
    target_img = (target_img + 1) * 127.5
    generated_img = (generated_img + 1) * 127.5

    # Conversion en type entier non signé pour les métriques
    target_img = target_img.astype(np.uint8)
    generated_img = generated_img.astype(np.uint8)

    # Taille de la fenêtre adaptée pour le calcul du SSIM
    min_dim = min(target_img.shape[:2])  # Plus petite dimension spatiale de l'image
    win_size = min(7, min_dim)  # La taille de la fenêtre doit être ≤ à la plus petite dimension

    # Calcul des métriques SSIM et PSNR
    ssim_score = ssim(
        target_img,
        generated_img,
        data_range=255,
        win_size=win_size,
        channel_axis=-1  # Indique que les canaux couleur sont sur le dernier axe
    )
    psnr_score = psnr(target_img, generated_img, data_range=255)

    return ssim_score, psnr_score

def evaluate_model(model_path, real_path, cartoon_path):
    """
    Évalue un modèle sur un ensemble de test en calculant SSIM et PSNR entre les images réelles et cartoonisées.

    Args:
        model_path (str): Chemin du modèle préentraitné sauvegardé.
        real_path (str): Chemin des images réelles de test.
        cartoon_path (str): Chemin des images cartoonisées de test.
    """
    # Charger le modèle préentraitné
    model = tf.keras.models.load_model(model_path)
    print("Modèle chargé avec succès.")

    # Récupérer les chemins des images réelles et cartoonisées
    real_images = sorted(tf.io.gfile.glob(os.path.join(real_path, '*.png')))
    cartoon_images = sorted(tf.io.gfile.glob(os.path.join(cartoon_path, '*.png')))

    # Vérifier que le nombre d'images réelles correspond au nombre d'images cartoonisées
    if len(real_images) != len(cartoon_images):
        print("Erreur : Le nombre d'images réalistes et cartoonisées ne correspond pas.")
        return

    total_ssim = 0
    total_psnr = 0
    num_images = len(real_images)

    for real_img_path, cartoon_img_path in zip(real_images, cartoon_images):
        # Charger les images réelles et cibles
        real_img = load_image(real_img_path).numpy()
        target_img = load_image(cartoon_img_path).numpy()

        # Vérification des dimensions minimales pour le SSIM
        if real_img.shape[0] < 7 or real_img.shape[1] < 7:
            print(f"Image trop petite pour SSIM : {real_img_path}")
            continue

        # Générer une image cartoonisée avec le modèle
        generated_img = model(tf.expand_dims(real_img, axis=0), training=False)[0].numpy()

        # Calculer les métriques entre l'image cible et l'image générée
        ssim_score, psnr_score = evaluate_metrics(target_img, generated_img)
        total_ssim += ssim_score
        total_psnr += psnr_score

        # Afficher les résultats pour chaque image
        print(f"Image : {os.path.basename(real_img_path)} | SSIM : {ssim_score:.4f}, PSNR : {psnr_score:.4f}")

    # Calculer les métriques moyennes sur tout l'ensemble de test
    avg_ssim = total_ssim / num_images
    avg_psnr = total_psnr / num_images

    print("\n--- Résultats finaux ---")
    print(f"Moyenne SSIM : {avg_ssim:.4f}")
    print(f"Moyenne PSNR : {avg_psnr:.4f}")

# Exécution principale si le script est appelé directement
if __name__ == "__main__":
    evaluate_model(
        model_path="models/pix2pix_generator_epoch_100.h5",  # Chemin vers le modèle sauvegardé
        real_path="final_dataset/test/real",  # Chemin vers les images réalistes de test
        cartoon_path="final_dataset/test/cartoon"  # Chemin vers les images cartoonisées cibles
    )
