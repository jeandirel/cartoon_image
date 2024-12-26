import tensorflow as tf
import matplotlib.pyplot as plt

def load_image(image_path):
    """
    Charge et prétraite une image en la redimensionnant et en la normalisant.
    
    Cette fonction charge une image à partir du chemin spécifié, la redimensionne à 256x256 pixels
    et applique une normalisation des pixels entre -1 et 1.

    :param image_path: Chemin vers l'image à charger.
    :return: Image prétraitée sous forme de tensor.
    """
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)  # Remplace decode_jpeg par decode_png si nécessaire
    img = tf.image.resize(img, [256, 256])  # Redimensionner l'image à 256x256 pixels
    img = (img / 127.5) - 1  # Normalisation des pixels entre [-1, 1]
    return img

def test_image(image_path, model_path):
    """
    Teste le modèle sur une image donnée, génère l'image cartoonisée et l'affiche.
    
    Cette fonction charge une image, l'applique au modèle de générateur pour obtenir une version cartoonisée,
    puis affiche côte à côte l'image originale et l'image transformée.

    :param image_path: Chemin vers l'image réaliste à tester.
    :param model_path: Chemin vers le modèle de génération.
    """
    # Charger l'image et le modèle
    img = load_image(image_path)  # Charger et prétraiter l'image
    img = tf.expand_dims(img, axis=0)  # Ajouter une dimension batch pour le modèle
    model = tf.keras.models.load_model(model_path)  # Charger le modèle de génération

    # Générer l'image cartoonisée
    cartoonized_img = model(img, training=False)  # Générer l'image cartoonisée
    cartoonized_img = (cartoonized_img[0].numpy() + 1) * 127.5  # Re-normaliser pour obtenir une image entre [0, 255]
    cartoonized_img = cartoonized_img.astype('uint8')  # Convertir en entier de type uint8

    # Affichage des résultats
    plt.figure(figsize=(8, 4))  # Définir la taille de la figure
    plt.subplot(1, 2, 1)  # Afficher l'image réaliste dans la première colonne
    plt.title("Image Réaliste")
    plt.imshow((img[0].numpy() + 1) / 2)  # Re-normalisation de l'image pour l'affichage (entre [0, 1])
    plt.axis('off')  # Masquer les axes
    plt.subplot(1, 2, 2)  # Afficher l'image cartoonisée dans la deuxième colonne
    plt.title("Image Cartoonisée")
    plt.imshow(cartoonized_img)  # Afficher l'image générée
    plt.axis('off')  # Masquer les axes
    plt.show()  # Afficher la figure avec les deux images

# Exemple d'utilisation
test_image("2.jpg", "models/pix2pix_generator_epoch_100.h5")
