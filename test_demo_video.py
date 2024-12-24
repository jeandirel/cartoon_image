import tensorflow as tf
import numpy as np
import cv2
import os

def load_image_from_frame(frame):
    """
    Prétraite une image (frame) extraite d'une vidéo en la redimensionnant et en la normalisant.

    :param frame: Frame (image) de la vidéo.
    :return: Image prétraitée sous forme de tensor.
    """
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convertir de BGR (OpenCV) à RGB (TensorFlow)
    img = cv2.resize(img, (256, 256))  # Redimensionner l'image
    img = tf.convert_to_tensor(img, dtype=tf.float32)  # Convertir en tensor float32
    img = (img / 127.5) - 1  # Normalisation entre [-1, 1]
    return img

def cartoonize_video(input_video_path, output_video_path, model_path):
    """
    Applique le modèle de cartoonisation à une vidéo et génère une nouvelle vidéo transformée.

    :param input_video_path: Chemin vers la vidéo d'entrée.
    :param output_video_path: Chemin pour enregistrer la vidéo cartoonisée.
    :param model_path: Chemin vers le modèle de génération.
    """
    # Charger le modèle
    model = tf.keras.models.load_model(model_path)

    # Charger la vidéo
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Erreur: Impossible d'ouvrir la vidéo.")
        return

    # Récupérer les propriétés de la vidéo
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialiser l'écriture vidéo
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Traiter chaque frame
    print("Traitement en cours...")
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        # Prétraiter la frame
        preprocessed_frame = load_image_from_frame(frame)
        preprocessed_frame = tf.expand_dims(preprocessed_frame, axis=0)  # Ajouter une dimension batch

        # Générer la frame cartoonisée
        cartoonized_frame = model(preprocessed_frame, training=False)
        cartoonized_frame = (cartoonized_frame[0].numpy() + 1) * 127.5  # Re-normalisation
        cartoonized_frame = cartoonized_frame.astype('uint8')  # Conversion en uint8
        cartoonized_frame = cv2.cvtColor(cartoonized_frame, cv2.COLOR_RGB2BGR)  # Convertir de RGB à BGR pour OpenCV

        # Redimensionner à la taille originale et écrire la frame
        cartoonized_frame = cv2.resize(cartoonized_frame, (frame_width, frame_height))
        out.write(cartoonized_frame)

        # Affichage en console pour le suivi
        print(f"Frame {i+1}/{frame_count} traitée")

    # Libérer les ressources
    cap.release()
    out.release()
    print(f"Vidéo cartoonisée sauvegardée dans {output_video_path}")

# Exemple d'utilisation
if __name__ == "__main__":
    cartoonize_video(
        input_video_path="video.mp4", 
        output_video_path="output_video_cartoon.mp4", 
        model_path="models/pix2pix_generator_epoch_100.h5"
    )
