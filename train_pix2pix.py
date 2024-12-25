import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, ReLU, LeakyReLU, Dropout, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Vérifier la disponibilité du GPU et configurer l'utilisation de la mémoire
def setup_gpu():
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU(s) configuré(s) : {[gpu.name for gpu in gpus]}")
            return True
        else:
            print("Aucun GPU détecté. Utilisation du CPU à la place.")
            return False
    except RuntimeError as e:
        print(f"Erreur lors de la configuration GPU : {e}")
        return False

if setup_gpu():
    print("TensorFlow utilisera le GPU pour les calculs.")
else:
    print("TensorFlow utilisera uniquement le CPU.")

# Fonction pour l'augmentation de données
def data_augmentation(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    return image

# Charger une image, la décode et la normalise
def load_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, [256, 256])
    img = (img / 127.5) - 1  # Normalisation entre [-1, 1]
    return img

# Charger le dataset et appliquer l'augmentation de données
def load_dataset(real_path, cartoon_path):
    real_images = tf.data.Dataset.list_files(os.path.join(real_path, '*.png'), shuffle=False)
    cartoon_images = tf.data.Dataset.list_files(os.path.join(cartoon_path, '*.png'), shuffle=False)
    
    real_images = real_images.map(lambda x: data_augmentation(load_image(x)))
    cartoon_images = cartoon_images.map(lambda x: data_augmentation(load_image(x)))
    
    dataset = tf.data.Dataset.zip((real_images, cartoon_images))
    return dataset.batch(16).shuffle(100)

# Charger les datasets train et test
train_dataset = load_dataset('final_dataset/train/real', 'final_dataset/train/cartoon')
test_dataset = load_dataset('final_dataset/test/real', 'final_dataset/test/cartoon')

# Construire le générateur (U-Net)
def build_generator():
    inputs = Input(shape=(256, 256, 3))
    x = inputs

    # Downsampling
    down_stack = [
        Conv2D(64, 4, strides=2, padding='same'),
        Conv2D(128, 4, strides=2, padding='same'),
        Conv2D(256, 4, strides=2, padding='same'),
        Conv2D(512, 4, strides=2, padding='same')
    ]
    skips = []
    for down in down_stack:
        x = down(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        skips.append(x)

    # Upsampling
    up_stack = [
        Conv2DTranspose(512, 4, strides=2, padding='same'),
        Conv2DTranspose(256, 4, strides=2, padding='same'),
        Conv2DTranspose(128, 4, strides=2, padding='same'),
        Conv2DTranspose(64, 4, strides=2, padding='same')
    ]
    skips = reversed(skips[:-1])
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = ReLU()(x)
        x = Concatenate()([x, skip])

    x = Conv2DTranspose(3, 4, strides=2, padding='same', activation='tanh')(x)
    return Model(inputs, x)

# Construire le discriminateur
def build_discriminator():
    inputs_real = Input(shape=(256, 256, 3))
    inputs_cartoon = Input(shape=(256, 256, 3))
    x = Concatenate()([inputs_real, inputs_cartoon])
    x = Conv2D(64, 4, strides=2, padding='same')(x)
    x = LeakyReLU()(x)
    x = Conv2D(128, 4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(256, 4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(1, 4, strides=1, padding='same', activation='sigmoid')(x)
    return Model([inputs_real, inputs_cartoon], x)

generator = build_generator()
discriminator = build_discriminator()

# Optimiseurs et fonctions de perte
gen_optimizer = Adam(2e-4, beta_1=0.5)
disc_optimizer = Adam(2e-4, beta_1=0.5)
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target):
    adv_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    return adv_loss + (100 * l1_loss)

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    return real_loss + generated_loss

# Fonction pour calculer SSIM et PSNR
def calculate_metrics(real, generated):
    ssim_value = tf.image.ssim(real, generated, max_val=2.0)
    psnr_value = tf.image.psnr(real, generated, max_val=1.0)
    return ssim_value, psnr_value

# Entraînement du modèle
@tf.function
def train_step(real_image, cartoon_image):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(real_image, training=True)
        disc_real_output = discriminator([real_image, cartoon_image], training=True)
        disc_generated_output = discriminator([real_image, gen_output], training=True)
        
        gen_loss = generator_loss(disc_generated_output, gen_output, cartoon_image)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
    
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    gen_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return gen_loss, disc_loss

# Boucle d'entraînement
EPOCHS = 300  # Augmenter le nombre d'epochs
for epoch in range(EPOCHS):
    for real_image, cartoon_image in train_dataset:
        gen_loss, disc_loss = train_step(real_image, cartoon_image)
    
    print(f"Epoch {epoch+1}, Gen Loss: {gen_loss}, Disc Loss: {disc_loss}")

    # Sauvegarder le modèle après chaque epoch (optionnel)
    if (epoch + 1) % 5 == 0:
        generator.save(f"pix2pix_generator_epoch_{epoch+1}.h5")
        print(f"Modèle sauvegardé après {epoch+1} epochs.")

    # Calculer SSIM et PSNR sur un échantillon de test
    if (epoch + 1) % 10 == 0:
        for real_image, cartoon_image in test_dataset.take(1):  # Prendre un lot pour l'évaluation
            generated_image = generator(real_image, training=False)
            ssim_value, psnr_value = calculate_metrics(cartoon_image, generated_image)
            print(f"Epoch {epoch+1} - SSIM: {ssim_value.numpy()}, PSNR: {psnr_value.numpy()}")

# Sauvegarder le modèle final
generator.save("models/pix2pix_generator_final.h5")
print("Modèle final sauvegardé sous le nom pix2pix_generator_final.h5")
