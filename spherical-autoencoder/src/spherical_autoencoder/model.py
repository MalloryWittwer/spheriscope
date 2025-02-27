import numpy as np
from pathlib import Path
import skimage.io
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.saving import register_keras_serializable


def reconstruction_loss(y_true, outputs):
    y_pred_flat = outputs[:, :-3]
    y_true_rescaled = tf.image.resize(y_true, [64, 64]) / 255.0
    y_true_flat = layers.Flatten()(y_true_rescaled)
    loss = tf.reduce_mean(tf.square(y_true_flat - y_pred_flat), axis=1)
    return loss


def variance_loss(y_true, outputs):
    z = outputs[:, -3:]
    var = tf.math.reduce_variance(z, axis=0)  # (3,) - batch varianc along x, y and z
    loss = 1 / tf.reduce_mean(var)  # (1,) - variance in position in the batch
    loss = tf.sqrt(loss)
    return loss


def autoencoder_loss(y_true, outputs):
    return tf.reduce_mean(reconstruction_loss(y_true, outputs) + variance_loss(y_true, outputs))


@register_keras_serializable(name="unit_vectorize")
def unit_vectorize(x):
    return tf.math.divide(x, tf.expand_dims(tf.norm(x, axis=1), -1))


def load_images_from_folder(folder_path: str) -> np.ndarray:
    image_files = list(Path(folder_path).iterdir())
    X_train = np.array([skimage.io.imread(file, as_gray=True) for file in image_files])
    # Normalize the gray levels to 0-255
    X_train = X_train.astype(np.float64)
    X_train = X_train - X_train.min()
    X_train = X_train / X_train.max()
    X_train = X_train * 255
    X_train = X_train.astype(np.uint8)  # Images should be RGB
    return X_train


class SphericalAutoencoder:
    def __init__(self, optimizer="rmsprop"):
        inputs = layers.Input(shape=(None, None, 1))
        resize_layer = layers.Resizing(64, 64)(inputs)  # Images are resized to 64x64px
        rescale_layer = layers.Rescaling(1.0 / 255)(
            resize_layer
        )  # Images are scaled to 0-1

        x = layers.Conv2D(2, (3, 3), activation="relu", padding="same", strides=4)(
            rescale_layer
        )
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2), padding="same")(x)

        x = layers.Flatten()(x)
        x = layers.Dense(16, activation="relu")(x)

        z_mean = layers.Dense(3, name="z_mean")(x)
        z_mean = layers.BatchNormalization()(z_mean)

        bottleneck = layers.Lambda(unit_vectorize, name="bottleneck")(z_mean)

        # latent_inputs = layers.Input(shape=(3,))  # For the decoder

        x = layers.Dense(16, activation="relu")(bottleneck)
        x = layers.Dense(32, activation="relu")(x)
        x = layers.Reshape((1, 1, 32))(x)
        x = layers.Conv2DTranspose(
            32, (3, 3), strides=2, activation="relu", padding="same"
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2DTranspose(
            16, (3, 3), strides=2, activation="relu", padding="same"
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2DTranspose(
            8, (3, 3), strides=2, activation="relu", padding="same"
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2DTranspose(
            4, (3, 3), strides=2, activation="relu", padding="same"
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2DTranspose(
            2, (3, 3), strides=2, activation="relu", padding="same"
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2DTranspose(
            1, (3, 3), strides=2, activation="relu", padding="same"
        )(x)
        x = layers.BatchNormalization()(x)
        reconstruction = layers.Conv2D(
            1, (3, 3), activation="sigmoid", padding="same", name="reconstruction"
        )(x)
        flat_reconstruction = layers.Flatten()(reconstruction)
        vae_outputs = layers.Concatenate()([flat_reconstruction, bottleneck])

        # decoder = Model(latent_inputs, reconstruction, name="decoder")
        self.encoder = Model(inputs, bottleneck, name="encoder")
        self.autoencoder = Model(inputs, vae_outputs, name="autoencoder")

        self.autoencoder.compile(
            optimizer=optimizer,
            loss=autoencoder_loss,
            metrics=[variance_loss, reconstruction_loss],
        )

    def train(self, images_folder, epochs: int = 1, batch_size: int = 32):
        X_train = load_images_from_folder(images_folder)
        if len(X_train.shape) == 3:
            X_train = X_train[..., None]  # Add a channel dimension

        history = self.autoencoder.fit(
            x=X_train,
            y=X_train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            # validation_data=(X_test, X_test),
        )
        return history

    def save_encoder(self, save_path):
        """Save the encoder in .keras format"""
        self.encoder.save(save_path)
        print(f"Encoder saved to: {save_path}")
