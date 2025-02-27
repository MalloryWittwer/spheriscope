import numpy as np
import tensorflow as tf
from tensorflow.keras.saving import register_keras_serializable
from transformers import AutoImageProcessor, AutoModel
from tensorflow.keras import layers, Model
from pathlib import Path
import skimage.io
from skimage.transform import resize


def reconstruction_loss(y_true, outputs):
    y_pred = outputs[:, :-3]
    loss = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)
    return loss


def variance_loss(y_true, outputs):  # TODO: This loss doesn't seem to have any effect..
    y_pred = outputs[:, -3:]
    var = tf.math.reduce_variance(y_pred, axis=0)  # (3,) - batch varianc along x, y and z
    loss = -tf.reduce_mean(var)  # (1,) - variance in position in the batch
    return loss


def autoencoder_loss(y_true, outputs):
    return tf.reduce_mean(reconstruction_loss(y_true, outputs) + variance_loss(y_true, outputs))


@register_keras_serializable(name="unit_vectorize")
def unit_vectorize(x):
    return tf.math.divide(x, tf.expand_dims(tf.norm(x, axis=1), -1))


class DinoV2SphericalAutoencoder:
    def __init__(
        self, input_image_size=224, middle_layer_params=16, optimizer="rmsprop"
    ):
        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        self.dinov2_model = AutoModel.from_pretrained("facebook/dinov2-small")
        self.input_image_size = input_image_size
        self.middle_layer_params = middle_layer_params

        self.encoder, self.autoencoder = self._create_model()

        self.autoencoder.compile(
            optimizer=optimizer,
            loss=autoencoder_loss,
            # loss=variance_loss,
            metrics=[variance_loss, reconstruction_loss],
        )

    def _create_model(self):
        hidden_size = self.dinov2_model.config.hidden_size  # 384

        inputs = layers.Input(shape=(hidden_size,))
        x = layers.Dense(self.middle_layer_params, activation="relu")(inputs)
        x = layers.BatchNormalization()(x)
        z_mean = layers.Dense(3, name="z_mean")(x)
        z_mean = layers.BatchNormalization()(z_mean)
        bottleneck = layers.Lambda(
            unit_vectorize, name="bottleneck", output_shape=(3,)
        )(z_mean)
        x = layers.Dense(self.middle_layer_params, activation="relu")(bottleneck)
        x = layers.BatchNormalization()(x)
        flat_reconstruction = layers.Dense(hidden_size, activation=None)(x)
        vae_outputs = layers.Concatenate()([flat_reconstruction, bottleneck])

        encoder = Model(inputs, bottleneck, name="encoder")
        autoencoder = Model(inputs, vae_outputs, name="autoencoder")

        return encoder, autoencoder

    def train(self, images_folder, epochs: int = 1, batch_size: int = 32):
        image_files = list(Path(images_folder).iterdir())

        image_input_shape = (self.input_image_size, self.input_image_size)

        images = np.array(
            [
                resize(skimage.io.imread(file), output_shape=image_input_shape)
                for file in image_files
            ]
        )

        inputs = self.processor(images, return_tensors="pt")
        outputs = self.dinov2_model(**inputs)
        outputs = outputs.last_hidden_state.cpu().detach().numpy()

        cls_outputs = outputs[:, 0, :]

        history = self.autoencoder.fit(
            x=cls_outputs,
            y=cls_outputs,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
        )

        return history

    def save_encoder(self, save_path):
        """Save the encoder in .keras format"""
        self.encoder.save(save_path)
        print(f"Encoder saved to: {save_path}")
