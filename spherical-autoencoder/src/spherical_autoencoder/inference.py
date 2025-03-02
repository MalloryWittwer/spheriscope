import numpy as np
from tensorflow import keras
from transformers import AutoImageProcessor, AutoModel

from spherical_autoencoder.model import unit_vectorize


def cartesian2spherical(cartesian_coords):
    z, y, x = cartesian_coords
    theta = float(np.arccos(z))
    phi = float(np.arctan2(y, x))
    return theta, phi


class TrainedSphericalEncoder:
    def __init__(self, keras_file: str):
        self.encoder = keras.models.load_model(
            keras_file,
            custom_objects={"unit_vectorize": unit_vectorize},
        )

    def predict(self, image: np.ndarray):
        # Add a channel dimension if the images are grayscale
        if len(image.shape) == 2:
            image = np.expand_dims(image[None], -1)
        
        cartesian_coords = self.encoder.predict(image)
        cartesian_coords = cartesian_coords[0]  # Remove the batch dimension

        theta, phi = cartesian2spherical(cartesian_coords)
        
        return {
            "theta": theta,
            "phi": phi,
        }


class TrainedDinoV2SphericalEncoder:
    def __init__(self, keras_file: str):
        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        self.dinov2_model = AutoModel.from_pretrained("facebook/dinov2-small")

        self.encoder = keras.models.load_model(
            keras_file,
            custom_objects={"unit_vectorize": unit_vectorize},
        )

    def predict(self, image: np.ndarray):
        inputs = self.processor(image, return_tensors="pt")

        outputs = self.dinov2_model(**inputs)
        outputs = outputs.last_hidden_state.cpu().detach().numpy()

        cls_outputs = outputs[:, 0, :]  # Classification features

        cartesian_coords = self.encoder.predict(cls_outputs)
        cartesian_coords = cartesian_coords[0]  # Remove the batch dimension

        theta, phi = cartesian2spherical(cartesian_coords)
        
        return {
            "theta": theta,
            "phi": phi,
        }