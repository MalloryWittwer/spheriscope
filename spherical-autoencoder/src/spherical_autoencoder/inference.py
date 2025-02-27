import numpy as np
from tensorflow import keras
from transformers import AutoImageProcessor, AutoModel

from spherical_autoencoder.model import unit_vectorize


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
        z, y, x = cartesian_coords
        theta = np.arccos(z)
        phi = np.arctan2(y, x)
        theta = float(theta)
        phi = float(phi)
        
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

        cls_outputs = outputs[:, 0, :]

        cartesian_coords = self.encoder.predict(cls_outputs)
        cartesian_coords = cartesian_coords[0]  # Remove the batch dimension
        z, y, x = cartesian_coords
        theta = np.arccos(z)
        phi = np.arctan2(y, x)
        theta = float(theta)
        phi = float(phi)
        
        return {
            "theta": theta,
            "phi": phi,
        }