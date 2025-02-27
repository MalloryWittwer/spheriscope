import os
import argparse
from pathlib import Path
from spherical_autoencoder import SphericalAutoencoder, TrainedSphericalEncoder
from spherical_autoencoder import DinoV2SphericalAutoencoder, TrainedDinoV2SphericalEncoder
import matplotlib.pyplot as plt
import skimage.io
import requests


BACKEND_URL = "http://localhost:8000"


def train(autoencoder, images_dir, model_output_dir, epochs=1000, batch_size=32):
    model_output_dir = Path(model_output_dir).resolve()
    if not model_output_dir.exists():
        os.makedirs(model_output_dir)
        print("Created output folder: ", model_output_dir)

    history = autoencoder.train(images_dir, epochs=epochs, batch_size=batch_size)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, epochs)
    ax.plot(history.history["loss"], label="loss")
    ax.plot(history.history["reconstruction_loss"], label="reconstruction_loss")
    ax.plot(history.history["variance_loss"], label="variance_loss")
    ax.set_ylabel("loss")
    ax.set_xlabel("epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(model_output_dir / 'training_loss.png'))
    plt.close()
    
    model_path = str(model_output_dir / 'encoder.keras')
    autoencoder.save_encoder(model_path)
    print(f"Saved {model_path}")


def predict(encoder, image_file):
    image = skimage.io.imread(image_file)
    print(f"{image.shape=}")
    
    encoded_image = encoder.predict(image)
    print(f"{encoded_image=}")


def upload(encoder, images_dir):
    try:
        response = requests.get(BACKEND_URL)
        if response.status_code != 200:
            print(f"Backend URL {BACKEND_URL} is not available. Status code: {response.status_code}")
            return
    except requests.exceptions.RequestException as e:
        print(f"Failed to connect to {BACKEND_URL}. Exception: {e}")
        return

    for image_file in Path(images_dir).iterdir():
        image_file = str(image_file)
        print(f"Uploading {image_file}")

        image = skimage.io.imread(image_file)

        encoded_image = encoder.predict(image)
        print(f"{encoded_image=}")
        
        with open(image_file, 'rb') as img:
            response = requests.post(f"{BACKEND_URL}/images/", files={"file": img}, data=encoded_image)

        if response.status_code == 200:
            print("Response:", response.json())
        else:
            print("Failed to upload image. Status code:", response.status_code)
            print("Response:", response.text)


def main():
    parser = argparse.ArgumentParser(description='Spherical Autoencoder CLI')
    subparsers = parser.add_subparsers(dest='command')

    train_parser = subparsers.add_parser('train', help='Train the autoencoder')
    train_parser.add_argument('images_dir', help='Directory containing images')
    train_parser.add_argument('model_output_dir', help='Directory to save the trained model')
    train_parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    train_parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    train_parser.add_argument('--model', choices=['dinov2', 'cnn'], default='dinov2', help='Model type to use for training')
    # input_image_size=224, middle_layer_params=16

    predict_parser = subparsers.add_parser('predict', help='Predict using the trained autoencoder')
    predict_parser.add_argument('image_file', help='Image file to project')
    predict_parser.add_argument('model_path', help='Path to the trained model file')
    predict_parser.add_argument('--model', choices=['dinov2', 'cnn'], default='dinov2', help='Model type to use for prediction')
    
    upload_parser = subparsers.add_parser('upload', help='Upload images to the backend')
    upload_parser.add_argument('images_dir', help='Directory containing images')
    upload_parser.add_argument('model_path', help='Path to the trained model file')
    upload_parser.add_argument('--model', choices=['dinov2', 'cnn'], default='dinov2', help='Model type to use for prediction')

    args = parser.parse_args()

    if args.command == 'train':
        if args.model == "dinov2":
            autoencoder = DinoV2SphericalAutoencoder()
        else:
            autoencoder = SphericalAutoencoder()
        train(autoencoder, args.images_dir, args.model_output_dir, args.epochs, args.batch_size)
    elif args.command == 'predict':
        if args.model == "dinov2":
            encoder = TrainedDinoV2SphericalEncoder(args.model_path)
        else:
            encoder = TrainedSphericalEncoder(args.model_path)
        predict(encoder, args.image_file)
    elif args.command == 'upload':
        if args.model == "dinov2":
            encoder = TrainedDinoV2SphericalEncoder(args.model_path)
        else:
            encoder = TrainedSphericalEncoder(args.model_path)
        upload(encoder, args.images_dir)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()