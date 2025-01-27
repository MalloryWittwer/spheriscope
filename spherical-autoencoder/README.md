# Spherical Autoencoders

![animation](https://github.com/MalloryWittwer/spheriscope/blob/master/screenshots/anim_autoencoder.gif?raw=true)

Autoencoders are neural networks designed to project input data onto a low-dimensional space (the latent space) and then reconstruct it as accurately as possible. This package provides an implementation of an autoencoder with a spherical latent space. This constraint makes it possible to navigate and visualize images projected onto the autoencoder's latent space using the [Spheriscope](../README.md) app.

## Installation

To install the package from the git repository, run:

```
pip install git+https://github.com/MalloryWittwer/spheriscope.git
```

## Training a Model

To train a model, you need a **dataset of images**. You can use a subset of the [MNIST](https://search.brave.com/search?q=mnist&source=desktop) dataset as an example.

Ensure that:

- The images can be resized to **64x64 pixels** without losing significant visual quality. The autoencoder operates at this image size. If the images are not already this size, they will be resized internally by the autoencoder.
- The images are **grayscale** with pixel values between 0 and 255. If the images are in RGB, they will be converted to grayscale.

Save your images in PNG, JPG, or TIF format in a dataset folder. For example:

```
images/
├── img1.png
├── img2.png
├── ...
```

Then, run the training script:

```sh
spheriscope train <images_dir> <model_output_dir> --epochs 1000 --batch_size 32
```

This will save a model file `encoder.keras` in the specified output directory.

## Projecting an Image

To project an image onto the autoencoder's latent space, use the script:

```sh
spheriscope predict <image_file> <model_path>
```

This will print the polar coordinates of the image in the latent space, for example: `encoded_image={'theta': 2.096, 'phi': 0.549}`.

## Using *Spheriscope*

To visualize a dataset of images on the autoencoder's latent space, install and run the [Spheriscope](../README.md) app.

Then, upload the images to the backend server using:

```sh
spheriscope upload <images_dir> <model_output_dir>
```

After the command completes, reload the page at http://localhost:3000. You should see the projected image dataset 🎉.

