# Embedding visualizer

Try the app at [https://embedding-visualizer.herokuapp.com/](https://embedding-visualizer.herokuapp.com/). This web app is meant as a tool to visualize and interact with the latent space of a convolutional VAE in an intuitive and user-friendly way. Users can "navigate" through the latent space by grabbing and panning over the canvas, and generate new points at desired locations. This is made possible by constraining the latent space of the VAE on a unit sphere, which is projected on the canvas using the orthographic, stereographic, or cylindrical sphere projection. The MNIST and Fashion-MNIST datasets are used as examples, however the technique could be applied or adapted to any other dataset.

#### Training a VAE with a spherical latent space

[Variational autoencoders](https://en.wikipedia.org/wiki/Variational_autoencoder) or VAE are neural networks that attempt to compress the input data (encoding) into a low-dimensional latent space, and then to reconstruct it as accurately as possible (decoding). Once the model is trained, new data can be generated from the latent space by the decoder model. Here, we have constrained the latent space to lie on the surface of a unit (three-dimensional) sphere. Therefore, the latent space is [closed](https://en.wikipedia.org/wiki/Surface_(topology)#Closed_surfaces); it is compact, without any boundaries. Introducing this constraint makes visualization and navigation of the latent space in our web app optimal for a user.

#### Visualizing and navigating the latent space

Soon.

#### Contact

Any questions about this project?

=> Send me a 📧 at **mallory.wittwer@gmail.com**.
