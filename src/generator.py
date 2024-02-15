import sys
import logging
import argparse
import torch.nn as nn
from collections import OrderedDict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    filemode="w",
    filename="./logs/generator.log",
)

sys.path.append("src/")

from utils import total_params, model_info


class Generator(nn.Module):
    """
    A generator model for a Generative Adversarial Network (GAN) that generates images from a latent space.

    This model takes a latent space vector as input and outputs an image. It is comprised of a series
    of transposed convolutional layers that progressively upsample the input vector to the desired
    output image size and depth. Batch normalization and ReLU activations are used between layers,
    with a Tanh activation at the output to generate images with pixel values in the range [-1, 1].

    Parameters
    ----------
    latent_space : int, default=50
        The dimensionality of the latent space vector from which images are generated.
    image_size : int, default=64
        Defines the depth of the output images. This value is scaled throughout the network to
        determine the depth of intermediate layers.
    in_channels : int, default=1
        The number of channels in the output images. For example, this should be 3 for RGB images
        and 1 for grayscale images.

    Attributes
    ----------
    layers_config : list of tuple
        A configuration list where each tuple defines the parameters for a layer in the model.
        These parameters include the number of input channels, output channels, kernel size, stride,
        padding, and a boolean indicating whether batch normalization should be applied.
    model : nn.Sequential
        The sequential model comprising all layers specified in `layers_config`.

    Methods
    -------
    forward(x):
        Defines the forward pass of the generator.

    Notes
    -----
    The generator is a fundamental component of GANs, tasked with creating images that are
    indistinguishable from real images to the discriminator. It learns to map points from the
    latent space to the space of real images.

    Examples
    --------
    >>> generator = Generator(latent_space=100, image_size=64, in_channels=3)
    >>> print(generator)
    Generator(
        (model): Sequential(...)
    )

    The model can be used within a GAN framework to generate images from random noise vectors.
    """

    def __init__(self, latent_space=50, image_size=64, in_channels=1):
        self.latent_space = latent_space
        self.image_size = image_size
        self.in_channels = in_channels
        self.layers_config = [
            (self.latent_space, image_size * 8, 4, 1, 0, True),
            (self.image_size * 8, image_size * 4, 4, 2, 1, True),
            (self.image_size * 4, image_size * 2, 4, 2, 1, True),
            (self.image_size * 2, image_size, 4, 2, 1, True),
            (self.image_size, self.in_channels, 4, 2, 1),
        ]
        super(Generator, self).__init__()

        self.model = self.connected_layer(layers_config=self.layers_config)

    def connected_layer(self, layers_config=None):
        """
        Constructs a sequence of layers based on a given configuration.

        Parameters
        ----------
        layers_config : list of tuple, optional
            A list where each tuple contains parameters for each layer in the model.

        Returns
        -------
        nn.Sequential
            A sequential container of the configured layers.

        Raises
        ------
        ValueError
            If `layers_config` is empty or not provided.
        """
        layers = OrderedDict()
        if layers_config is not None:
            for index, (
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                batch_norm,
            ) in enumerate(self.layers_config[:-1]):
                layers[f"{index + 1}_convTrans"] = nn.ConvTranspose2d(
                    in_channels, out_channels, kernel_size, stride, padding
                )
                if batch_norm:
                    layers[f"{index + 1}_batchNorm"] = nn.BatchNorm2d(out_channels)

                layers[f"{index + 1}_relu"] = nn.ReLU()

            (in_channels, out_channels, kernel_size, stride, padding) = (
                self.layers_config[-1]
            )
            layers["out_convTrans"] = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding
            )
            layers["out_tanh"] = nn.Tanh()

            return nn.Sequential(layers)
        else:
            raise ValueError("Layers config is empty".capitalize())

    def forward(self, x):
        """
        Defines the forward pass of the generator.

        Parameters
        ----------
        x : torch.Tensor
            The input latent space vector.

        Returns
        -------
        torch.Tensor
            The output image tensor with pixel values in the range [-1, 1].
        """
        return self.model(x) if x is not None else "ERROR"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Generator".capitalize())

    parser.add_argument(
        "--image_size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument(
        "--in_channels",
        type=int,
        default=1,
        help="Image input channels size for training",
    )
    parser.add_argument(
        "--latent_space",
        type=int,
        default=1,
        help="Image latent dimension size for training",
    )

    args = parser.parse_args()

    if args.image_size and args.in_channels and args.latent_space:
        logging.info(f"Image size: {args.image_size}")
        net_G = Generator(
            image_size=args.image_size,
            in_channels=args.in_channels,
            latent_space=args.latent_space,
        )

        try:
            logging.info(
                "Total params in the Generator # {}".format(total_params(model=net_G))
            )
            logging.info(list(model_info(model=net_G)))
        except ValueError as e:
            logging.error(f"Error caught in the section # {e}".capitalize())
    else:
        raise ValueError("Image size and input channels are required".capitalize())
