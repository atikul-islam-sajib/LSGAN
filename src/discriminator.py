import sys
import logging
import argparse
import torch.nn as nn
from collections import OrderedDict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    filemode="w",
    filename="./logs/discriminator.log",
)

sys.path.append("src/")

from utils import total_params, model_info


class Discriminator(nn.Module):
    """
    A discriminator model for a Generative Adversarial Network (GAN), designed to distinguish
    between real and generated images.

    This model acts as the adversarial counterpart to a generator model in GAN architecture, aiming
    to classify images as real or fake. It is implemented as a sequence of convolutional layers,
    optionally followed by batch normalization and LeakyReLU activation, culminating in a single
    output node representing the probability of the input image being real.

    Parameters
    ----------
    image_size : int, default=64
        Defines the depth of the first convolutional layer. Typically, this is set to match the
        dimensionality of the images being discriminated. The depth of subsequent layers is scaled
        from this initial value.
    input_channels : int, default=3
        The number of channels in the input images. For example, this should be 3 for RGB images
        and 1 for grayscale images.

    Attributes
    ----------
    layers_config : list of tuple
        A configuration list where each tuple defines the parameters for a layer in the model.
        These parameters include the number of input channels, output channels, kernel size, stride,
        padding, slope of LeakyReLU, and a boolean indicating whether batch normalization should be
        applied.
    model : nn.Sequential
        The sequential model comprising all layers specified in `layers_config`.

    Methods
    -------
    forward(x):
        Defines the forward pass of the discriminator.

    Notes
    -----
    The discriminator is a critical component of GANs, providing feedback to the generator on the
    quality of its output. It is trained to minimize the probability assigned to generated images
    while maximizing the probability assigned to real images.

    Examples
    --------
    >>> discriminator = Discriminator(image_size=64, input_channels=3)
    >>> print(discriminator)
    Discriminator(
        (model): Sequential(...)
    )

    The model can then be used in the training loop of a GAN, where it will be trained to distinguish
    between real and fake images.
    """

    def __init__(self, image_size=64, input_channels=3):
        self.image_size = image_size
        self.input_channels = input_channels
        self.layers_config = [
            (self.input_channels, self.image_size, 4, 2, 1, 0.2, False),
            (self.image_size, self.image_size * 2, 4, 2, 1, 0.2, True),
            (self.image_size * 2, self.image_size * 4, 4, 2, 1, 0.2, True),
            (self.image_size * 4, self.image_size * 8, 4, 2, 1, 0.2, True),
            (self.image_size * 8, 1, 4, 1, 0),
        ]
        super(Discriminator, self).__init__()

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
                slope,
                batch_norm,
            ) in enumerate(layers_config[:-1]):
                layers[f"{index + 1}_conv"] = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
                if batch_norm:
                    layers[f"{index + 1}_batch_norm"] = nn.BatchNorm2d(out_channels)

                layers[f"{index + 1}_leaky_relu"] = nn.LeakyReLU(slope)

            (in_channels, out_channels, kernel_size, stride, padding) = layers_config[
                -1
            ]
            layers[f"out_conv"] = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )

            return nn.Sequential(layers)
        else:
            raise ValueError("layers_config is empty".capitalize())

    def forward(self, x):
        """
        Defines the forward pass of the discriminator.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor containing a batch of images.

        Returns
        -------
        torch.Tensor
            The output tensor representing the probability of the input being real images.
        """
        return self.model(x) if x is not None else "ERROR"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Discriminator".capitalize())

    parser.add_argument(
        "--image_size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument(
        "--in_channels",
        type=int,
        default=1,
        help="Image input channels size for training",
    )

    args = parser.parse_args()

    if args.image_size and args.in_channels:
        logging.info(f"Image size: {args.image_size}")
        net_D = Discriminator(
            image_size=args.image_size, input_channels=args.in_channels
        )

        try:
            logging.info(
                "Total params in the discriminator # {}".format(
                    total_params(model=net_D)
                )
            )
            logging.info(list(model_info(model=net_D)))
        except ValueError as e:
            logging.error(f"Error caught in the section # {e}".capitalize())
    else:
        raise ValueError("Image size and input channels are required".capitalize())
