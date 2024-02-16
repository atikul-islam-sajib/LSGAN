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
    A discriminator model within the Generative Adversarial Network (GAN) framework, tasked with distinguishing between real and synthetic images.

    Constructed from a series of convolutional layers, this model evaluates the authenticity of input images, leveraging batch normalization and LeakyReLU activations, and finalizing with a single output node to indicate the likelihood of an image being real.

    ### Parameters

    | Parameter        | Type | Default | Description                                                                 |
    |------------------|------|---------|-----------------------------------------------------------------------------|
    | `image_size`     | int  | 64      | Initial layer depth, typically aligned with input image dimensionality.    |
    | `input_channels` | int  | 3       | Channel count for input images (3 for RGB, 1 for grayscale).                |

    ### Attributes

    | Attribute      | Type          | Description                                                                 |
    |----------------|---------------|-----------------------------------------------------------------------------|
    | `layers_config`| list of tuples| Layer configurations detailing input/output channels, kernel size, etc.     |
    | `model`        | nn.Sequential | Composed model from `layers_config`, representing the discriminator's architecture. |

    ### Methods

    - `forward(x)`: Executes the discriminator's forward pass, inputting an image tensor and outputting its realness probability.

    ### Notes

    As GANs' evaluative mechanism, the discriminator assesses generator output quality, aiming to accurately identify generated content, thereby guiding the generator's improvements.

    ### Examples

    ```python
    discriminator = Discriminator(image_size=64, input_channels=3)
    print(discriminator)
    ```

    This instantiation creates a discriminator for 64x64 RGB images, ready to participate in GAN training to discern real from generated images.
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
