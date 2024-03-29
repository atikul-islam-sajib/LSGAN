import sys
import logging
import argparse
import math
import numpy as np
import os
import torch
import imageio
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    filemode="w",
    filename="./logs/test.log",
)

sys.path.append("src/")

from config import BEST_MODEL_PATH, SAVED_IMAGE_PATH, GEN_PATH, GIF_PATH
from utils import device_init
from generator import Generator


class Test:
    """
        A class for testing a trained Generator model from a Generative Adversarial Network (GAN) to generate synthetic images and create a GIF animation from those images.

        This class is designed to load a trained Generator model, generate a specified number of synthetic images from random latent space vectors, save these images, and optionally create a GIF animation from a sequence of generated images.

    For documenting parameters and attributes in a concise, table-like format within a Markdown or similar text documentation, you can follow this structure:

    ### Parameters

    | Parameter      | Type  | Default | Description                                                                                       |
    |----------------|-------|---------|---------------------------------------------------------------------------------------------------|
    | `latent_space` | int   | 100     | The dimensionality of the latent space vector from which images are generated.                    |
    | `num_samples`  | int   | 20      | The number of synthetic images to generate.                                                       |
    | `device`       | str   | "mps"   | The device to run the model on ("cuda" for GPU or "cpu" for CPU). "mps" is used for Apple GPUs.  |


    ### Attributes

    | Attribute        | Type           | Description                                                                 |
    |------------------|----------------|-----------------------------------------------------------------------------|
    | `best_model_path`| str            | Path to the directory containing the trained Generator model file.          |
    | `generator`      | Generator      | The Generator model loaded with trained weights for image generation.       |
    | `device`         | torch.device   | The PyTorch device context, determining where the model computations occur. |

    This structured format provides a clear and accessible overview of the parameters and attributes, making it easier for readers to understand the key configurations and components of a class or function, especially when viewed in environments that support Markdown rendering.

    Methods
    -------
    test():
        Loads the trained Generator model, generates synthetic images, saves them, and creates a GIF animation.

    Notes
    -----
    This class assumes the presence of specific path constants (`BEST_MODEL_PATH`, `SAVED_IMAGE_PATH`, `GEN_PATH`, `GIF_PATH`) that are defined elsewhere in the project, typically in a configuration file.

    Examples
    --------
        >>> test = Test(latent_space=100, num_samples=20, device="cuda")
        >>> test.test()
    This will load the trained Generator model, generate 20 synthetic images, save them, and if possible, create a GIF animation of the generated images.

    Raises
    ------
        FileNotFoundError
            If the paths specified in the configuration for saving images or GIFs do not exist.
        Exception
            For errors encountered during image generation or GIF creation.
    """

    def __init__(self, latent_space=100, num_samples=20, device="mps"):
        self.latent_space = latent_space
        self.num_samples = num_samples
        self.best_model_path = BEST_MODEL_PATH
        self.generator = None
        self.device = device_init(device=device)

    def _plot_synthetic_image(self, images):
        """
        Plots and saves a grid of generated synthetic images.

        Parameters
        ----------
        images : torch.Tensor
            A batch of synthetic images generated by the Generator model.
        """
        plt.figure(figsize=(10, 8))

        num_columns = int(math.sqrt(self.num_samples))
        num_rows = self.num_samples // num_columns + (
            self.num_samples % num_columns > 0
        )

        for index, image in enumerate(images):
            plt.subplot(num_rows, num_columns, index + 1)
            to_image = image.cpu().detach().permute(1, 2, 0).numpy()
            to_image = (to_image - to_image.min()) / (to_image.max() - to_image.min())
            plt.imshow(to_image, cmap="gray")
            plt.axis("off")

        plt.tight_layout()

        if os.path.exists(SAVED_IMAGE_PATH):
            plt.savefig(os.path.join(SAVED_IMAGE_PATH, "fake_image.png"))
        else:
            raise FileNotFoundError(f"{SAVED_IMAGE_PATH} does not exist")
        plt.show()

    def _create_gif(self):
        """
        Main method to load the trained Generator model, generate synthetic images, save them,
        and create a GIF animation from those images.
        """
        if os.path.exists(GEN_PATH):
            images = [
                imageio.imread(os.path.join(GEN_PATH, image))
                for image in os.listdir(GEN_PATH)
            ]
            imageio.mimsave(
                os.path.join(GIF_PATH, "image.gif"), images, "GIF", duration=10
            )
        else:
            raise Exception("Generated image path is not found".capitalize())

    def test(self):
        if os.path.exists(self.best_model_path):
            self.generator = Generator(
                latent_space=self.latent_space, image_size=64, in_channels=3
            ).to(self.device)
            load_state_dict = torch.load(
                os.path.join(self.best_model_path, "best_model.pth")
            )
            self.generator.load_state_dict(load_state_dict)
            noise_samples = torch.randn(self.num_samples, self.latent_space, 1, 1).to(
                self.device
            )
            generated_images = self.generator(noise_samples)

            logging.info("Saving the generated images".capitalize())

            try:
                self._plot_synthetic_image(images=generated_images)
                self._create_gif()
            except Exception as e:
                print(
                    f"Error occurred while creating the synthetic images: {e}".capitalize()
                )
                logging.info(
                    f"Error occurred while creating the synthetic images: {e}".capitalize()
                )
        else:
            raise Exception("No model found".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Creating the synthetic images".title()
    )
    parser.add_argument(
        "--latent_space",
        type=int,
        default=100,
        help="Define the latent space".capitalize(),
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Define the number of samples".capitalize(),
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Define the device".capitalize()
    )

    args = parser.parse_args()

    if args.latent_space and args.num_samples and args.device:
        logging.info("Creating the synthetic images".capitalize())

        try:
            test = Test()
        except Exception as e:
            logging.info("Exception caught in the section # {}".format(e))
        else:
            test.test()

        logging.info("Synthetic images created successfully".capitalize())
    else:
        raise Exception("Invalid arguments".capitalize())
