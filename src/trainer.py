import sys
import logging
import argparse
import torch
import os
import numpy as np
import joblib as pickle
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    filemode="w",
    filename="./logs/trainer.log",
)

sys.path.append("src/")

from config import PROCESSED_PATH, GEN_PATH, CHECKPOINT_PATH, BEST_MODEL_PATH
from utils import weight_init, device_init, clean_folder
from generator import Generator
from discriminator import Discriminator


class Trainer:
    """
    A class for training a Generative Adversarial Network (GAN) comprising a Generator and Discriminator model.

    This class handles the training process of GANs, including initializing the models, setting up the
    optimizers, and running the training loop. It supports saving model checkpoints and generating synthetic
    images during training to visualize progress.

    Parameters
    ----------
    image_size : int, default=64
        The size of the images to generate. Typically a power of 2.
    input_channels : int, default=3
        The number of channels in the images to generate. For example, 3 for RGB images.
    latent_space : int, default=50
        The dimensionality of the latent space from which the generator creates images.
    lr : float, default=0.0002
        The learning rate for the Adam optimizers.
    epochs : int, default=100
        The number of training epochs.
    beta1 : float, default=0.5
        The beta1 hyperparameter for the Adam optimizers.
    beta2 : float, default=0.999
        The beta2 hyperparameter for the Adam optimizers.
    device : str, default="mps"
        The device to run the training on ("cuda" for GPU or "cpu" for CPU). "mps" is used for Apple Silicon GPUs.
    display : bool, default=True
        Whether to print progress during training.
    folder : bool, default=True
        Whether to clean the folder specified for saving generated images before starting training.

    Attributes
    ----------
    net_G : Generator
        The generator model.
    net_D : Discriminator
        The discriminator model.
    dataloader_path : str
        The path to the dataloader object, used to load training data.
    criterion : torch.nn.Module
        The loss function used for training.
    optimizer_G : torch.optim.Optimizer
        The optimizer for the generator.
    optimizer_D : torch.optim.Optimizer
        The optimizer for the discriminator.

    Methods
    -------
    train():
        Runs the training loop for the specified number of epochs, updating the generator and discriminator
        at each step and optionally saving checkpoints and generating images to visualize progress.

    Notes
    -----
    - The training loop alternates between updating the discriminator, based on its ability to distinguish real
      from generated images, and updating the generator, to improve its ability to create realistic images.
    - Model checkpoints are saved periodically, and the best model can be saved at the end of training.

    Examples
    --------
    >>> trainer = Trainer(image_size=64, input_channels=3, latent_space=100, epochs=10, device='cuda')
    >>> trainer.train()
    This will train the GAN models for 10 epochs, periodically saving checkpoints and generating images to
    visualize the generator's progress.

    Raises
    ------
    FileNotFoundError
        If the dataloader object cannot be found at the specified path.
    Exception
        If an error occurs during the training process.
    """

    def __init__(
        self,
        image_size=64,
        input_channels=3,
        latent_space=50,
        lr=0.0002,
        epochs=100,
        beta1=0.5,
        beta2=0.999,
        device="mps",
        display=True,
        folder=True,
    ):
        self.image_size = image_size
        self.input_channels = input_channels
        self.latent_space = latent_space
        self.lr = lr
        self.epochs = epochs
        self.beta1 = beta1
        self.beta2 = beta2
        self.display = display

        self.steps = 50

        self.device = device_init(device=device)
        self.net_G, self.net_D = self.model_init()

        try:
            self.net_G.apply(weight_init)
            self.net_D.apply(weight_init)
        except Exception as e:
            print("Exception caught in the section # {}".format(e))
        else:
            self.criterion = nn.MSELoss()
            self.optimizer_G = optim.Adam(
                params=self.net_G.parameters(),
                lr=self.lr,
                betas=(self.beta1, self.beta2),
            )
            self.optimizer_D = optim.Adam(
                params=self.net_D.parameters(),
                lr=self.lr,
                betas=(self.beta1, self.beta2),
            )
        finally:
            clean_folder(clean=folder)
            self.dataloader_path = PROCESSED_PATH

    def model_init(self):
        """
        Initializes the Generator and Discriminator models and moves them to the specified device.
        """
        generator = Generator(
            latent_space=self.latent_space,
            image_size=self.image_size,
            in_channels=self.input_channels,
        ).to(self.device)

        discriminator = Discriminator(
            image_size=self.image_size, input_channels=self.input_channels
        ).to(self.device)

        return generator, discriminator

    def load_data(self):
        """
        Loads the training data from the dataloader object.
        """
        dataloader = os.path.join(self.dataloader_path, "dataloader.pkl")
        if os.path.exists(dataloader):
            return pickle.load(dataloader)
        else:
            raise Exception("DataLoader not found".capitalize())

    def train_discriminator(self, **kwargs):
        """
        Trains the discriminator model using both real and generated images.
        """
        self.optimizer_D.zero_grad()

        real_loss = self.criterion(
            self.net_D(kwargs["real_images"]), kwargs["real_labels"]
        )
        fake_loss = self.criterion(
            self.net_D(kwargs["fake_images"].detach()), kwargs["fake_labels"]
        )
        total_loss = 0.5 * (real_loss + fake_loss)

        total_loss.backward()
        self.optimizer_D.step()

        return total_loss.item()

    def train_generator(self, **kwargs):
        """
        Trains the generator model to improve its ability to generate realistic images.
        """
        self.optimizer_G.zero_grad()

        generated_loss = 0.5 * self.criterion(
            self.net_D(self.net_G(kwargs["noise_samples"])), kwargs["real_labels"]
        )

        generated_loss.backward()
        self.optimizer_G.step()

        return generated_loss.item()

    def saved_checkpoints(self, epoch):
        """
        Saves model checkpoints during training.
        """
        if os.path.exists(CHECKPOINT_PATH) and os.path.exists(BEST_MODEL_PATH):
            logging.info("Checkpoints found".capitalize())

            if epoch != self.epochs:
                torch.save(
                    self.net_G.state_dict(),
                    os.path.join(CHECKPOINT_PATH, "model_{}.pth".format(epoch)),
                )
            else:
                torch.save(
                    self.net_G.state_dict(),
                    os.path.join(BEST_MODEL_PATH, "best_model.pth"),
                )

            logging.info("Saved model checkpoint".capitalize())
        else:
            os.makedirs(CHECKPOINT_PATH)
            os.makedirs(BEST_MODEL_PATH)
            logging.info("Created checkpoint directory".capitalize())

    def display_progress(self, **kwargs):
        """
        Optionally prints training progress information.
        """
        print(
            f"Epoch [{kwargs['epoch']}/{kwargs['num_epochs']}], Step [{kwargs['i']}/{len(self.load_data())}], "
            f"Generator Loss: {kwargs['g_loss']:.4f}, Discriminator Loss: {kwargs['d_loss']:.4f}"
        )

    def train(self):
        """
        Runs the training loop, updating the generator and discriminator models based on the loss from their
        predictions on real and generated images.
        ""
        dataloader = self.load_data()

        for epoch in range(self.epochs):
            d_loss = []
            g_loss = []
            for index, (real_images, _) in enumerate(dataloader):
                real_images = real_images.to(self.device)
                batch_size = real_images.shape[0]

                real_labels = torch.ones(batch_size, 1).to(self.device)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)

                noise_samples = torch.randn(batch_size, self.latent_space, 1, 1).to(
                    self.device
                )
                fake_images = self.net_G(noise_samples)

                G_loss = self.train_discriminator(
                    real_images=real_images,
                    fake_images=fake_images,
                    real_labels=real_labels,
                    fake_labels=fake_labels,
                )
                D_loss = self.train_generator(
                    noise_samples=noise_samples, real_labels=real_labels
                )

                try:
                    d_loss.append(D_loss)
                    g_loss.append(G_loss)

                except Exception as e:
                    logging.info(f"Error at epoch {epoch} and batch {index}: {e}")
                else:
                    if self.display:
                        if (index + 1) % self.steps == 0:
                            self.display_progress(
                                d_loss=D_loss,
                                g_loss=G_loss,
                                epoch=epoch + 1,
                                index=index,
                                num_epochs=self.epochs,
                                i=index,
                            )

            print("[Epochs - {}/{} ] is Completed....".format(epoch + 1, self.epochs))
            print(
                "[===============] D_loss: {} - G_loss: {}".format(
                    np.array(d_loss).mean(), np.array(g_loss).mean()
                )
            )
            try:
                if os.path.exists(GEN_PATH):
                    file_path = os.path.join(
                        GEN_PATH, "images_{}.png".format(epoch + 1)
                    )
            except Exception as e:
                logging.info(
                    f"Error at epoch {epoch} and batch {index}: {e}".capitalize()
                )
                os.makedirs(GEN_PATH)
                file_path = os.path.join(GEN_PATH, "images_{}.png".format(epoch + 1))
            else:
                nose_samples = torch.randn(8, self.latent_space, 1, 1).to(self.device)
                fake_images = self.net_G(nose_samples)
                save_image(fake_images, file_path, nrow=8, normalize=True)
                self.saved_checkpoints(epoch=epoch + 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model training".title())

    parser.add_argument(
        "--image_size", type=int, default=64, help="Image size".capitalize()
    )
    parser.add_argument(
        "--input_channels", type=int, default=3, help="Input channels".capitalize()
    )
    parser.add_argument(
        "--latent_space", type=int, default=100, help="Latent space".capitalize()
    )
    parser.add_argument(
        "--lr", type=float, default=0.0002, help="Learning rate".capitalize()
    )
    parser.add_argument("--epochs", type=int, default=1, help="Epochs".capitalize())
    parser.add_argument("--device", type=str, default="mps", help="Device".capitalize())
    parser.add_argument(
        "--folder", action="store_true", help="Clean the folder".capitalize()
    )

    args = parser.parse_args()

    if args.folder:
        if (
            args.image_size
            and args.input_channels
            and args.latent_space
            and args.lr
            and args.epochs
            and args.device
        ):
            logging.info("Cleaning the folder...".capitalize())

            trainer = Trainer(
                image_size=args.image_size,
                input_channels=args.input_channels,
                latent_space=args.latent_space,
                lr=args.lr,
                epochs=args.epochs,
                device=args.device,
            )
            trainer.train()

            logging.info("Training completed successfully".capitalize())

        else:
            logging.exception("All parameters are not provided".capitalize())
