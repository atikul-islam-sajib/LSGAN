import sys
import argparse
import logging

sys.path.append("src/")

from dataloader import Loader
from generator import Generator
from discriminator import Discriminator
from trainer import Trainer
from test import Test

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)-15s %(message)s",
    filemode="w",
    filename="./logs/cli.log",
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model with CLI".title())

    parser.add_argument(
        "--image_path", type=str, help="Path to the image folder".capitalize()
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch Size".capitalize()
    )
    parser.add_argument(
        "--image_size", type=int, default=64, help="Image Size".capitalize()
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
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Define the number of samples".capitalize(),
    )

    parser.add_argument(
        "--lr", type=float, default=0.0002, help="Learning rate".capitalize()
    )
    parser.add_argument("--epochs", type=int, default=1, help="Epochs".capitalize())
    parser.add_argument("--device", type=str, default="mps", help="Device".capitalize())
    parser.add_argument(
        "--folder", action="store_true", help="Clean the folder".capitalize()
    )
    parser.add_argument(
        "--test", action="store_true", help="Creating the synthetic images".capitalize()
    )

    args = parser.parse_args()

    if args.image_path:
        if (
            args.image_size
            and args.batch_size
            and args.in_channels
            and args.latent_space
            and args.lr
            and args.epochs
            and args.device
            and args.folder
        ):
            logging.info("All the arguments are correct".capitalize())
            loader = Loader(
                image_path=args.image_path,
                batch_size=args.batch_size,
                image_size=args.image_size,
            )
            loader.unzip_folder()
            loader.create_dataloader()

            logging.info("DataLoader created".capitalize())

            trainer = Trainer(
                image_size=args.image_size,
                input_channels=args.in_channels,
                latent_space=args.latent_space,
                lr=args.lr,
                epochs=args.epochs,
                device=args.device,
                folder=args.folder,
            )

            trainer.train()

            logging.info("Training completed".capitalize())

    if args.test:
        if args.latent_space and args.device and args.num_samples:

            logging.info("All the arguments are correct".capitalize())

            test = Test(
                latent_space=args.latent_space,
                num_samples=args.num_samples,
                device=args.device,
            )
            test.test()

            logging.info("Test completed".capitalize())
