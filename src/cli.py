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

    """
    The script you've provided serves as a command-line interface (CLI) for training and testing a Generative Adversarial Network (GAN) using PyTorch. It allows users to specify parameters for data loading, model training, and synthetic image generation through command-line arguments. Here's a breakdown of its functionality and usage:

    ### Overview
    - The script sets up a command-line parser to accept various arguments related to the dataset, model parameters, and training configuration.
    - It uses the `Loader` class to preprocess the dataset and create a DataLoader.
    - It initializes the `Trainer` class with user-specified or default parameters to train the GAN models.
    - Optionally, it can run a test to generate synthetic images using the trained Generator model, controlled by the `--test` flag.

    ### Key Arguments
    - `--image_path`: Specifies the path to the zip file containing the dataset.
    - `--batch_size`, `--image_size`: Control the size of batches and the dimensions of images during training.
    - `--in_channels`, `--latent_space`: Define the number of input channels (e.g., 3 for RGB images) and the size of the latent space vector.
    - `--num_samples`: Determines the number of synthetic images to generate during the test phase.
    - `--lr`, `--epochs`, `--device`: Set the learning rate, number of training epochs, and the computation device (`cuda`, `mps`, or `cpu`).
    - `--folder`: A flag indicating whether to clean the training and model directories before starting.
    - `--test`: A flag to trigger the generation of synthetic images after training.

    ### Execution Flow
    1. **Data Loading**: If an image path is provided, it unzips the dataset, processes it, and creates a DataLoader.
    2. **Model Training**: If all required arguments for training are provided, it trains the GAN models using the specified parameters.
    3. **Testing**: If the `--test` flag is set, it generates synthetic images using the trained Generator model.

    ### Example Usage
    To train the model with custom settings:
    ```shell
    python script.py --image_path "/path/to/dataset.zip" --batch_size 64 --image_size 64 --in_channels 3 --latent_space 100 --lr 0.0002 --epochs 10 --device cuda --folder
    ```
    To generate synthetic images after training:
    ```shell
    python script.py --test --latent_space 100 --num_samples 20 --device cuda
    ```

    ### Logging
    - The script logs key events (e.g., data loading, training start/end, testing start/end) to a log file at `./logs/cli.log`.

    ### Error Handling
    - It checks for the presence of required arguments and logs information or errors accordingly.
    - It does not explicitly handle exceptions thrown by the underlying classes (`Loader`, `Trainer`, `Test`), which might be an area for improvement.

    This CLI provides a flexible way to work with GANs, allowing for experimentation with different configurations without modifying the codebase.
    """
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
