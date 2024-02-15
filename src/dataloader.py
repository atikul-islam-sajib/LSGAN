import sys
import logging
import argparse
import os
import zipfile
import joblib as pickle
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)-15s %(message)s",
    filemode="w",
    filename="./logs/dataloader.log",
)

sys.path.append("src/")

from utils import create_pickle
from config import PROCESSED_PATH, RAW_PATH


class Loader:
    """
    A data loader class for preprocessing and loading image datasets for deep learning models.

    This class handles the unzipping of image datasets, normalization, and creation of PyTorch dataloaders
    for training deep learning models. It requires the dataset to be in a zip file format and will process
    the images to the specified size and batch size for training.

    Parameters
    ----------
    image_path : str, default=None
        The path to the zip file containing the image dataset.
    batch_size : int, default=64
        The number of images to process in each batch.
    image_size : int, default=64
        The size (width and height) to which each image will be resized.

    Attributes
    ----------
    to_extract : str
        The path where the dataset will be extracted. It is defined by the RAW_PATH variable from the config.
    to_processed : str
        The path where the processed data will be saved. It is defined by the PROCESSED_PATH variable from the config.

    Methods
    -------
    unzip_folder():
        Extracts the dataset from the zip file to the specified path.
    create_dataloader():
        Creates a PyTorch DataLoader after preprocessing the dataset.

    Notes
    -----
    This class is designed to be used within a specific project structure and requires a configuration
    file (`config.py`) that specifies `RAW_PATH` and `PROCESSED_PATH` variables for dataset management.
    It utilizes torchvision transforms for image preprocessing.

    Examples
    --------
    >>> from loader import Loader
    >>> # Assuming 'dataset.zip' is your dataset and it's located in the current directory.
    >>> loader = Loader(image_path='dataset.zip', batch_size=32, image_size=128)
    >>> loader.unzip_folder()
    >>> loader.create_dataloader()
    >>> # Now, the DataLoader is ready and saved as 'dataloader.pkl' in the processed data path.
    """

    def __init__(self, image_path=None, batch_size=64, image_size=64):
        self.image_path = image_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.to_extract = RAW_PATH
        self.to_processed = PROCESSED_PATH

    def unzip_folder(self):
        """
        Extracts the dataset from the zip file to the specified path.
        If the extraction path does not exist, it creates the directory.
        """
        with zipfile.ZipFile(self.image_path, "r") as zip_ref:
            if os.path.exists(self.to_extract):
                zip_ref.extractall(path=self.to_extract)
            else:
                print("There is no path to extract the dataset".capitalize())
                os.makedirs(self.to_extract)
                zip_ref.extractall(path=self.to_extract)

    def _do_normalization(self, transform=False):
        """
        Internal method to apply normalization and other transformations to the dataset.
        """
        if transform:
            return transforms.Compose(
                [
                    transforms.Resize((self.image_size)),
                    transforms.CenterCrop(self.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        else:
            raise Exception("Error while doing the normalization".capitalize())

    def create_dataloader(self):
        """
        Processes the images in the dataset and creates a PyTorch DataLoader.
        The DataLoader is then pickled and saved to the processed data path.
        If the processed data path does not exist, it creates the directory before saving.
        """
        if os.path.exists(self.to_extract):
            dataset = ImageFolder(
                root=self.to_extract, transform=self._do_normalization(transform=True)
            )
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            try:
                if os.path.exists(self.to_processed):
                    create_pickle(
                        data=dataloader,
                        filename=os.path.join(self.to_processed, "dataloader.pkl"),
                    )
                else:
                    print(
                        "There is no data path named processed & creating the data path...".capitalize()
                    )
                    os.makedirs(self.to_processed)
                    create_pickle(
                        data=dataloader,
                        filename=os.path.join(self.to_processed, "dataloader.pkl"),
                    )

            except Exception as e:
                print("Error caught in the section # {}".format(e))
        else:
            raise Exception("There is no path to create the dataloader".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Data Loader".title())
    parser.add_argument(
        "--image_path", type=str, help="Path to the image folder".capitalize()
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch Size".capitalize()
    )
    parser.add_argument(
        "--image_size", type=int, default=64, help="Image Size".capitalize()
    )

    args = parser.parse_args()

    if args.image_path and args.batch_size and args.image_size:
        logging.info("Creating Data Loader...".capitalize())

        loader = Loader(
            image_path=args.image_path,
            batch_size=args.batch_size,
            image_size=args.image_size,
        )
        loader.unzip_folder()
        loader.create_dataloader()

        logging.info("Data Loader Created Successfully...".capitalize())
    else:
        logging.exception("Please provide all the required arguments".capitalize())
