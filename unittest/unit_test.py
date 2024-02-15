import sys
import os
import unittest
import joblib as pickle

sys.path.append("src/")

from config import PROCESSED_PATH
from discriminator import Discriminator
from generator import Generator
from utils import total_params


class UnitTest(unittest.TestCase):
    """
    A unit test class for testing the functionality and integrity of components in a GAN implementation.

    This class extends unittest.TestCase and includes tests for the dataloader, discriminator, and generator
    to ensure they are initialized correctly and contain the expected number of parameters or dataset items.

    Attributes
    ----------
    dataloader : DataLoader
        The DataLoader object loaded with the dataset for testing. It is initialized in the setUp method
        and used in the tests to verify the quantity of data loaded.

    Methods
    -------
    setUp():
        Prepares the test environment before each test method is executed. Specifically, it loads the
        dataloader with the dataset from a specified path.

    test_quantity_dataset():
        Tests whether the total number of items in the dataset matches the expected quantity.

    test_total_params_discriminator():
        Verifies that the total number of parameters in the Discriminator model matches the expected count.

    test_total_params_generator():
        Checks if the total number of parameters in the Generator model is as expected based on its configuration.

    Notes
    -----
    - The tests assume the presence of a pre-processed dataset saved in a pickle file.
    - The `total_params` utility function is used to calculate the total parameters of the models.
    - The expected values for dataset items and model parameters are hardcoded and may need to be adjusted
      if the models or dataset preprocessing steps change.
    """

    def setUp(self):
        self.dataloader = pickle.load(os.path.join(PROCESSED_PATH, "dataloader.pkl"))

    ###################
    #    Dataloader   #
    ###################

    def test_quantity_dataset(self):
        self.assertEqual(sum(image.shape[0] for image, _ in self.dataloader), 34557)

    ######################
    #    Discriminator   #
    ######################

    def test_total_params_discriminator(self):
        self.assertEqual(total_params(Discriminator()), 2766529)

    ######################
    #      Generator     #
    ######################

    def test_total_params_generator(self):
        self.assertEqual(
            total_params(
                Generator(
                    latent_space=50,
                    image_size=64,
                    in_channels=3,
                )
            ),
            3168067,
        )


if __name__ == "__main__":
    unittest.main()
