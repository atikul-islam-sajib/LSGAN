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
