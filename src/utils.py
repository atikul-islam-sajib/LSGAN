import os
import joblib as pickle
import torch
import torch.nn as nn
from config import CHECKPOINT_PATH, BEST_MODEL_PATH


def create_pickle(data=None, filename=None):
    """
    Saves provided data to a specified file using joblib's pickle functionality.

    Parameters
    ----------
    data : any serializable object, optional
        The data to be saved. If None, the function raises a ValueError.
    filename : str, optional
        The path and filename where the data should be saved. If None, the function raises a ValueError.

    Raises
    ------
    ValueError
        If either `data` or `filename` is not provided.
    """
    if data is not None and filename is not None:
        pickle.dump(value=data, filename=filename)
    else:
        raise ValueError("No data provided".capitalize())


def total_params(model=None):
    """
    Calculates the total number of parameters in a given PyTorch model.

    Parameters
    ----------
    model : torch.nn.Module, optional
        The model from which to count parameters. If None, the function raises an Exception.

    Returns
    -------
    int
        The total number of parameters in the model.

    Raises
    ------
    Exception
        If the `model` is not provided.
    """
    if model is not None:
        return sum(params.numel() for params in model.parameters())
    else:
        raise Exception("Model is empty".capitalize())


def model_info(model=None):
    """
    Generates and yields information about each layer in a given PyTorch model, including the layer's name and the number of parameters it contains.

    Parameters
    ----------
    model : torch.nn.Module, optional
        The model to inspect. If None, the function raises a ValueError.

    Yields
    ------
    str
        Information about each layer in the model, formatted as "layer # {layer} and params # {params.numel()}".

    Raises
    ------
    ValueError
        If no model is provided.
    """
    if model is not None:
        return (
            f"layer # {layer} and params # {params.numel()} "
            for layer, params in model.named_parameters()
        )
    else:
        raise ValueError("No model provided".capitalize())


def weight_init(m=None):
    """
    Initializes weights for Convolutional and BatchNorm layers in a PyTorch model using a normal distribution.

    Parameters
    ----------
    m : torch.nn.Module, optional
        The module to initialize. If None, the function raises a ValueError.

    Raises
    ------
    ValueError
        If no module is provided.
    """
    if m is not None:
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
        if classname.find("BatchNorm") != -1:
            nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
            nn.init.constant_(m.bias.data, 0.0)
    else:
        raise ValueError("No model provided".capitalize())


def device_init(device="mps"):
    """
    Initializes and returns the specified PyTorch device based on availability.

    Parameters
    ----------
    device : str, default="mps"
        The preferred device to use ("cuda", "mps", or "cpu").

    Returns
    -------
    torch.device
        The device that will be used for computations.
    """
    if device == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    elif device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device("cpu")


def clean_folder(clean=True):
    """
    Cleans the specified folders by removing all files within them. If the folders do not exist, they are created.

    Parameters
    ----------
    clean : bool, default=True
        Whether to clean the folders specified by `CHECKPOINT_PATH` and `BEST_MODEL_PATH`.

    Notes
    -----
    This function is particularly useful for setting up or resetting the environment before starting a new training session.
    """
    if clean:
        if os.path.exists(CHECKPOINT_PATH) and os.path.exists(BEST_MODEL_PATH):
            for file in os.listdir(CHECKPOINT_PATH):
                os.remove(os.path.join(CHECKPOINT_PATH, file))

            for file in os.listdir(BEST_MODEL_PATH):
                os.remove(os.path.join(BEST_MODEL_PATH, file))
        else:
            os.makedirs(CHECKPOINT_PATH)
            os.makedirs(BEST_MODEL_PATH)
