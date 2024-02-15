import os
import joblib as pickle
import torch
import torch.nn as nn
from config import CHECKPOINT_PATH, BEST_MODEL_PATH


def create_pickle(data=None, filename=None):
    if data is not None and filename is not None:
        pickle.dump(value=data, filename=filename)
    else:
        raise ValueError("No data provided".capitalize())


def total_params(model=None):
    if model is not None:
        return sum(params.numel() for params in model.parameters())
    else:
        raise Exception("Model is empty".capitalize())


def model_info(model=None):
    if model is not None:
        return (
            f"layer # {layer} and params # {params.numel()} "
            for layer, params in model.named_parameters()
        )
    else:
        raise ValueError("No model provided".capitalize())


def weight_init(m=None):
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
    if device == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    elif device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device("cpu")


def clean_folder(clean=True):
    if clean:
        if os.path.exists(CHECKPOINT_PATH) and os.path.exists(BEST_MODEL_PATH):
            for file in os.listdir(CHECKPOINT_PATH):
                os.remove(os.path.join(CHECKPOINT_PATH, file))

            for file in os.listdir(BEST_MODEL_PATH):
                os.remove(os.path.join(BEST_MODEL_PATH, file))
        else:
            os.makedirs(CHECKPOINT_PATH)
            os.makedirs(BEST_MODEL_PATH)
