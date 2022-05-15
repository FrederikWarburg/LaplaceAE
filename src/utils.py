from builtins import breakpoint
import torch.nn.functional as F
import dill
import numpy as np


def softclip(tensor, min):
    """Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials"""
    result_tensor = min + F.softplus(tensor - min)
    return result_tensor


def save_laplace(la, filepath):
    with open(filepath, "wb") as outpt:
        dill.dump(la, outpt)


def load_laplace(filepath):
    with open(filepath, "rb") as inpt:
        la = dill.load(inpt)
    return la


def create_exp_name(config):

    name = config["exp_name"]

    for key in [
        "backend",
        "approximation",
        "no_conv",
        "train_samples",
        "dropout_rate",
        "use_var_decoder",
    ]:
        if key in config:
            name += f"[{key}_{config[key]}]_"

    return name


def compute_typicality_score(train_log_likelihood, test_example_log_like):

    log_like_mean = train_log_likelihood.mean()
    typicality_score = np.linalg.norm(log_like_mean - test_example_log_like, axis=1)

    return typicality_score.reshape(-1, 1)
