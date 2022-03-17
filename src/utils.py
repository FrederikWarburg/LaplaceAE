import torch.nn.functional as F
import dill


def softclip(tensor, min):
    """ Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials """
    result_tensor = min + F.softplus(tensor - min)

    return result_tensor


def save_laplace(la, filepath):
    with open(filepath, 'wb') as outpt:
        dill.dump(la, outpt)


def load_laplace(filepath):
    with open(filepath, 'rb') as inpt:
        la = dill.load(inpt)
    return la