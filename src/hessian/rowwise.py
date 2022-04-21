from abc import abstractmethod

import torch
from asdfghjkl import batch_gradient

# from laplace.curvature.asdl import _get_batch_grad
import time


class HessianCalculator:
    def __init__(self):
        super(HessianCalculator, self).__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    @abstractmethod
    def compute_batch(self, *args, **kwargs):
        pass

    def compute(self, loader, model, output_size):
        hessian = None
        for batch in loader:
            Hs = self.compute_batch(model, output_size, *batch)
            if hessian is None:
                hessian = Hs
            else:
                hessian += Hs
        return hessian


class MseHessianCalculator(HessianCalculator):
    def __init__(self, hessian_structure):
        super(MseHessianCalculator, self).__init__()
        self.hessian_structure = hessian_structure

    def compute_batch(self, model, output_size, x, *args, **kwargs):
        x = x.to(self.device)

        Js, f = jacobians(x, model, output_size=output_size)
        if self.hessian_structure == "diag":
            Hs = torch.einsum("nij,nij->nj", Js, Js)
        elif self.hessian_structure == "full":
            Hs = torch.einsum("nij,nkl->njl", Js, Js)
        else:
            raise NotImplementedError

        return Hs.sum(0)


def jacobians(x, model, output_size=784):
    """Compute Jacobians \\(\\nabla_\\theta f(x;\\theta)\\)
       at current parameter \\(\\theta\\)
    using asdfghjkl's gradient per output dimension.
    Parameters
    ----------
    x : torch.Tensor
        input data `(batch, input_shape)` on compatible device with
        model.
    Returns
    -------
    Js : torch.Tensor
        Jacobians `(batch, parameters, outputs)`
    f : torch.Tensor
        output function `(batch, outputs)`
    """
    jacobians = list()
    f = None
    for i in range(output_size):

        def loss_fn(outputs, targets):
            return outputs[:, i].sum()

        f = batch_gradient(model, loss_fn, x, None).detach()
        jacobian_i = _get_batch_grad(model)

        jacobians.append(jacobian_i)
    jacobians = torch.stack(jacobians, dim=1)
    return jacobians, f


def _get_batch_grad(model):
    batch_grads = list()
    for module in model.modules():
        if hasattr(module, "op_results"):
            res = module.op_results["batch_grads"]
            if "weight" in res:
                batch_grads.append(_flatten_after_batch(res["weight"]))
            if "bias" in res:
                batch_grads.append(_flatten_after_batch(res["bias"]))
            if len(set(res.keys()) - {"weight", "bias"}) > 0:
                raise ValueError(f"Invalid parameter keys {res.keys()}")
    return torch.cat(batch_grads, dim=1)


def _flatten_after_batch(tensor: torch.Tensor):
    if tensor.ndim == 1:
        return tensor.unsqueeze(-1)
    else:
        return tensor.flatten(start_dim=1)
