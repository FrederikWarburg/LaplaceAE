from abc import abstractmethod
from builtins import breakpoint
from multiprocessing import reduction

import torch
import sys

sys.path.append("../../backpack")

from backpack import backpack, extend, memory_cleanup
from backpack.extensions import (
    DiagGGNExact,
    DiagGGNMC,
    KFAC,
    KFLR,
    SumGradSquared,
    BatchGrad,
)
from backpack.context import CTX


class HessianCalculator:
    def __init__(self):
        super(HessianCalculator, self).__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    @abstractmethod
    def compute_batch(self, *args, **kwargs):
        pass

    def compute(self, loader):
        hessian = None
        for batch in loader:
            Hs = self.compute_batch(*batch)
            if hessian is None:
                hessian = Hs
            else:
                hessian += Hs
        return hessian.to("cpu")


class MseHessianCalculator(HessianCalculator):
    def __init__(self, model=None):
        super(MseHessianCalculator, self).__init__()
        self.factor = 0.5  # for regression
        self.stochastic = False
        self.context = DiagGGNMC if self.stochastic else DiagGGNExact
        self.lossfunc = torch.nn.MSELoss(reduction="sum")
        self.lossfunc = extend(self.lossfunc)
        if model is not None:
            self.model = extend(model)

    def compute_batch(self, x, *args, **kwargs):
        x = x.to(self.device)
        loss, dggn = self.diag(x, x)

        return dggn

    def __call__(self, net, feature_maps, X, **kwargs):
        b = X.shape[0]
        y_hat = net(X)
        loss = self.lossfunc(y_hat.view(b, -1), X.view(b, -1))
        with backpack(self.context()):
            loss.backward()
        loss = loss.detach()
        return self.factor * self._get_diag_ggn(net).detach()

    def diag(self, X, y, **kwargs):
        b = X.shape[0]

        f = self.model(X)
        loss = self.lossfunc(f.view(b, -1), y.view(b, -1))
        with backpack(self.context()):
            loss.backward()
        dggn = self._get_diag_ggn(self.model)

        return self.factor * loss.detach(), self.factor * dggn

    def _get_diag_ggn(self, model):
        if self.stochastic:
            return torch.cat([p.diag_ggn_mc.data.flatten() for p in model.parameters()])
        else:
            return torch.cat(
                [p.diag_ggn_exact.data.flatten() for p in model.parameters()]
            )
