from builtins import breakpoint
import copy
from abc import abstractmethod

import torch
from torch.nn.utils import parameters_to_vector

import sys
sys.path.append("../stochman")
from stochman import nnj

class HessianCalculator:
    def __init__(self):
        super(HessianCalculator, self).__init__() 
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    @abstractmethod
    def compute_batch(self, *args, **kwargs):
        pass

    def compute(self, loader, model, output_size):
        # keep track of running sum
        H_running_sum = torch.zeros_like(parameters_to_vector(model.parameters()))
        counter = 0

        self.feature_maps = []

        def fw_hook_get_latent(module, input, output):
            self.feature_maps.append(output.detach())

        for k in range(len(model)):
            model[k].register_forward_hook(fw_hook_get_latent)

        for batch in loader:
            H = self.compute_batch(model, output_size, *batch)
            H_running_sum += H

        return H_running_sum


class MseHessianCalculator(HessianCalculator):
    def __init__(self, diag):
        super(MseHessianCalculator, self).__init__() 

        self.diag = diag

    def compute_batch(self, net, output_size, x, *args, **kwargs):
        x = x.to(self.device)
        H = []

        self.feature_maps = []
        net(x)

        if x.ndim == 4:
            bs, c, h, w = x.shape
            output_size = c*h*w
        else:
            bs, output_size = x.shape

        self.feature_maps = [x] + self.feature_maps
        tmp = torch.diag_embed(torch.ones(bs, output_size, device=x.device))

        if self.diag:
            tmp = torch.diagonal(tmp, dim1=1, dim2=2)

        with torch.no_grad():
            for k in range(len(net) - 1, -1, -1):
                
                if isinstance(net[k], nnj.Reshape):
                    self.diag = False
                    tmp = torch.diag_embed(tmp, dim1=1, dim2=2)
                elif isinstance(net[k], nnj.Flatten):
                    self.diag = True
                    tmp = torch.diagonal(tmp, dim1=1, dim2=2)
                
                # jacobian w.r.t weight
                h_k = net[k]._jacobian_wrt_weight_sandwich(
                    self.feature_maps[k],
                    self.feature_maps[k + 1],
                    tmp,
                    diag=self.diag,
                )
                if h_k is not None:
                    H = [h_k.sum(dim=0)] + H
                
                # jacobian w.r.t input
                tmp = net[k]._jacobian_wrt_input_sandwich(
                    self.feature_maps[k],
                    self.feature_maps[k + 1],
                    tmp,
                    diag=self.diag,
                )
        H = torch.cat(H, dim=0)

        return H