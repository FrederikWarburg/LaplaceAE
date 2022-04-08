import copy
from abc import abstractmethod

import torch
from torch.nn.utils import parameters_to_vector


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
    def __init__(self):
        super(MseHessianCalculator, self).__init__() 

    def compute_batch(self, model, output_size, x, *args, **kwargs):
        x = x.to(self.device)
        bs = x.shape[0]

        self.feature_maps = []
        model(x)
        self.feature_maps = [x] + self.feature_maps

        # Saves the product of the Jacobians wrt layer input
        tmp = torch.diag_embed(torch.ones(bs, output_size, device=x.device))

        H = []
        with torch.no_grad():
            for k in range(len(model) - 1, -1, -1):
                if isinstance(model[k], torch.nn.Linear):

                    diag_elements = torch.einsum("bii->bi", tmp)
                    h_k = torch.einsum("bi,bj,bj->bij", diag_elements,
                                       self.feature_maps[k], self.feature_maps[k])

                    h_k = h_k.view(bs, -1)
                    if model[k].bias is not None:
                        h_k = torch.cat([h_k, diag_elements], dim=1)

                    H = [h_k] + H

                if k == 0:
                    break

                # Calculate the Jacobian wrt to the inputs
                if isinstance(model[k], torch.nn.Linear):
                    jacobian_x = model[k].weight.expand(
                        (bs, *model[k].weight.shape))
                elif isinstance(model[k], torch.nn.Tanh):
                    jacobian_x = torch.diag_embed(
                        torch.ones(self.feature_maps[k + 1].shape, device=x.device)
                        - self.feature_maps[k + 1] ** 2
                    )
                elif isinstance(model[k], torch.nn.ReLU):
                    jacobian_x = torch.diag_embed(
                        (self.feature_maps[k + 1] > 0).float()
                    )
                else:
                    raise NotImplementedError

                # TODO: make more efficent by using row vectors
                tmp = torch.einsum("bnm,bnj,bjk->bmk", jacobian_x, tmp,
                                   jacobian_x)

        return torch.cat(H, dim=1).sum(dim=0)