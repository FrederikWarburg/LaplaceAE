from hessian import layerwise as lw
from hessian import backpack as bp
from backpack import extend
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import torch
import time
from torch.nn import functional as F

from laplace.laplace import BlockLaplace, DiagLaplace
laplace_methods = {
    "block": BlockLaplace,
    "exact": DiagLaplace,
    "approx": DiagLaplace,
    "mix": DiagLaplace,
}


class OnlineLaplace:
    def __init__(self, net, dataset_size, config):
        super(OnlineLaplace, self).__init__()

        device = (
            "cuda:0" if torch.cuda.is_available() else "cpu"
        ) 

        self.alpha = float(config["alpha"])
        self.net = net.to(device)

        self.prior_prec = torch.tensor(float(config["prior_precision"])).to(device)
        self.hessian_scale = torch.tensor(float(config["hessian_scale"])).to(device)
        self.dataset_size = dataset_size
        self.n_samples = config["train_samples"]
        self.one_hessian_per_sampling = config["one_hessian_per_sampling"]
        self.update_hessian = config["update_hessian"]
        self.hessian_memory_factor = float(config["hessian_memory_factor"])

        self.sigma_n = 1.0
        self.constant = 1.0 / (2 * self.sigma_n**2)

        if config["backend"] == "backpack":
            self.HessianCalculator = bp.MseHessianCalculator()
            self.laplace = DiagLaplace()
            self.net = extend(self.net)

        else:
            self.feature_maps = []

            def fw_hook_get_latent(module, input, output):
                self.feature_maps.append(output.detach())

            for k in range(len(self.net)):
                self.net[k].register_forward_hook(fw_hook_get_latent)

            self.HessianCalculator = lw.MseHessianCalculator(config["approximation"])
            self.laplace = laplace_methods[config["approximation"]]()

        self.hessian = self.laplace.init_hessian(self.dataset_size, self.net, device)

        # logging of time:
        self.timings = {
            "forward_nn": 0,
            "compute_hessian": 0,
            "entire_training_step": 0,
        }

    def elbo(self, x, train=True):
        self.timings["forward_nn"] = 0
        self.timings["compute_hessian"] = 0
        self.timings["entire_training_step"] = time.time()
        
        sigma_q = self.laplace.posterior_scale(
            self.hessian, self.hessian_scale, self.prior_prec
        )
        mu_q = parameters_to_vector(self.net.parameters()).unsqueeze(1)
        regularizer = weight_decay(mu_q, self.prior_prec)

        mse_running_sum = 0
        hessian = []
        x_recs = []

        # draw samples from the nn (sample nn)
        samples = self.laplace.sample(mu_q, sigma_q, self.n_samples)
        for net_sample in samples:

            # replace the network parameters with the sampled parameters
            vector_to_parameters(net_sample, self.net.parameters())

            # reset or init
            self.feature_maps = []

            # predict with the sampled weights
            start = time.time()
            x_rec = self.net(x)

            self.timings["forward_nn"] += time.time() - start

            # compute mse for sample net
            mse_running_sum += F.mse_loss(x_rec.view(*x.shape), x)

            if (not self.one_hessian_per_sampling) and train:
                # compute hessian for sample net
                start = time.time()

                # H = J^T J
                h_s = self.HessianCalculator.__call__(self.net, self.feature_maps, x)
                h_s = self.laplace.scale(h_s, x.shape[0], self.dataset_size)

                self.timings["compute_hessian"] += time.time() - start

                # append results
                hessian.append(h_s)
            x_recs.append(x_rec)

        # reset the network parameters with the mean parameter (MAP estimate parameters)
        vector_to_parameters(mu_q, self.net.parameters())
        mse = mse_running_sum / self.n_samples

        if self.one_hessian_per_sampling and train:

            # reset or init
            self.feature_maps = []
            # predict with the sampled weights
            x_rec = self.net(x)
            # compute hessian for sample net
            start = time.time()

            # H = J^T J
            h_s = self.HessianCalculator.__call__(self.net, self.feature_maps, x)
            hessian = [self.laplace.scale(h_s, x.shape[0], self.dataset_size)]

            self.timings["compute_hessian"] += time.time() - start

        if train:
            # take mean over hessian compute for different sampled NN
            hessian = self.laplace.average_hessian_samples(hessian, self.constant)

            if self.update_hessian:
                self.hessian = (
                    self.hessian_memory_factor * self.hessian + hessian
                )
            else:
                self.hessian = (
                    1 - self.hessian_memory_factor
                ) * hessian + self.hessian_memory_factor * self.hessian

        loss = self.constant * mse + self.alpha * regularizer

        # store some stuff for loggin purposes
        self.mse_loss = mse#.detach()
        self.regularizer_loss = self.alpha * regularizer#.detach()
        self.x_recs = x_recs#.detach()

        return loss

    def sample(self, n_samples = 100):
        sigma_q = self.laplace.posterior_scale(
            self.hessian, self.hessian_scale, self.prior_prec
        )
        mu_q = parameters_to_vector(self.net.parameters()).unsqueeze(1)

        samples = self.laplace.sample(mu_q, sigma_q, n_samples)

        return samples

    def load_hessian(self, path):
        self.h = torch.load(path)

    def save_hessian(self, path):
        torch.save(self.hessian, path)

def weight_decay(mu_q, prior_prec):

    return 0.5 * (torch.matmul(mu_q.T, mu_q) / prior_prec + torch.log(prior_prec))

