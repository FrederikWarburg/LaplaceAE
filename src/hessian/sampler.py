from builtins import breakpoint
import torch
from abc import abstractmethod
from torch.nn.utils import parameters_to_vector


class Sampler:
    def __init__(self):
        super(Sampler, self).__init__()

    @abstractmethod
    def sample(self, *args, **kwargs):
        pass


class DiagSampler(Sampler):
    def sample(self, parameters, posterior_scale, n_samples=100):
        n_params = len(parameters)
        samples = torch.randn(n_samples, n_params, device=parameters.device)
        samples = samples * posterior_scale.reshape(1, n_params)
        return parameters.reshape(1, n_params) + samples

    def invert(self, hessian):
        return 1.0 / (hessian + 1e-6)

    def kl_div(self, mu_q, sigma_q):
        k = len(mu_q)

        return 0.5 * (
            torch.log(1.0 / sigma_q)
            - k
            + torch.matmul(mu_q.T, mu_q)
            + torch.sum(sigma_q)
        )

    def init_hessian(self, data_size, net, device):

        hessian = data_size * torch.ones_like(
            parameters_to_vector(net.parameters()), device=device
        )
        return hessian

    def scale(self, h_s, b, data_size):
        return h_s / b * data_size

    def aveage_hessian_samples(self, hessian, constant):
        hessian = torch.stack(hessian).mean(dim=0) if len(hessian) > 1 else hessian[0]
        return constant * hessian + 1


class BlockSampler(Sampler):
    def sample(self, parameters, posterior_scale, n_samples=100):
        n_samples = torch.tensor([n_samples])
        count = 0
        param_samples = []
        for post_scale_layer in posterior_scale:
            n_param_layer = len(post_scale_layer)

            layer_param = parameters[count : count + n_param_layer]
            normal = torch.distributions.multivariate_normal.MultivariateNormal(
                layer_param, covariance_matrix=post_scale_layer
            )
            samples = normal.sample(n_samples)
            param_samples.append(samples)

            count += n_param_layer

        param_samples = torch.cat(param_samples, dim=1).to(parameters.device)
        return param_samples

    def invert(self, hessian):
        posterior_scale = [torch.cholesky_inverse(h) for h in hessian]
        return posterior_scale

    def kl_div(self, mu_q, sigma_q):

        k = len(mu_q)

        # TODO: fix later.
        sigma_q = torch.cat([torch.diagonal(s) for s in sigma_q])

        return 0.5 * (
            torch.log(1.0 / sigma_q)
            - k
            + torch.matmul(mu_q.T, mu_q)
            + torch.sum(sigma_q)
        )

    def init_hessian(self, data_size, net, device):

        hessian = []
        for layer in net:
            # if parametric layer
            if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
                params = parameters_to_vector(layer.parameters())
                n_params = len(params)
                hessian.append(
                    data_size * torch.ones(n_params, n_params, device=device)
                )

        return hessian

    def scale(self, h_s, b, data_size):
        return [h / b * data_size for h in h_s]

    def aveage_hessian_samples(self, hessian, constant):
        n_samples = len(hessian)
        n_layers = len(hessian[0])
        hessian_mean = []
        for i in range(n_layers):
            tmp = None
            for s in range(n_samples):
                if tmp is None:
                    tmp = hessian[s][i]
                else:
                    tmp += hessian[s][i]

            tmp = tmp / n_samples
            tmp = constant * tmp + torch.diag_embed(
                torch.ones(len(tmp), device=tmp.device)
            )
            hessian_mean.append(tmp)

        return hessian_mean
