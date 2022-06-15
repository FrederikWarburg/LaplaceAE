from hessian import layerwise as lw
from torch.nn.utils import parameters_to_vector
import torch
from tqdm import tqdm
from torch.nn import functional as F

from laplace.laplace import BlockLaplace, DiagLaplace
laplace_methods = {
    "block": BlockLaplace,
    "exact": DiagLaplace,
    "approx": DiagLaplace,
    "mix": DiagLaplace,
}


def log_det_ratio(hessian, prior_prec):
    posterior_precision = hessian + prior_prec
    log_det_prior_precision = len(hessian) * prior_prec.log()
    log_det_posterior_precision = posterior_precision.log().sum()
    return log_det_posterior_precision - log_det_prior_precision


def scatter(mu_q, prior_precision_diag):
    return (mu_q * prior_precision_diag) @ mu_q


def log_marginal_likelihood(mu_q, hessian, prior_prec):
    # we ignore neg log likelihood as it is constant wrt prior_prec
    neg_log_marglik = -0.5 * (
        log_det_ratio(hessian, prior_prec) + scatter(mu_q, prior_prec)
    )
    return neg_log_marglik


def optimize_prior_precision(mu_q, hessian, prior_prec, n_steps=100):

    log_prior_prec = prior_prec.log()
    log_prior_prec.requires_grad = True
    optimizer = torch.optim.Adam([log_prior_prec], lr=1e-1)
    for _ in range(n_steps):
        optimizer.zero_grad()
        prior_prec = log_prior_prec.exp()
        neg_log_marglik = -log_marginal_likelihood(mu_q, hessian, prior_prec)
        neg_log_marglik.backward()
        optimizer.step()

    prior_prec = log_prior_prec.detach().exp()

    return prior_prec
    

class PosthocLaplace:
    def __init__(self, net, approx):
        super(PosthocLaplace, self).__init__()

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.feature_maps = []
        self.net = net

        def fw_hook_get_latent(module, input, output):
            self.feature_maps.append(output.detach())

        for k in range(len(net)):
            self.net[k].register_forward_hook(fw_hook_get_latent)

        self.HessianCalculator = lw.MseHessianCalculator(approx)

    def fit(self, train_loader):

        hessian = None
        for X, y in tqdm(train_loader):
            X = X.to(self.device)
            with torch.inference_mode():
                self.feature_maps = []
                x_rec = self.net(X)
            h_s = self.HessianCalculator.__call__(self.net, self.feature_maps, x_rec)

            if hessian is None:
                hessian = h_s
            else:
                hessian += h_s

    def optimize_precision(self):
        mu_q = parameters_to_vector(self.net.parameters())
        self.prior_prec = optimize_prior_precision(mu_q, self.hessian, torch.tensor(1))
        