from builtins import breakpoint
import math
from random import sample
import torch
import torch.nn as nn
import torch.nn.functional as F

# implementation inspired from
# https://github.com/nitarshan/bayes-by-backprop/blob/master/Weight%20Uncertainty%20in%20Neural%20Networks.ipynb
# but changed for classification to regression


class Gaussian(object):
    def __init__(self, mu, rho):
        super().__init__()
        self.mu = mu
        self.rho = rho
        self.normal = torch.distributions.Normal(0, 1)
        self.device = "cuda:0"

    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho))

    def sample(self):
        epsilon = self.normal.sample(self.rho.size()).to(self.device)
        return self.mu + self.sigma * epsilon

    def log_prob(self, input):
        return (
            -math.log(math.sqrt(2 * math.pi))
            - torch.log(self.sigma)
            - ((input - self.mu) ** 2) / (2 * self.sigma**2)
        ).sum()


class ScaleMixtureGaussian(object):
    def __init__(self, pi, sigma1, sigma2):
        super().__init__()
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.gaussian1 = torch.distributions.Normal(0, sigma1)
        self.gaussian2 = torch.distributions.Normal(0, sigma2)

    def log_prob(self, input):
        prob1 = torch.exp(self.gaussian1.log_prob(input))
        prob2 = torch.exp(self.gaussian2.log_prob(input))
        return (torch.log(self.pi * prob1 + (1 - self.pi) * prob2)).sum()


class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        # hyper parameters
        PI = 0.5
        SIGMA_1 = torch.cuda.FloatTensor([math.exp(-0)])
        SIGMA_2 = torch.cuda.FloatTensor([math.exp(-6)])

        self.in_features = in_features
        self.out_features = out_features
        # Weight parameters
        self.weight_mu = nn.Parameter(
            torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2)
        )
        self.weight_rho = nn.Parameter(
            torch.Tensor(out_features, in_features).uniform_(-5, -4)
        )
        self.weight = Gaussian(self.weight_mu, self.weight_rho)
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5, -4))
        self.bias = Gaussian(self.bias_mu, self.bias_rho)
        # Prior distributions
        self.weight_prior = Gaussian(PI, SIGMA_1, SIGMA_2)
        self.bias_prior = Gaussian(PI, SIGMA_1, SIGMA_2)
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, input, sample=False, calculate_log_probs=False):
        
        if self.training or sample:
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            weight = self.weight.mu
            bias = self.bias.mu
        if self.training or calculate_log_probs:
            self.log_prior = self.weight_prior.log_prob(
                weight
            ) + self.bias_prior.log_prob(bias)
            self.log_variational_posterior = self.weight.log_prob(
                weight
            ) + self.bias.log_prob(bias)
        else:
            self.log_prior, self.log_variational_posterior = 0, 0

        return F.linear(input, weight, bias)


class BayesianAE(nn.Module):
    def __init__(self, latent_size):
        super().__init__()

        self.e1 = BayesianLinear(784, 512)
        self.e2 = BayesianLinear(512, 256)
        self.e3 = BayesianLinear(256, 128)
        self.e4 = BayesianLinear(128, latent_size)

        self.d1 = BayesianLinear(latent_size, 128)
        self.d2 = BayesianLinear(128, 256)
        self.d3 = BayesianLinear(256, 512)
        self.d4 = BayesianLinear(512, 784)

        self.latent_size = latent_size

    def forward(self, x, sample=False):

        x = torch.tanh(self.e1(x, sample))
        x = torch.tanh(self.e2(x, sample))
        x = torch.tanh(self.e3(x, sample))

        z = self.e4(x, sample)
        x = torch.tanh(z)

        x = torch.tanh(self.d1(x, sample))
        x = torch.tanh(self.d2(x, sample))
        x = torch.tanh(self.d3(x, sample))
        x = self.d4(x, sample)

        return x, z

    def log_prior(self):
        return (
            self.e1.log_prior
            + self.e2.log_prior
            + self.e3.log_prior
            + self.e4.log_prior
            + self.d1.log_prior
            + self.d2.log_prior
            + self.d3.log_prior
            + self.d4.log_prior
        )

    def log_variational_posterior(self):
        return (
            self.e1.log_variational_posterior
            + self.e2.log_variational_posterior
            + self.e3.log_variational_posterior
            + self.e4.log_variational_posterior
            + self.d1.log_variational_posterior
            + self.d2.log_variational_posterior
            + self.d3.log_variational_posterior
            + self.d4.log_variational_posterior
        )

    def sample_decoder(self, z, samples=1):

        x_rec = None
        x_rec2 = None
        for i in range(samples):
            x = torch.tanh(z)
            x = torch.tanh(self.d1(x, True))
            x = torch.tanh(self.d2(x, True))
            x = torch.tanh(self.d3(x, True))
            x = torch.tanh(self.d4(x, True))

            if x_rec is None:
                x_rec = x
                x_rec2 = x**2
            else:
                x_rec += x
                x_rec2 += x**2

        x_rec_mu = x_rec / samples
        x_rec_var = x_rec2 / samples - x_rec_mu**2

        return x_rec_mu, x_rec_var

    def sample_elbo(self, input, target, kl_weight, samples=1):
        bs, outsize = input.shape

        # allocate memory
        x_rec = None
        x_rec2 = None
        z = None
        z2 = None
        log_priors = None
        log_variational_posteriors = None

        # sample NN
        for i in range(samples):
            outputs_i, z_i = self(input, sample=True)
            log_priors_i = self.log_prior()
            log_variational_posteriors_i = self.log_variational_posterior()

            if z is None:
                z = z_i
                z2 = z_i**2
                x_rec = outputs_i
                x_rec2 = outputs_i**2
                log_priors = log_priors_i
                log_variational_posteriors = log_variational_posteriors_i
            else:
                z += z_i
                z2 += z_i**2
                x_rec += outputs_i
                x_rec2 += outputs_i**2
                log_priors += log_priors_i
                log_variational_posteriors += log_variational_posteriors_i

        log_prior = log_priors / samples
        log_variational_posterior = log_variational_posteriors / samples

        mu_x_hat = x_rec / samples
        sigma_x_hat = (x_rec2 / samples - mu_x_hat**2) ** 0.5
        mu_z_hat = z / samples
        sigma_z_hat = (z2 / samples - mu_z_hat**2) ** 0.5
        negative_log_likelihood = F.mse_loss(mu_x_hat, target, reduction="mean")

        loss = (
            log_variational_posterior - log_prior
        ) * kl_weight + negative_log_likelihood

        return (
            loss,
            log_prior,
            log_variational_posterior,
            negative_log_likelihood,
            mu_x_hat,
            sigma_x_hat,
            mu_z_hat,
            sigma_z_hat,
        )
