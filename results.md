# Results

We have conducted experiments for the following models. 

Models:
- AE
- VAE
- AE dropout
- LAE post hoc
  -- decoder only
  -- entire model
- LAE elbo

For each model, we compute the following metrics and or figures:

Figures:
- Output space:
   - Mean and variance of reconstruction of in-distribution (ID) examples
   - Mean and variance of reconstruction of out-of-distribution (OOD) examples
   - Distristribution of variances for ID and OOD examples
   - [TODO: Distribution of likelihood for ID and OOD examples]
   - [TODO: ROC curves for ID and OOD examples]
- Latent space:
   - Mean embedding space on top of contour grid of variances
   - Ellipoid of latent embeddings for ID and OOD exampels.

Metrics:
- Output space:
   - [TODO: MSE]
   - [TODO: Likelihood]
- Latent space:
   - [TODO: some geodesic stuff?]

# Latent space
## LAE

![](figures/mnist/lae_elbo/ae_contour.png)
![](figures/mnist/lae_elbo/ood_latent_space.png)
![](figures/mnist/lae_elbo/ood_z_sigma_distribution.png)

# Output space

## LAE

### In distribution
![](figures/mnist/lae_elbo/recon_0.png)
![](figures/mnist/lae_elbo/recon_1.png)
![](figures/mnist/lae_elbo/recon_2.png)
![](figures/mnist/lae_elbo/recon_3.png)

### Out-of-distribution
![](figures/mnist/lae_elbo/ood_recon_0.png)
![](figures/mnist/lae_elbo/ood_recon_1.png)
![](figures/mnist/lae_elbo/ood_recon_2.png)
![](figures/mnist/lae_elbo/ood_recon_3.png)

### Distribution of variances

![](figures/mnist/lae_elbo/ood_x_rec_sigma_distribution.png)