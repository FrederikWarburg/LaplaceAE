# Results

We have conducted experiments for the following models. 

Models:
- AE
  - MSE
  - NLL
- VAE
- AE dropout
- LAE post hoc
  - decoder only
  - entire model
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

## LAE

<table>
  <tr>
    <td>Encoder uncertainties</td>
     <td>Decoder uncertainties</td>
  </tr>
  <tr>
   <td><img src="figures/mnist/lae_elbo/ood_latent_space.png">
    <td><img src="figures/mnist/lae_elbo/ae_contour.png"></td>
    </td>
  </tr>
  <tr>
    <td><img src="figures/mnist/lae_elbo/ood_z_sigma_distribution.png"></td>
    <td><img src="figures/mnist/lae_elbo/ood_x_rec_sigma_distribution.png"></td>
  </tr>
</table>

<table>
  <tr>
    <td>In distribution</td>
     <td>Out-of-distribution</td>
  </tr>
  <tr>
    <td><img src="figures/mnist/lae_elbo/recon_0.png"></td>
    <td><img src="figures/mnist/lae_elbo/ood_recon_0.png"></td>
  </tr>
  <tr>
    <td><img src="figures/mnist/lae_elbo/recon_1.png"></td>
    <td><img src="figures/mnist/lae_elbo/ood_recon_1.png"></td>
  </tr>
  <tr>
    <td><img src="figures/mnist/lae_elbo/recon_2.png"></td>
    <td><img src="figures/mnist/lae_elbo/ood_recon_2.png"></td>
  </tr>
  <tr>
    <td><img src="figures/mnist/lae_elbo/recon_3.png"></td>
    <td><img src="figures/mnist/lae_elbo/ood_recon_3.png"></td>
  </tr>
  <tr>
    <td><img src="figures/mnist/lae_elbo/recon_4.png"></td>
    <td><img src="figures/mnist/lae_elbo/ood_recon_4.png"></td>
  </tr>
 </table>

## LAE post hoc [Encoder & Decoder]

<table>
  <tr>
    <td>Encoder uncertainties</td>
     <td>Decoder uncertainties</td>
  </tr>
  <tr>
   <td><img src="figures/mnist/lae_post_hoc_[use_la_encoder=True]/ood_latent_space.png">
    <td><img src="figures/mnist/lae_post_hoc_[use_la_encoder=True]/ae_contour.png"></td>
    </td>
  </tr>
  <tr>
    <td><img src="figures/mnist/lae_post_hoc_[use_la_encoder=True]/ood_z_sigma_distribution.png"></td>
    <td><img src="figures/mnist/lae_post_hoc_[use_la_encoder=True]/ood_x_rec_sigma_distribution.png"></td>
  </tr>
</table>

<table>
  <tr>
    <td>In distribution</td>
     <td>Out-of-distribution</td>
  </tr>
  <tr>
    <td><img src="figures/mnist/lae_post_hoc_[use_la_encoder=True]/recon_0.png"></td>
    <td><img src="figures/mnist/lae_post_hoc_[use_la_encoder=True]/ood_recon_0.png"></td>
  </tr>
  <tr>
    <td><img src="figures/mnist/lae_post_hoc_[use_la_encoder=True]/recon_1.png"></td>
    <td><img src="figures/mnist/lae_post_hoc_[use_la_encoder=True]/ood_recon_1.png"></td>
  </tr>
  <tr>
    <td><img src="figures/mnist/lae_post_hoc_[use_la_encoder=True]/recon_2.png"></td>
    <td><img src="figures/mnist/lae_post_hoc_[use_la_encoder=True]/ood_recon_2.png"></td>
  </tr>
  <tr>
    <td><img src="figures/mnist/lae_post_hoc_[use_la_encoder=True]/recon_3.png"></td>
    <td><img src="figures/mnist/lae_post_hoc_[use_la_encoder=True]/ood_recon_3.png"></td>
  </tr>
  <tr>
    <td><img src="figures/mnist/lae_post_hoc_[use_la_encoder=True]/recon_4.png"></td>
    <td><img src="figures/mnist/lae_post_hoc_[use_la_encoder=True]/ood_recon_4.png"></td>
  </tr>
 </table>

## LAE post hoc [Only Decoder]

<table>
  <tr>
    <td>Encoder uncertainties</td>
     <td>Decoder uncertainties</td>
  </tr>
  <tr>
   <td><img src="figures/mnist/lae_post_hoc_[use_la_encoder=False]/ood_latent_space.png">
    <td><img src="figures/mnist/lae_post_hoc_[use_la_encoder=False]/ae_contour.png"></td>
    </td>
  </tr>
  <tr>
    <td><img src="figures/mnist/lae_post_hoc_[use_la_encoder=False]/ood_z_sigma_distribution.png"></td>
    <td><img src="figures/mnist/lae_post_hoc_[use_la_encoder=False]/ood_x_rec_sigma_distribution.png"></td>
  </tr>
</table>

<table>
  <tr>
    <td>In distribution</td>
     <td>Out-of-distribution</td>
  </tr>
  <tr>
    <td><img src="figures/mnist/lae_post_hoc_[use_la_encoder=False]/recon_0.png"></td>
    <td><img src="figures/mnist/lae_post_hoc_[use_la_encoder=False]/ood_recon_0.png"></td>
  </tr>
  <tr>
    <td><img src="figures/mnist/lae_post_hoc_[use_la_encoder=False]/recon_1.png"></td>
    <td><img src="figures/mnist/lae_post_hoc_[use_la_encoder=False]/ood_recon_1.png"></td>
  </tr>
  <tr>
    <td><img src="figures/mnist/lae_post_hoc_[use_la_encoder=False]/recon_2.png"></td>
    <td><img src="figures/mnist/lae_post_hoc_[use_la_encoder=False]/ood_recon_2.png"></td>
  </tr>
  <tr>
    <td><img src="figures/mnist/lae_post_hoc_[use_la_encoder=False]/recon_3.png"></td>
    <td><img src="figures/mnist/lae_post_hoc_[use_la_encoder=False]/ood_recon_3.png"></td>
  </tr>
  <tr>
    <td><img src="figures/mnist/lae_post_hoc_[use_la_encoder=False]/recon_4.png"></td>
    <td><img src="figures/mnist/lae_post_hoc_[use_la_encoder=False]/ood_recon_4.png"></td>
  </tr>
 </table>

## AE [MSE]

<table>
  <tr>
    <td>Encoder uncertainties</td>
     <td>Decoder uncertainties</td>
  </tr>
  <tr>
   <td><img src="figures/mnist/ae_[use_var_dec=False]/ood_latent_space.png">
    <td><img src="figures/mnist/ae_[use_var_dec=False]/ae_contour.png"></td>
    </td>
  </tr>
  <tr>
    <td><img src="figures/mnist/ae_[use_var_dec=False]/ood_z_sigma_distribution.png"></td>
    <td><img src="figures/mnist/ae_[use_var_dec=False]/ood_x_rec_sigma_distribution.png"></td>
  </tr>
</table>

<table>
  <tr>
    <td>In distribution</td>
     <td>Out-of-distribution</td>
  </tr>
  <tr>
    <td><img src="figures/mnist/ae_[use_var_dec=False]/recon_0.png"></td>
    <td><img src="figures/mnist/ae_[use_var_dec=False]/ood_recon_0.png"></td>
  </tr>
  <tr>
    <td><img src="figures/mnist/ae_[use_var_dec=False]/recon_1.png"></td>
    <td><img src="figures/mnist/ae_[use_var_dec=False]/ood_recon_1.png"></td>
  </tr>
  <tr>
    <td><img src="figures/mnist/ae_[use_var_dec=False]/recon_2.png"></td>
    <td><img src="figures/mnist/ae_[use_var_dec=False]/ood_recon_2.png"></td>
  </tr>
  <tr>
    <td><img src="figures/mnist/ae_[use_var_dec=False]/recon_3.png"></td>
    <td><img src="figures/mnist/ae_[use_var_dec=False]/ood_recon_3.png"></td>
  </tr>
  <tr>
    <td><img src="figures/mnist/ae_[use_var_dec=False]/recon_4.png"></td>
    <td><img src="figures/mnist/ae_[use_var_dec=False]/ood_recon_4.png"></td>
  </tr>
 </table>


## AE [NLL]

<table>
  <tr>
    <td>Encoder uncertainties</td>
     <td>Decoder uncertainties</td>
  </tr>
  <tr>
   <td><img src="figures/mnist/ae_[use_var_dec=True]/ood_latent_space.png">
    <td><img src="figures/mnist/ae_[use_var_dec=True]/ae_contour.png"></td>
    </td>
  </tr>
  <tr>
    <td><img src="figures/mnist/ae_[use_var_dec=True]/ood_z_sigma_distribution.png"></td>
    <td><img src="figures/mnist/ae_[use_var_dec=True]/ood_x_rec_sigma_distribution.png"></td>
  </tr>
</table>

<table>
  <tr>
    <td>In distribution</td>
     <td>Out-of-distribution</td>
  </tr>
  <tr>
    <td><img src="figures/mnist/ae_[use_var_dec=True]/recon_0.png"></td>
    <td><img src="figures/mnist/ae_[use_var_dec=True]/ood_recon_0.png"></td>
  </tr>
  <tr>
    <td><img src="figures/mnist/ae_[use_var_dec=True]/recon_1.png"></td>
    <td><img src="figures/mnist/ae_[use_var_dec=True]/ood_recon_1.png"></td>
  </tr>
  <tr>
    <td><img src="figures/mnist/ae_[use_var_dec=True]/recon_2.png"></td>
    <td><img src="figures/mnist/ae_[use_var_dec=True]/ood_recon_2.png"></td>
  </tr>
  <tr>
    <td><img src="figures/mnist/ae_[use_var_dec=True]/recon_3.png"></td>
    <td><img src="figures/mnist/ae_[use_var_dec=True]/ood_recon_3.png"></td>
  </tr>
  <tr>
    <td><img src="figures/mnist/ae_[use_var_dec=True]/recon_4.png"></td>
    <td><img src="figures/mnist/ae_[use_var_dec=True]/ood_recon_4.png"></td>
  </tr>
 </table>


## AE [MC-DROPOUT]


## VAE [Sampling]


<table>
  <tr>
    <td>Encoder uncertainties</td>
     <td>Decoder uncertainties</td>
  </tr>
  <tr>
   <td><img src="figures/mnist/vae_[use_var_dec=False]/ood_latent_space.png">
    <td><img src="figures/mnist/vae_[use_var_dec=False]/ae_contour.png"></td>
    </td>
  </tr>
  <tr>
    <td><img src="figures/mnist/vae_[use_var_dec=False]/ood_z_sigma_distribution.png"></td>
    <td><img src="figures/mnist/vae_[use_var_dec=False]/ood_x_rec_sigma_distribution.png"></td>
  </tr>
</table>

<table>
  <tr>
    <td>In distribution</td>
     <td>Out-of-distribution</td>
  </tr>
  <tr>
    <td><img src="figures/mnist/vae_[use_var_dec=False]/recon_0.png"></td>
    <td><img src="figures/mnist/vae_[use_var_dec=False]/ood_recon_0.png"></td>
  </tr>
  <tr>
    <td><img src="figures/mnist/vae_[use_var_dec=False]/recon_1.png"></td>
    <td><img src="figures/mnist/vae_[use_var_dec=False]/ood_recon_1.png"></td>
  </tr>
  <tr>
    <td><img src="figures/mnist/vae_[use_var_dec=False]/recon_2.png"></td>
    <td><img src="figures/mnist/vae_[use_var_dec=False]/ood_recon_2.png"></td>
  </tr>
  <tr>
    <td><img src="figures/mnist/vae_[use_var_dec=False]/recon_3.png"></td>
    <td><img src="figures/mnist/vae_[use_var_dec=False]/ood_recon_3.png"></td>
  </tr>
  <tr>
    <td><img src="figures/mnist/vae_[use_var_dec=False]/recon_4.png"></td>
    <td><img src="figures/mnist/vae_[use_var_dec=False]/ood_recon_4.png"></td>
  </tr>
 </table>



## VAE [Variance Decoder]

<table>
  <tr>
    <td>Encoder uncertainties</td>
     <td>Decoder uncertainties</td>
  </tr>
  <tr>
   <td><img src="figures/mnist/vae_[use_var_dec=True]/ood_latent_space.png">
    <td><img src="figures/mnist/vae_[use_var_dec=True]/ae_contour.png"></td>
    </td>
  </tr>
  <tr>
    <td><img src="figures/mnist/vae_[use_var_dec=True]/ood_z_sigma_distribution.png"></td>
    <td><img src="figures/mnist/vae_[use_var_dec=True]/ood_x_rec_sigma_distribution.png"></td>
  </tr>
</table>

<table>
  <tr>
    <td>In distribution</td>
     <td>Out-of-distribution</td>
  </tr>
  <tr>
    <td><img src="figures/mnist/vae_[use_var_dec=True]/recon_0.png"></td>
    <td><img src="figures/mnist/vae_[use_var_dec=True]/ood_recon_0.png"></td>
  </tr>
  <tr>
    <td><img src="figures/mnist/vae_[use_var_dec=True]/recon_1.png"></td>
    <td><img src="figures/mnist/vae_[use_var_dec=True]/ood_recon_1.png"></td>
  </tr>
  <tr>
    <td><img src="figures/mnist/vae_[use_var_dec=True]/recon_2.png"></td>
    <td><img src="figures/mnist/vae_[use_var_dec=True]/ood_recon_2.png"></td>
  </tr>
  <tr>
    <td><img src="figures/mnist/vae_[use_var_dec=True]/recon_3.png"></td>
    <td><img src="figures/mnist/vae_[use_var_dec=True]/ood_recon_3.png"></td>
  </tr>
  <tr>
    <td><img src="figures/mnist/vae_[use_var_dec=True]/recon_4.png"></td>
    <td><img src="figures/mnist/vae_[use_var_dec=True]/ood_recon_4.png"></td>
  </tr>
 </table>

