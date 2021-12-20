# LaplaceAE

Code for Laplace Auto-Encoding Bayes and the associate baseline methods.

To train and test a VAE on MNIST run:

```
cd src; CUDA_VISIBLE_DEVICES=0 python trainer_vae.py
```

To train and test a LAE on MNIST run:

```
cd src; CUDA_VISIBLE_DEVICES=0 python trainer_e2e.py
```

You can run ```trainer_lae.py``` if you only want to train the Laplace approximation and ```trainer_ae.py``` if you only want to train an autoencoder.

