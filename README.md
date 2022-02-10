# LaplaceAE

Code for Laplace Auto-Encoding Bayes and the associate baseline methods.

## Setup

```
git clone https://github.com/FrederikWarburg/Laplace;
```
This should give you a folder structure like this:

```
* Laplace
* src
   * ae_models.py
   * ...
```

## Train & Test

To train and test a VAE on MNIST run:

```
cd src; CUDA_VISIBLE_DEVICES=0 python trainer_vae.py
```

To train AE  on MNIST run:

```
cd src; CUDA_VISIBLE_DEVICES=0 python trainer_ae.py
```


To train LAE  on MNIST run:

```
cd src; CUDA_VISIBLE_DEVICES=0 python trainer_lae.py
```


To train and test a AE and then LAE on MNIST run:

```
cd src; CUDA_VISIBLE_DEVICES=0 python trainer_e2e.py
```

You can run ```trainer_lae.py``` if you only want to train the Laplace approximation and ```trainer_ae.py``` if you only want to train an autoencoder.

