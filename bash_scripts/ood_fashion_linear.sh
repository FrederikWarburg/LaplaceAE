#!/bin/sh


cd ../src/;
pwd
device=1

# ae mse
CUDA_VISIBLE_DEVICES=$device python trainer_ae.py --config ../configs/ood_experiments/fashionmnist/ae_linear.yaml;

# ae likelihood
CUDA_VISIBLE_DEVICES=$device python trainer_ae.py --config ../configs/ood_experiments/fashionmnist/ae_linear_prob.yaml;

# ae dropout
CUDA_VISIBLE_DEVICES=$device python trainer_mcdrop_ae.py --config ../configs/ood_experiments/fashionmnist/ae_dropout_linear.yaml;

# vae 
CUDA_VISIBLE_DEVICES=$device python trainer_vae.py --config ../configs/ood_experiments/fashionmnist/vae_linear.yaml;

# lae elbo
CUDA_VISIBLE_DEVICES=$device python trainer_lae_elbo.py --config ../configs/ood_experiments/fashionmnist/lae_elbo_linear.yaml;

# lae post-hoc
CUDA_VISIBLE_DEVICES=$device python trainer_lae_posthoc.py --config ../configs/ood_experiments/fashionmnist/lae_posthoc_linear.yaml;

# lae bae
CUDA_VISIBLE_DEVICES=$device python trainer_bae.py --config ../configs/ood_experiments/fashionmnist/bae_linear.yaml;