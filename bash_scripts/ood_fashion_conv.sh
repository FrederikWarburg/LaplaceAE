#!/bin/sh


cd ../src/;
pwd
device=2

# ae mse
CUDA_VISIBLE_DEVICES=$device python trainer_ae.py --config ../configs/ood_experiments/fashionmnist/ae_conv.yaml;

# ae likelihood
CUDA_VISIBLE_DEVICES=$device python trainer_ae.py --config ../configs/ood_experiments/fashionmnist/ae_conv_prob.yaml;

# ae dropout
CUDA_VISIBLE_DEVICES=$device python trainer_mcdrop_ae.py --config ../configs/ood_experiments/fashionmnist/ae_dropout_conv.yaml;

# vae 
CUDA_VISIBLE_DEVICES=$device python trainer_vae.py --config ../configs/ood_experiments/fashionmnist/vae_conv.yaml;

# lae elbo
CUDA_VISIBLE_DEVICES=$device python trainer_lae_elbo.py --config ../configs/ood_experiments/fashionmnist/lae_elbo_conv.yaml;

# lae post-hoc
# CUDA_VISIBLE_DEVICES=$device python trainer_lae_posthoc.py --config ../configs/ood_experiments/fashionmnist/lae_posthoc_conv.yaml;

# lae bae
# CUDA_VISIBLE_DEVICES=$device python trainer_bae.py --config ../configs/ood_experiments/fashionmnist/bae_linear.yaml;