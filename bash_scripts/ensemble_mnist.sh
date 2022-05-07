#!/bin/sh
cd ../src/;
pwd
device=3

# ae mse
#for i in {1..4}
#do
#    echo $i
#    CUDA_VISIBLE_DEVICES=$device python trainer_ae.py --config ../configs/mnist_model_selection/ae_linear.yaml --version $i &
#done
#CUDA_VISIBLE_DEVICES=$device python trainer_ae.py --config ../configs/mnist_model_selection/ae_linear.yaml --version $((i+1));

# ae likelihood
#for i in {1..4}
#do
#    echo $i
#    CUDA_VISIBLE_DEVICES=$device python trainer_ae.py --config ../configs/mnist_model_selection/ae_linear_prob.yaml --version $i &
#done
#CUDA_VISIBLE_DEVICES=$device python trainer_ae.py --config ../configs/mnist_model_selection/ae_linear_prob.yaml --version $((i+1));

# ae dropout
#for i in {1..4}
#do
#    echo $i
#    CUDA_VISIBLE_DEVICES=$device python trainer_mcdrop_ae.py --config ../configs/mnist_model_selection/ae_dropout_linear.yaml --version $i &
#done
#CUDA_VISIBLE_DEVICES=$device python trainer_mcdrop_ae.py --config ../configs/mnist_model_selection/ae_dropout_linear.yaml --version $((i+1));

# vae 
#for i in {1..4}
#do
#    echo $i
#    CUDA_VISIBLE_DEVICES=$device python trainer_vae.py --config ../configs/mnist_model_selection/vae_linear.yaml --version $i &
#done
#CUDA_VISIBLE_DEVICES=$device python trainer_vae.py --config ../configs/mnist_model_selection/vae_linear.yaml --version $((i+1));

# lae elbo
for i in {1..4}
do
    echo $i
    CUDA_VISIBLE_DEVICES=$i python trainer_lae_elbo.py --config ../configs/mnist_model_selection/lae_elbo_linear.yaml --version $i &
done
CUDA_VISIBLE_DEVICES=$((i+1)) python trainer_lae_elbo.py --config ../configs/mnist_model_selection/lae_elbo_linear.yaml --version $((i+1));
