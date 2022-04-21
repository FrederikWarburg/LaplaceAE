#!/usr/bin/env bash

number_of_layers=3
data_sizes=(2 4 8 16 24 32 48 64)
backends=("layer_block" "layer_exact" "layer_approx" "layer_mix" "row" "la" "backpack")

for size in ${data_sizes[@]}
do
    for backend in ${backends[@]}
    do
        CUDA_VISIBLE_DEVICES=0 python run_time_comparisons_conv.py --data_size $size --number_of_layers $number_of_layers --backend $backend
    done
done


number_of_layers=(3 5 10 20 30 40 50 60 70 80 90 100)
data_sizes=16
backends=("layer_block" "layer_exact" "layer_approx" "layer_mix" "row" "la" "backpack")

for layers in ${number_of_layers[@]}
do
    for backend in ${backends[@]}
    do
        CUDA_VISIBLE_DEVICES=0 python run_time_comparisons_conv.py --data_size $data_sizes --number_of_layers $layers --backend $backend
    done
done