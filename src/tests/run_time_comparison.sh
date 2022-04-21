#!/usr/bin/env bash

number_of_layers=5
data_sizes=(10 50 100 500 1000 1500 2000 2500)
backends=("layer" "row" "la")

for size in ${data_sizes[@]}
do
    for backend in ${backends[@]}
    do
        CUDA_VISIBLE_DEVICES=0 python run_time_comparisons.py --data_size $size --number_of_layers $number_of_layers --backend $backend
    done
done


number_of_layers=(5 10 20 30 40 50 60 70 80 90 100)
data_sizes=500
backends=("layer" "row" "la")

for layers in ${number_of_layers[@]}
do
    for backend in ${backends[@]}
    do
        CUDA_VISIBLE_DEVICES=0 python run_time_comparisons.py --data_size $data_sizes --number_of_layers $layers --backend $backend
    done
done