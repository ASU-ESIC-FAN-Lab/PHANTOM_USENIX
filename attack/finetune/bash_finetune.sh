#!/bin/bash

python finetune.py --gpu 0 --manual_seed 42 --optim SGD --lr 0.9 --momentum 0.9 --epoch 150 --sch CosineAnnealingLR --dataset cifar10 &
python finetune.py --gpu 0 --manual_seed 42 --optim SGD --lr 0.5 --momentum 0.9 --epoch 150 --sch CosineAnnealingLR --dataset cifar10 &
python finetune.py --gpu 0 --manual_seed 42 --optim SGD --lr 0.1 --momentum 0.9 --epoch 150 --sch CosineAnnealingLR --dataset cifar10 &
python finetune.py --gpu 0 --manual_seed 42 --optim SGD --lr 0.01 --momentum 0.9 --epoch 150 --sch CosineAnnealingLR --dataset cifar10

wait

python exe.py --gpu 1 --manual_seed 42 --optim SGD --lr 0.001 --momentum 0.9 --epoch 150 --sch CosineAnnealingLR --dataset cifar10 &
python exe.py --gpu 1 --manual_seed 42 --optim SGD --lr 0.0001 --momentum 0.9 --epoch 150 --sch CosineAnnealingLR --dataset cifar10 &
python exe.py --gpu 1 --manual_seed 42 --optim SGD --lr 0.00001 --momentum 0.9 --epoch 150 --sch CosineAnnealingLR --dataset cifar10 &
python exe.py --gpu 1 --manual_seed 42 --optim SGD --lr 0.000001 --momentum 0.9 --epoch 150 --sch CosineAnnealingLR --dataset cifar10 &

wait

python finetune.py --gpu 0 --manual_seed 42 --optim SGD --lr 0.9 --momentum 0.9 --epoch 150 --sch CosineAnnealingLR --dataset stl10 &
python finetune.py --gpu 0 --manual_seed 42 --optim SGD --lr 0.5 --momentum 0.9 --epoch 150 --sch CosineAnnealingLR --dataset stl10 &
python finetune.py --gpu 0 --manual_seed 42 --optim SGD --lr 0.1 --momentum 0.9 --epoch 150 --sch CosineAnnealingLR --dataset stl10 &
python finetune.py --gpu 0 --manual_seed 42 --optim SGD --lr 0.01 --momentum 0.9 --epoch 150 --sch CosineAnnealingLR --dataset stl10

wait

python exe.py --gpu 1 --manual_seed 42 --optim SGD --lr 0.001 --momentum 0.9 --epoch 150 --sch CosineAnnealingLR --dataset stl10 &
python exe.py --gpu 1 --manual_seed 42 --optim SGD --lr 0.0001 --momentum 0.9 --epoch 150 --sch CosineAnnealingLR --dataset stl10 &
python exe.py --gpu 1 --manual_seed 42 --optim SGD --lr 0.00001 --momentum 0.9 --epoch 150 --sch CosineAnnealingLR --dataset stl10 &
python exe.py --gpu 1 --manual_seed 42 --optim SGD --lr 0.000001 --momentum 0.9 --epoch 150 --sch CosineAnnealingLR --dataset stl10

wait

python finetune.py --gpu 0 --manual_seed 42 --optim SGD --lr 0.9 --momentum 0.9 --epoch 150 --sch CosineAnnealingLR --dataset cifar100 &
python finetune.py --gpu 0 --manual_seed 42 --optim SGD --lr 0.5 --momentum 0.9 --epoch 150 --sch CosineAnnealingLR --dataset cifar100 &
python finetune.py --gpu 0 --manual_seed 42 --optim SGD --lr 0.1 --momentum 0.9 --epoch 150 --sch CosineAnnealingLR --dataset cifar100 &
python finetune.py --gpu 0 --manual_seed 42 --optim SGD --lr 0.01 --momentum 0.9 --epoch 150 --sch CosineAnnealingLR --dataset cifar100

wait

python exe.py --gpu 1 --manual_seed 42 --optim SGD --lr 0.001 --momentum 0.9 --epoch 150 --sch CosineAnnealingLR --dataset cifar100 &
python exe.py --gpu 1 --manual_seed 42 --optim SGD --lr 0.0001 --momentum 0.9 --epoch 150 --sch CosineAnnealingLR --dataset cifar100 &
python exe.py --gpu 1 --manual_seed 42 --optim SGD --lr 0.00001 --momentum 0.9 --epoch 150 --sch CosineAnnealingLR --dataset cifar100 &
python exe.py --gpu 1 --manual_seed 42 --optim SGD --lr 0.000001 --momentum 0.9 --epoch 150 --sch CosineAnnealingLR --dataset cifar100

wait