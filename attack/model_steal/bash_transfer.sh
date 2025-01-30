#!/usr/bin/env bash


#### NN-Splitter

# python knockoff/adversary/transfer.py random \
#         --victim_model_dir models/victim/nnsplitter-cifar10-alexnet \
#         --out_dir models/adversary/nnsplitter-cifar10-alexnet-random \
#         --budget 30000 \
#         --queryset CIFAR100 \
#         --batch_size 64 \
#         --root cifar100data \

# python knockoff/adversary/transfer.py random \
#         --victim_model_dir models/victim/nnsplitter-cifar100-alexnet \
#         --out_dir models/adversary/nnsplitter-cifar100-alexnet-random \
#         --budget 30000 \
#         --queryset CIFAR100 \
#         --batch_size 64 \
#         --root cifar100data \

# python knockoff/adversary/transfer.py random \
#         --victim_model_dir models/victim/nnsplitter-stl10-alexnet \
#         --out_dir models/adversary/nnsplitter-stl10-alexnet-random \
#         --budget 30000 \
#         --queryset CIFAR100 \
#         --batch_size 64 \
#         --root CIFAR100 \

# python knockoff/adversary/transfer.py random \
#         --victim_model_dir models/victim/nnsplitter-cifar10-resnet \
#         --out_dir models/adversary/nnsplitter-cifar10-resnet-random \
#         --budget 30000 \
#         --queryset CIFAR100 \
#         --batch_size 64 \
#         --root cifar10data \

# python knockoff/adversary/transfer.py random \
#         --victim_model_dir models/victim/nnsplitter-cifar100-resnet \
#         --out_dir models/adversary/nnsplitter-cifar100-resnet-random \
#         --budget 30000 \
#         --queryset CIFAR100 \
#         --batch_size 64 \
#         --root cifar100data \

# python knockoff/adversary/transfer.py random \
#         --victim_model_dir models/victim/nnsplitter-stl10-resnet \
#         --out_dir models/adversary/nnsplitter-stl10-resnet-random \
#         --budget 30000 \
#         --queryset CIFAR100 \
#         --batch_size 64 \
#         --root stl10data \

# python knockoff/adversary/transfer.py random \
#         --victim_model_dir models/victim/nnsplitter-cifar10-vgg \
#         --out_dir models/adversary/nnsplitter-cifar10-vgg-random \
#         --budget 30000 \
#         --queryset CIFAR100 \
#         --batch_size 64 \

# python knockoff/adversary/transfer.py random \
#         --victim_model_dir models/victim/nnsplitter-cifar100-vgg \
#         --out_dir models/adversary/nnsplitter-cifar100-vgg-random \
#         --budget 30000 \
#         --queryset CIFAR100 \
#         --batch_size 64 \

# python knockoff/adversary/transfer.py random \
#         --victim_model_dir models/victim/nnsplitter-stl10-vgg \
#         --out_dir models/adversary/nnsplitter-stl10-vgg-random \
#         --budget 30000 \
#         --queryset CIFAR100 \
#         --batch_size 64 \


#### Ours - TOP3

# python knockoff/adversary/transfer.py random \
#         --victim_model_dir models/victim/ours-cifar10-alexnet-top3 \
#         --out_dir models/adversary/ours-cifar10-alexnet-top3-random \
#         --budget 30000 \
#         --queryset CIFAR100 \
#         --batch_size 64 \

# python knockoff/adversary/transfer.py random \
#         --victim_model_dir models/victim/ours-cifar100-alexnet-top3 \
#         --out_dir models/adversary/ours-cifar100-alexnet-top3-random \
#         --budget 30000 \
#         --queryset CIFAR100 \
#         --batch_size 64 \

# python knockoff/adversary/transfer.py random \
#         --victim_model_dir models/victim/ours-stl10-alexnet-top3 \
#         --out_dir models/adversary/ours-stl10-alexnet-top3-random \
#         --budget 30000 \
#         --queryset CIFAR100 \
#         --batch_size 64 \

# python knockoff/adversary/transfer.py random \
#         --victim_model_dir models/victim/ours-cifar10-vgg-top3 \
#         --out_dir models/adversary/ours-cifar10-vgg-top3-random \
#         --budget 30000 \
#         --queryset CIFAR100 \
#         --batch_size 64 \

# python knockoff/adversary/transfer.py random \
#         --victim_model_dir models/victim/ours-cifar100-vgg-top3 \
#         --out_dir models/adversary/ours-cifar100-vgg-top3-random \
#         --budget 30000 \
#         --queryset CIFAR100 \
#         --batch_size 64 \

# python knockoff/adversary/transfer.py random \
#         --victim_model_dir models/victim/ours-stl10-vgg-top3 \
#         --out_dir models/adversary/ours-stl10-vgg-top3-random \
#         --budget 30000 \
#         --queryset CIFAR100 \
#         --batch_size 64 \

# python knockoff/adversary/transfer.py random \
#         --victim_model_dir models/victim/ours-cifar10-resnet-top3 \
#         --out_dir models/adversary/ours-cifar10-resnet-top3-random \
#         --budget 30000 \
#         --queryset CIFAR100 \
#         --batch_size 64 \

# python knockoff/adversary/transfer.py random \
#         --victim_model_dir models/victim/ours-cifar100-resnet-top3 \
#         --out_dir models/adversary/ours-cifar100-resnet-top3-random \
#         --budget 30000 \
#         --queryset CIFAR100 \
#         --batch_size 64 \

# python knockoff/adversary/transfer.py random \
#         --victim_model_dir models/victim/ours-stl10-resnet-top3 \
#         --out_dir models/adversary/ours-stl10-resnet-top3-random \
#         --budget 30000 \
#         --queryset CIFAR100 \
#         --batch_size 64 \





#### Ours - Whole

# python knockoff/adversary/transfer.py random \
#         --victim_model_dir models/victim/ours-cifar10-alexnet-whole \
#         --out_dir models/adversary/ours-cifar10-alexnet-whole-random \
#         --budget 30000 \
#         --queryset CIFAR100 \
#         --batch_size 64 \

# python knockoff/adversary/transfer.py random \
#         --victim_model_dir models/victim/ours-cifar100-alexnet-whole \
#         --out_dir models/adversary/ours-cifar100-alexnet-whole-random \
#         --budget 30000 \
#         --queryset CIFAR100 \
#         --batch_size 64 \

# python knockoff/adversary/transfer.py random \
#         --victim_model_dir models/victim/ours-stl10-alexnet-whole \
#         --out_dir models/adversary/ours-stl10-alexnet-whole-random \
#         --budget 30000 \
#         --queryset CIFAR100 \
#         --batch_size 64 \

# python knockoff/adversary/transfer.py random \
#         --victim_model_dir models/victim/ours-cifar10-vgg-whole \
#         --out_dir models/adversary/ours-cifar10-vgg-whole-random \
#         --budget 30000 \
#         --queryset CIFAR100 \
#         --batch_size 64 \

# python knockoff/adversary/transfer.py random \
#         --victim_model_dir models/victim/ours-cifar100-vgg-whole \
#         --out_dir models/adversary/ours-cifar100-vgg-whole-random \
#         --budget 30000 \
#         --queryset CIFAR100 \
#         --batch_size 64 \

# python knockoff/adversary/transfer.py random \
#         --victim_model_dir models/victim/ours-stl10-vgg-whole \
#         --out_dir models/adversary/ours-stl10-vgg-whole-random \
#         --budget 30000 \
#         --queryset CIFAR100 \
#         --batch_size 64 \

# python knockoff/adversary/transfer.py random \
#         --victim_model_dir models/victim/ours-cifar10-resnet-whole \
#         --out_dir models/adversary/ours-cifar10-resnet-whole-random \
#         --budget 30000 \
#         --queryset CIFAR100 \
#         --batch_size 64 \

