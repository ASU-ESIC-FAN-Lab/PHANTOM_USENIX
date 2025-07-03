# Phantom Artifacts

# Citation

```bash

@inproceedings{bai2025phantom,
  title={Phantom: Privacy-Preserving Deep Neural Network Model Obfuscation in Heterogeneous TEE and GPU System},
  author={Bai, Juyang and Chowdhuryy, Md Hafizul Islam and Li, Jingtao and Yao, Fan and Chakrabarti, Chaitali and Fan, Deliang},
  year={2025},
  organization={34th USENIX Security Symposium}
}
```


## Overview

This repository contains the code for the paper "Phantom: Privacy-Preserving Deep Neural Network Model Obfuscation in Heterogeneous TEE and GPU System". In alignment with our commitment to open science, we provide the source code for the complete Phantom framework, along with evaluation scripts for model training and attack simulation. The detailed description of the code is as follows:

Code Structure:

- `attack/`: contains the code for the fine-tuning attack and model stealing attack mentioned in the paper. 
    - `finetune/`: contains the code and bash script for the fine-tuning attack.
    - `model_stealing/`: contains the code and bash script for the model stealing attack.
- `data/`: contains the data configuration and dataloader for the CIFAR10, CIFAR100, and STL10 datasets.
- `models/`: contains the code for the hyperparameterized AlexNet, ResNet, and VGG models.
    - `alexnet/`: contains the code for the AlexNet model. The `obfuscated_model.py` is the search space setting for the obfuscation alexnet model. The `base_model.py` define model architecture for the hyperparameterized alexnet model.
    - `resnet/`: contains the code for the ResNet model. The `obfuscated_model.py` is the search space setting for the obfuscation resnet model. The `base_model.py` define model architecture for the hyperparameterized resnet model.
    - `vgg/`: contains the code for the VGG model. The `obfuscated_model.py` is the search space setting for the obfuscation vgg model. The `base_model.py` define model architecture for the hyperparameterized vgg model.
- `modules/`: contains the code for the modules.
    - `mix.py`: contains the code for the definition of candidate operations.
    - `layers.py`: contains the code for the defination of different lightweight layer candidates.
- `nas/`: contains the code for the neural architecture search (NAS) and the search space setting.
    - `config.py`: contains the code for the configuration of the NAS, including dataset configuration and the RL search configuration. The reward function we mentioned in the paper is defined in this file.
    - `manager.py`: contains the code for RL search algorithm, including the training and updating process.
- `sensitive_layer/`: contains the code for the sensitive layer analysis, which is used to find the Top-K sensitive layers mentioned in the paper.
- `utils/`: contains the code for the utils functions used for search and attack evaluation.
- `main.py`: contains the main function for architecture search algorithms.
- `environment.yml`: contains the code for setting up the conda environment.
- `bash_search.sh`: bash script to run the architecture search.

## Installation

### Setup

Create and activate conda environment:

```bash
conda env create -f environment.yml
conda activate phantom
```

### Dataset
Download the following datasets from the following links and extract them to the `data` folder:

- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
- [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)
- [STL-10](https://cs.stanford.edu/people/esteva/stl10/)
- [ImageNet](http://www.image-net.org/index.php)

## How to Use

### Architecture Search

We provide the bash command to search for the best architecture for CIFAR-10, CIFAR-100, and STL-10 each:

Run the following command to search for the best architecture for CIFAR-10:
```bash
python main.py --path "./test_cifar10/" --gpu 0 --n_epochs 120 --dataset "cifar10" --train_batch_size 1024 --test_batch_size 1024 --valid_size 1024
```

Run the following command to search for the best architecture for CIFAR-100:
```bash
python main.py --path "./test_cifar100/" --gpu 0 --n_epochs 120 --dataset "cifar10" --train_batch_size 1024 --test_batch_size 1024 --valid_size 1024
```

Run the following command to search for the best architecture for STL-10:
```bash
python main.py --path "./test_stl10/" --gpu 0 --n_epochs 120 --dataset "stl10" --train_batch_size 1024 --test_batch_size 1024 --valid_size 1024
```

We also provide the bash script to search for the best architecture for CIFAR-10, CIFAR-100, and STL-10:

```bash
bash bash_search.sh
```

### Attack Evaluation

#### Fine-tuning Attack

After the architecture search, we provide the bash command to evaluate the fine-tuning attack on the searched architecture under different attack settings (different learning rate, ranging from 0.000001 to 0.9):

```bash
bash bash_finetune.sh
```

#### Model Stealing Attack

The model stealing attack is evaluated using the implementation of the Knockoff Nets. For more details of setting up the environment and usage, please refer to the [Knockoff Nets](https://github.com/alinlab/knockoffnets). We also provide the instructions to run the model stealing attack in the `attack/model_stealing/` folder.

We also provide the bash command to evaluate the model stealing attack on the searched architecture:

```bash
bash bash_train.sh
bash bash_transfer.sh
```
