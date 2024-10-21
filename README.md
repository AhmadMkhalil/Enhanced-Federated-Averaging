Here's a cleaner and more polished version of your README file with improved formatting and structure, eliminating the need for user input prompts:

---

# Federated Learning (PyTorch)

This repository is based on the original work from [AshwinRJ/Federated-Learning-PyTorch](https://github.com/AshwinRJ/Federated-Learning-PyTorch), with significant modifications and added features to fit the scope of my research topic on **Federated Learning**.

## Overview
This repository showcases federated learning experiments using the MNIST, Fashion MNIST, and CIFAR-10 datasets under both IID (Independent and Identically Distributed) and non-IID conditions. It allows for flexible data distribution among users, supporting both equal and unequal splits. Simple yet effective models such as MLP (Multi-Layer Perceptron) and CNN (Convolutional Neural Networks) are employed to assess the performance of federated learning.

The primary extension beyond the original implementation is the introduction of advanced data distribution schemes, including the ability to simulate overlapping data samples among users and control the degree of sample overlap. These features provide a more nuanced exploration of federated learning in heterogeneous environments.

## Key Features and Modifications
- **Label-Aware Aggregation** method, developed as part of my research, to enhance federated learning with non-IID data distributions.
- Additional hyperparameter tuning options.
- Support for unequal data splits across users.
- Improved logging and visualization.

---

## Requirements
Ensure you have the following dependencies installed. All required packages are listed in `requirements.txt`:

- Python 3.x
- PyTorch
- Torchvision

To install the dependencies, run:

```bash
pip install -r requirements.txt
```

---

## Data

- The datasets (MNIST, Fashion MNIST, CIFAR-10) are automatically downloaded via `torchvision.datasets`. Alternatively, you can manually place them in the `data/` directory.
- To use your own dataset, move it to the `data/` folder and create a custom data wrapper using PyTorch's Dataset class.

---

## Running the Experiments

### Baseline Experiment
The baseline experiment trains the model in a traditional (non-federated) manner.

#### Example: Running a Baseline Experiment on MNIST with MLP
```bash
python src/baseline_main.py --model=mlp --dataset=mnist --epochs=10
```

#### Example: Running the Baseline Experiment on GPU
```bash
python src/baseline_main.py --model=mlp --dataset=mnist --gpu=0 --epochs=10
```

### Federated Experiment
Federated experiments involve training a global model by aggregating updates from local models trained on distributed user data.

#### Example: Running a Federated Experiment on CIFAR-10 with CNN (IID)
```bash
python src/federated_main.py --model=cnn --dataset=cifar --gpu=0 --iid=1 --epochs=10
```

#### Example: Running a Federated Experiment on CIFAR-10 with CNN (non-IID)
```bash
python src/federated_main.py --model=cnn --dataset=cifar --gpu=0 --iid=0 --epochs=10
```

You can customize other parameters to simulate different conditions. Refer to the options section below for details.

---

## Options and Hyperparameters

The key configurable parameters can be found and modified in `options.py`. Below are some frequently used options:

### General Parameters
- `--dataset`: Dataset to be used. Default: `mnist`. Options: `mnist`, `fmnist`, `cifar`.
- `--model`: Model architecture. Default: `mlp`. Options: `mlp`, `cnn`.
- `--gpu`: GPU ID. Set to `None` (default) for CPU execution.
- `--epochs`: Number of training epochs. Default: `10`.
- `--lr`: Learning rate. Default: `0.01`.
- `--verbose`: Detailed logging. Default: `1` (enabled). Set to `0` to disable.
- `--seed`: Random seed for reproducibility. Default: `1`.

### Federated Learning Parameters
- `--iid`: Data distribution among users. Default: `1` (IID). Set to `0` for non-IID distribution.
- `--num_users`: Number of clients (users) participating in federated learning. Default: `100`.
- `--frac`: Fraction of clients used for each round of federated updates. Default: `0.1`.
- `--local_ep`: Number of local epochs (training iterations) per client. Default: `10`.
- `--local_bs`: Batch size for local training. Default: `10`.
- `--unequal`: Set to `1` for unequal data splits among users (non-IID). Default: `0` (equal splits).
