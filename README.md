# Computer vision experiments

This repository contains code for the second assignment for the *Computer
vision* subject at the University of Granada, course 2020/21. It contains
experiments I ran in order to improve a baseline model that was given.

## Purpose

The goal was to improve a baseline model (`basenet` object in
[definitions-1.py](definitions-1.py)) performance on a subset of CIFAR-100 with
25 classes. Improvements were to be made to the architecture, through the
addition and modification of layers, and also through data augmentation and
early stopping. However, other elements of training, such as the optimizer,
should be fixed across all experiments.

## How it works

Each of the model definition scripts (`definitions-*.py`) contains definitions
of several architectures. Each of the experiment scripts (`experiment-*.py`)
loads previously saved architectures from a specified directory and trains them
using a specific combination of data augmentation and early stopping with
specific parameters.

## Usage on cloud infrastructure

The [`makefile`](makefile) contains orders to run these experiments on
Paperspace cloud. Gradient CLI should be installed, and an API key should be
provided.

## License

The software in this repository is distributed under the terms of the GPLv3
license. The text of this license can be found in the [LICENSE](LICENSE) file.
