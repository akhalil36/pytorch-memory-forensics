#!/bin/bash

echo "Downloading LibTorch..."
wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-latest.zip
unzip libtorch-shared-with-deps-latest.zip
rm libtorch-shared-with-deps-latest.zip

echo "Downloading CIFAR-10 dataset..."
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -P data/
tar -xzf data/cifar-10-python.tar.gz -C data/

echo "Done!"
