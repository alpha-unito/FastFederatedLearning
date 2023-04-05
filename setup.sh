#!/bin/bash -f
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Libraries download
cd $SCRIPT_DIR
if [ ! -d libs ]; then
  mkdir libs
  cd libs/
  git clone -b v1.3.2 --single-branch --depth 1 https://github.com/USCiLab/cereal.git
  export CEREAL_HOME="$SCRIPT_DIR/libs/cereal/include"

  # wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcpu.zip
  # unzip libtorch-cxx11-abi-shared-with-deps-2.0.0+cpu.zip
  # rm libtorch-cxx11-abi-shared-with-deps-2.0.0+cpu.zip

  # Try to guess OS and ARCH for binary libtorch
  if [ ! -d torch ]; then
    ARCH=$(uname -m)
    OS=$(uname -s)
    [ "$OS" ==  "Darwin" ] && TORCHURL="https://files.pythonhosted.org/packages/4d/80/760f3edcf0179c3111fae496b97ee3fa9171116b4bccae6e073efe928e72/torch-2.0.0-cp39-none-macosx_11_0_$ARCH.whl"
    [ "$OS" ==  "Linux" ]  && [ "$ARCH" ==  "x86_64" ]  && TORCHURL="https://files.pythonhosted.org/packages/5f/24/16e94ac3a470027a2f6cf56dbbe2ce1b2742fa0ac98844f039fad103e142/torch-2.0.0-cp39-cp39-manylinux1_x86_64.whl"
    [ "$OS" ==  "Linux" ]  && [ "$ARCH" ==  "aarch64" ] && TORCHURL="https://files.pythonhosted.org/packages/36/60/aa7bf18070611e7b019886d34516337ce6a2fe9da60745bc90b448642a10/torch-2.0.0-cp39-cp39-manylinux2014_aarch64.whl"
    [ "$OS" ==  "Linux" ]  && [ "$ARCH" ==  "riscv64" ] && TORCHURL="https://gitlab.di.unito.it/alpha/riscv/torch/-/package_files/17/download"
    
    [ "$TORCHURL" == "" ] && echo "Could not determine the libtorch binary for this system. Please manualy install libtorch to $SCRIPT_DIR/libs/torch" && exit 1

    curl "$TORCHURL" -o torch.whl
    unzip torch.whl "torch/*"
    rm torch.whl
  fi
  

  # Build the dff_run utility
  git clone -b DistributedFF --single-branch --depth 1  https://github.com/fastflow/fastflow.git
  cd fastflow/ff/distributed/loader
  make
fi

# Environment variables setting
export TORCH_HOME="$SCRIPT_DIR/libs/torch"
export CEREAL_HOME="$SCRIPT_DIR/libs/cereal/include"
export FF_HOME="$SCRIPT_DIR/libs/fastflow"


# MNIST dataset download
cd $SCRIPT_DIR
if [ ! -d data ]; then
  mkdir data
  cd data
  curl -o - http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz | gunzip > train-images-idx3-ubyte
  curl -o - http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz | gunzip > train-labels-idx1-ubyte
  curl -o - http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz | gunzip > t10k-images-idx3-ubyte
  curl -o - http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz | gunzip > t10k-labels-idx1-ubyte
fi

# Code building
cd $SCRIPT_DIR
[ -d build ] && rm -r build
mkdir build
cd build/
cmake ../
make
cd examples/
