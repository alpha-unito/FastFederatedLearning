#!/bin/bash -f
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Libraries download
cd $SCRIPT_DIR
if [ ! -d libs ]; then
  mkdir libs
  cd libs/

  # Download cereal library
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
    [ "$OS" ==  "Linux" ]  && [ "$ARCH" ==  "x86_64" ]  && TORCHURL="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcpu.zip"
    [ "$OS" ==  "Linux" ]  && [ "$ARCH" ==  "aarch64" ] && TORCHURL="https://files.pythonhosted.org/packages/36/60/aa7bf18070611e7b019886d34516337ce6a2fe9da60745bc90b448642a10/torch-2.0.0-cp39-cp39-manylinux2014_aarch64.whl"
    [ "$OS" ==  "Linux" ]  && [ "$ARCH" ==  "riscv64" ] && TORCHURL="https://gitlab.di.unito.it/alpha/riscv/torch/-/package_files/17/download"
    
    [ "$TORCHURL" == "" ] && echo "Could not determine the libtorch binary for this system. Please manualy install libtorch to $SCRIPT_DIR/libs/torch" && exit 1

    curl "$TORCHURL" -o torch.whl
    unzip torch.whl "torch/*" "libtorch/*" > /dev/null
    mv libtorch torch  &> /dev/null
    rm -f torch.whl
  fi

  # Download and compile openCV
#  if [ ! -d opencv ]; then
#    git clone -b 4.7.0 --depth 1 https://github.com/opencv/opencv.git opencv_src
#    mkdir -p opencv_src/build
#    cd opencv_src/build
#    cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF -DBUILD_DOCS=OFF -DCMAKE_INSTALL_PREFIX="$SCRIPT_DIR/libs/opencv" ../
#    make -j $(getconf _NPROCESSORS_ONLN)
#    make install
#    cd -
#  fi

  # Download fastflow and build the dff_run utility
  git clone -b DistributedFF --single-branch --depth 1  https://github.com/fastflow/fastflow.git
  cd fastflow/ff/distributed/loader
  make
fi

# Environment variables setting
export TORCH_HOME="$SCRIPT_DIR/libs/torch"
#export OPENCV_HOME="$SCRIPT_DIR/libs/opencv"
export CEREAL_HOME="$SCRIPT_DIR/libs/cereal/include"
export FF_HOME="$SCRIPT_DIR/libs/fastflow"


# MNIST dataset download
cd $SCRIPT_DIR
if [ ! -d data ]; then
  wget https://datacloud.di.unito.it/index.php/s/6qgZGtMMeqm3Ytq/download
  unzip download
  rm -rf download __MACOSX
  cd data/
  [ ! -e train-images-idx3-ubyte ] && curl -o - http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz | gunzip > train-images-idx3-ubyte
  [ ! -e train-labels-idx1-ubyte ] && curl -o - http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz | gunzip > train-labels-idx1-ubyte
  [ ! -e t10k-images-idx3-ubyte ] && curl -o - http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz | gunzip > t10k-images-idx3-ubyte
  [ ! -e t10k-labels-idx1-ubyte ] && curl -o - http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz | gunzip > t10k-labels-idx1-ubyte
fi

# Code building
cd $SCRIPT_DIR
[ -d build ] && rm -rf build
mkdir build
cd build/
cmake ../
make -j $(getconf _NPROCESSORS_ONLN)
cd examples/
