# Libraries download
mkdir libs
cd libs/
git clone -b DistributedFF https://github.com/fastflow/fastflow.git
git clone -b v1.3.2 https://github.com/USCiLab/cereal.git
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.0.0+cpu.zip
rm libtorch-cxx11-abi-shared-with-deps-2.0.0+cpu.zip

# Environment variables setting
export TORCH_HOME=$(pwd)/libtorch
export CEREAL_HOME=$(pwd)/cereal/include
export FF_HOME=$(pwd)/fastflow

# Build the dff_run utility
cd fastflow/ff/distributed/loader
make
cd ../../../../../

# MNIST dataset download
mkdir data
cd data
curl -o - http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz | gunzip > train-images-idx3-ubyte
curl -o - http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz | gunzip > train-labels-idx1-ubyte
curl -o - http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz | gunzip > t10k-images-idx3-ubyte
curl -o - http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz | gunzip > t10k-labels-idx1-ubyte
cd ..

# Code building
mkdir build
cd build/
cmake ../
make
cd examples/
