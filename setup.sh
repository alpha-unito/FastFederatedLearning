mkdir libs
cd libs/

git clone -b DistributedFF https://github.com/fastflow/fastflow.git
git clone -b v1.3.2 https://github.com/USCiLab/cereal.git
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.0.0+cpu.zip
rm libtorch-cxx11-abi-shared-with-deps-2.0.0+cpu.zip

export TORCH_HOME=$(pwd)/libtorch
export Torch_DIR=${TORCH_HOME}/share/cmake
export CEREAL_HOME=$(pwd)/cereal/include
export FF_HOME=$(pwd)/fastflow

cd ..
