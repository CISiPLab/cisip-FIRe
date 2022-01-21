# Run on Arm (aarch64)
These procedures were tested on Oracle ARM Ampere (aarch64) instance. We do not test on ARM device with CUDA device.
## Create conda environment and install pytorch
```shell
conda craete -n fire python=3.9
conda activate fire
pip install numpy
pip install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
```
## Requirement to build fairs
First, install prerequisites for building faiss.
```shell
sudo apt-get install libatlas-base-dev libatlas3-base
sudo apt-get install clang-8
sudo apt-get install swig
sudo apt-get install build-essential libblas-dev gfortran liblapack-dev
sudo apt -y install libopenblas-dev libblis-dev
sudo update-alternatives --set libblas.so.3-aarch64-linux-gnu \
    /usr/lib/aarch64-linux-gnu/blis-openmp/libblas.so.3
```
Then install latest cmake:
```shell
wget https://github.com/Kitware/CMake/releases/download/v3.22.1/cmake-3.22.1-linux-aarch64.tar.gz
tar -xzf cmake-3.22.1-linux-aarch64.tar.gz
mv cmake-3.22.1-linux-aarch64 ~/cmake
alias cmake=~/cmake/bin/cmake
```

## Build faiss for aarch64
```shell
# clone
git clone https://github.com/facebookresearch/faiss.git
cd faiss/
# build
cmake  -B build -DCMAKE_CXX_COMPILER=clang++-8 -DFAISS_ENABLE_GPU=OFF  -DPython_EXECUTABLE=$(which python3) -DFAISS_OPT_LEVEL=generic -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DBLA_VENDOR=OpenBLAS
make -C build -j faiss
make -C build -j swigfaiss
# install python
(cd build/faiss/python && python setup.py install)
# install C headers (required for test)
sudo make -C build install
# testing
make -C build test
conda install pytest scipy numpy
PYTHONPATH="$(ls -d ./build/faiss/python/build/lib*/)" pytest tests/test_*.py
### note that test_binary might failed. But should be okay
make -C build demo_ivfpq_indexing
```

## Install dependencies
Now install the dependencies for `cisip-FIRe`.
```shell
conda install scipy scikit-learn tqdm matplotlib pyyaml pandas seaborn flask psutil
sudo apt-get install ffmpeg libsm6 libxext6  -y
pip install pytorch_memlab opencv-python kornia wandb
```

## Done
Now you can use the framework as usual.