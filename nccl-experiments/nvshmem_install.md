## Install

wget https://developer.download.nvidia.com/compute/redist/nvshmem/3.0.6/source/nvshmem_src_3.0.6-4.txz
tar -xvf nvshmem_src_3.0.6-4.txz

## Build

### Install 

[optional]
### Download UCX
wget https://github.com/openucx/ucx/releases/download/v1.17.0/ucx-1.17.0.tar.gz
tar -xvf ucx-1.17.0.tar.gz
cd ucx-1.17.0
./configure --prefix=/usr/local/ucx
make -j
sudo make install

[optional]
### Install OpenMPI with UCX support (it also comes with OpenSHMEM support)
wget https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.3.tar.gz
tar -xvf openmpi-5.0.3.tar.gz
cd openmpi-5.0.3
./configure --prefix=/usr/local/openmpi --with-ucx=/usr/local/ucx
make -j
sudo make install

### Set environment variables
export CUDA_HOME=/usr/local/cuda
export NVSHMEM_USE_GDRCOPY=0
export NVSHMEM_MPI_SUPPORT=1
export NVSHMEM_SHMEM_SUPPORT=1
export MPI_HOME=/usr/local/openmpi
export SHMEM_HOME=/usr/local/openmpi
# Set the path to mpicc if different from the default
export MPICC=$MPI_HOME/bin/mpicc
# Set UCX if needed
export NVSHMEM_UCX_SUPPORT=1
export UCX_HOME=/usr/local/ucx
export NVSHMEM_PREFIX=/usr/local/nvshmem-3.0.6
# ensure that the dynamic linker can find the NVSHMEM library (libnvshmem.so) and any other necessary shared libraries.
export NVSHMEM_HOME=/usr/local/nvshmem-3.0.6
export PATH=$NVSHMEM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$MPI_HOME/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH

/usr/local/nvshmem-3.0.6/lib:/usr/local/openmpi/lib

/usr/local/nvshmem-3.0.6/bin:/home/ubuntu/.vscode-server/bin/0ee08df0cf4527e40edc9aa28f4b5bd38bbff2b2/bin/remote-cli:/home/ubuntu/miniconda3/bin:/home/ubuntu/miniconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin

### Build and Install nvshmem
mkdir build
cd build
cmake .. -DCUDA_HOME=$CUDA_HOME -DGDRCOPY_HOME=$GDRCOPY_HOME -DMPI_HOME=$MPI_HOME -DSHMEM_HOME=$SHMEM_HOME -DUCX_HOME=$UCX_HOME -DLIBFABRIC_HOME=$LIBFABRIC_HOME -DNCCL_HOME=$NCCL_HOME -DPMIX_HOME=$PMIX_HOME -DNVSHMEM_USE_GDRCOPY=$NVSHMEM_USE_GDRCOPY -DNVSHMEM_MPI_SUPPORT=$NVSHMEM_MPI_SUPPORT -DNVSHMEM_SHMEM_SUPPORT=1 -DNVSHMEM_UCX_SUPPORT=$NVSHMEM_UCX_SUPPORT -DNVSHMEM_LIBFABRIC_SUPPORT=$NVSHMEM_LIBFABRIC_SUPPORT -DNVSHMEM_USE_NCCL=$NVSHMEM_USE_NCCL -DNVSHMEM_PMIX_SUPPORT=$NVSHMEM_PMIX_SUPPORT -DNVSHMEM_IBGDA_SUPPORT=$NVSHMEM_IBGDA_SUPPORT -DCMAKE_INSTALL_PREFIX=$NVSHMEM_PREFIX
make -j
sudo make install

### To use nvshmem-info
export PATH=/usr/local/nvshmem-3.0.6/bin:$PATH
nvshmem-info -n

## Install nvshmrun
export PATH=/home/ubuntu/hydra_install/bin:$PATH

## Test
# Navigate to the directory where you want to compile the example
cd ~/nvshmem-examples-test
cp /usr/local/nvshmem-3.0.6/examples/hello.cpp .

# Compile hello.cpp manually with the include path for NVSHMEM
nvcc -arch=sm_70 -rdc=true -ccbin g++ -I/usr/local/nvshmem-3.0.6/include -I/usr/local/openmpi/include -I/usr/local/cuda-12.1/include -L/usr/local/nvshmem-3.0.6/lib -L/usr/local/cuda-12.1/lib64 -lnvshmem_host -lnvshmem_device -lcudart -lcuda yue_test.cu -o yue_test

CUDA_VISIBLE_DEVICES=4,5,6,7 nvshmrun -n 4 -ppn 4 ./yue_test

nvcc -arch=sm_70 -rdc=true -ccbin g++ -I/usr/local/nvshmem-3.0.6/include -I/usr/local/cuda-12.1/include -L/usr/local/nvshmem-3.0.6/lib -L/usr/local/cuda-12.1/lib64 -lnvshmem_host -lnvshmem_device -lcudart -lcuda

# Run the compiled binary
./hello

## Others

ubuntu@ip-172-31-28-20:~/yue$ echo $LD_LIBRARY_PATH
/usr/local/nvshmem-3.0.6/lib:/usr/local/openmpi/lib:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/opt/aws-ofi-nccl/lib:/usr/local/cuda-12.1/lib:/usr/local/cuda-12.1/lib64:/usr/local/cuda-12.1:/usr/local/cuda-12.1/targets/x86_64-linux/lib/:/usr/local/cuda-12.1/extras/CUPTI/lib64:/usr/local/lib:/usr/lib:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/opt/aws-ofi-nccl/lib:/usr/local/cuda-12.1/lib:/usr/local/cuda-12.1/lib64:/usr/local/cuda-12.1:/usr/local/cuda-12.1/targets/x86_64-linux/lib/:/usr/local/cuda-12.1/extras/CUPTI/lib64:/usr/local/lib:/usr/lib:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/opt/aws-ofi-nccl/lib:/usr/local/cuda-12.1/lib:/usr/local/cuda-12.1/lib64:/usr/local/cuda-12.1:/usr/local/cuda-12.1/targets/x86_64-linux/lib/:/usr/local/cuda-12.1/extras/CUPTI/lib64:/usr/local/lib:/usr/lib


### debug

nvcc -arch=sm_90 -rdc=true -ccbin g++ -I/usr/local/nvshmem-3.0.6/include -I/usr/local/cuda-12.2/include -L/usr/local/nvshmem-3.0.6/lib -L/usr/local/cuda-12.2/lib64 -lnvshmem_host -lnvshmem_device -lcudart -lcuda all_gather_putsize.cu -o nvshmem_gather
