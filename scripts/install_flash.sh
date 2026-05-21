# 1. Load Jean Zay's CUDA toolkit modules to provide 'nvcc'
module load cuda/12.1.1 gcc/11.3.0

# 2. Tell the builder where CUDA lives and force the target architecture
export CUDA_HOME=$CUDA_DIR
export TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"

# 3. Double check that nvcc is active now (it should return a path)
which nvcc

# 4. Install using the correct Python 3.10 wheel directly with no isolation
pip install "flash_attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu130torch2.12cxx11abiTRUE-cp310-cp310-linux_x86_64.whl" --no-build-isolation