#!/bin/bash
#SBATCH --job-name=install_flash
#SBATCH --output=/lustre/fswork/projects/rech/rqn/ugy38tw/install_%x.out
#SBATCH --error=/lustre/fswork/projects/rech/rqn/ugy38tw/install_%x.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -C h100
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --time=00:10:00
#SBATCH --account=rqn@h100
# 1. Ensure you are on an H100 node
module purge
module load arch/h100
module load cuda/12.1.1 gcc/11.3.0
module load miniforge/24.9.0

# 2. Activate your env
conda activate /lustre/fswork/projects/rech/rqn/ugy38tw/triplecot

# 3. CRITICAL: Set these variables so the compiler knows what to do
export CUDA_HOME=$CUDA_DIR
export TORCH_CUDA_ARCH_LIST="8.0;9.0" 

# 4. Use the '--no-build-isolation' flag with the actual repository
# This forces it to compile for your SPECIFIC machine using your version of PyTorch
pip install flash-attn --no-build-isolation --no-cache-dir