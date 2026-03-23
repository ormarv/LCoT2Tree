#!/bin/bash
#SBATCH --job-name=prm
#SBATCH --output=prm_outputs/%x_%j_%a.out
#SBATCH --error=prm_outputs/%x_%j_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -C v100
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH --account=rqn@v100

echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

# 1. Clear the environment (Good practice!)
module purge

# 3. Load Miniforge to restore conda/mamba to your PATH
# (If 24.9.0 is unavailable, run `module avail miniforge` on the login node to find the latest)
module load miniforge/24.9.0

# 4. Activate your environment
# Note: On Jean Zay, it is generally safer to use `conda activate` even if you install with mamba
mamba activate /lustre/fswork/projects/rech/rqn/ugy38tw/lcot2tree

# Execute the Python script with specific arguments
#srun load_deltabench_gen_reasoning.py

srun src/cot2tree/split_lcot.py
#srun LLM-MindMap/edge_classification.py
# Print job completion time
echo "Job finished at: $(date)"