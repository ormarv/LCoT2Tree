#!/bin/bash
#SBATCH --job-name=main
#SBATCH --output=/lustre/fswork/projects/rech/rqn/ugy38tw/.local/main/%x_%j_%a.out
#SBATCH --error=/lustre/fswork/projects/rech/rqn/ugy38tw/.local/main/%x_%j_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -C v100
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --time=00:30:00
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
conda activate /lustre/fswork/projects/rech/rqn/ugy38tw/triplecot

# 5. Run the script
chmod +x src/cot2tree/gatv2.py
srun src/cot2tree/gatv2.py
