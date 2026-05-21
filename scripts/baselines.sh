#!/bin/bash
#SBATCH --job-name=baselines
#SBATCH --output=/lustre/fswork/projects/rech/rqn/ugy38tw/.local/cpp/%x_%j_%a.out
#SBATCH --error=/lustre/fswork/projects/rech/rqn/ugy38tw/.local/cpp/%x_%j_%a.err
#SBATCH --nodes=23
#SBATCH --ntasks-per-node=1
#SBATCH -C h100
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --time=2:00:00
#SBATCH --account=rqn@h100
#SBATCH --qos=qos_gpu_h100-dev

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

LINE_DATA=$(sed -n "${SLURM_ARRAY_TASK_ID}p" jobs.txt)

# Read the parameters directly
FILE=$(echo "$LINE_DATA" | awk '{print $1}')
LRM=$(echo "$LINE_DATA" | awk '{print $2}')
DATASET=$(echo "$LINE_DATA" | awk '{print $3}') # Integer assignment
chmod +x src/cot2tree/baselines.py
srun src/cot2tree/baselines.py -d $DATASET -l $LRM -f $FILE

echo "Job ended at: $(date)"