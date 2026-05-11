#!/bin/bash
#SBATCH --job-name=cpp
#SBATCH --output=/lustre/fswork/projects/rech/rqn/ugy38tw/.local/cpp/%x_%j_%a.out
#SBATCH --error=/lustre/fswork/projects/rech/rqn/ugy38tw/.local/cpp/%x_%j_%a.err
#SBATCH --array=0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -C h100
#SBATCH --gres=gpu:4
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
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

# We get the arguments for the Python script.
line_N=$( sed -n "${SLURM_ARRAY_TASK_ID:-1}p" scripts/input_file.txt)
read -r dataset model n_samples n_iterations <<< "$line_N"
# 5. Run the script
echo $dataset
echo $model
echo $n_samples
echo $n_iterations
chmod +x src/cot2tree/get_questions_dsr-distill-Q32B.py
srun src/cot2tree/get_questions_dsr-distill-Q32B.py -d "$dataset" -m "$model" -s "$n_samples" -i "$n_iterations"

echo "Job ended at: $(date)"