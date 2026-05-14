#!/bin/bash
#SBATCH --job-name=cpp
#SBATCH --output=/lustre/fswork/projects/rech/rqn/ugy38tw/.local/cpp/%x_%j_%a.out
#SBATCH --error=/lustre/fswork/projects/rech/rqn/ugy38tw/.local/cpp/%x_%j_%a.err
#SBATCH --array=1-4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -C h100
#SBATCH --gres=gpu:4
#SBATCH --hint=nomultithread
#SBATCH --time=00:30:00
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
INPUT_FILE="scripts/input_file.txt"
TARGET_LINE=$((SLURM_ARRAY_TASK_ID + 1))
# 2. Extract the line corresponding to the current Array Task ID
# This skips the header if you have one; if not, it grabs the Nth line.
LINE=$(sed -n "${TARGET_LINE}p" $INPUT_FILE)

# 3. Parse the columns into variables
# Format: ArrayId Dataset LRM N_samples N_iterations
read -r AID DATASET LRM N_SAMPLES N_ITER <<< "$LINE"

# 4. (Optional) Print for debugging to your log file
echo "Running Task ID $SLURM_ARRAY_TASK_ID"
echo "Dataset: $DATASET, LRM: $LRM, Samples: $N_SAMPLES, Iterations: $N_ITER"
chmod +x src/cot2tree/get_questions_dsr-distill-Q32B.py
srun src/cot2tree/get_questions_dsr-distill-Q32B.py -d $DATASET -m $LRM -s $N_SAMPLES -i $N_ITER

echo "Job ended at: $(date)"