#!/bin/bash
#SBATCH --job-name=makegraphs
#SBATCH --output=/lustre/fswork/projects/rech/rqn/ugy38tw/.local/cpp/%x_%j_%a.out
#SBATCH --error=/lustre/fswork/projects/rech/rqn/ugy38tw/.local/cpp/%x_%j_%a.err
#SBATCH --array=1
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


module purge

module load miniforge/24.9.0

conda activate /lustre/fswork/projects/rech/rqn/ugy38tw/triplecot

# We get the arguments for the Python script.
INPUT_FILE="scripts/tasks.txt"
TARGET_LINE=$((SLURM_ARRAY_TASK_ID))
LINE=$(sed -n "${TARGET_LINE}p" $INPUT_FILE)
read -r ID FILE <<< "$LINE"

echo "Running Task ID $SLURM_ARRAY_TASK_ID"
echo "Extracted Map ID: $ID"
echo "Processing File: $FILE"
chmod +x src/cot2tree/split_lcot_files.py
#srun src/cot2tree/split_lcot_files.py -f $FILE
srun src/cot2tree/split_lcot_files.py -f "src/cot2tree/lcots.txt"

echo "Job ended at: $(date)"