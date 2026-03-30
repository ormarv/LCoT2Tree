#!/bin/bash
#SBATCH --job-name=QwQ_test         # Name of your job
#SBATCH --output=test_runs/%x_%j.out            # Output file (%x for job name, %j for job ID)
#SBATCH --error=test_runs/%x_%j.err             # Error file
#SBATCH --nodes=1
#SBATCH --gpus=1                     
#SBATCH --time=00:05:00   
#SBATCH --account=rqn@a100
#SBATCH -C a100            
# Print job details
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
chmod +x src/cot2tree/QwQ_32B.py
srun src/cot2tree/QwQ_32B.py
#srun LLM-MindMap/edge_classification.py
# Print job completion time
echo "Job finished at: $(date)"