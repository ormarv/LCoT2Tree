#!/bin/bash
#SBATCH --job-name=testclient         # Name of your job
#SBATCH --output=load_qwq/%x_%j.out            # Output file (%x for job name, %j for job ID)
#SBATCH --error=load_qwq/%x_%j.err             # Error file
#SBATCH --partition=cpu              # Partition to submit to (A100, V100, etc.)
#SBATCH --nodes=1
#SBATCH --gpus=1                     
#SBATCH --time=00:05:00               
# Print job details
echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

# Define variables for your job

module load python/3.10.12

# Activate the environment
source ~/TripleCoT/triplecot2/bin/activate

# Execute the Python script with specific arguments
#srun load_deltabench_gen_reasoning.py
srun src/cot2tree/QwQ_32B.py
#srun LLM-MindMap/edge_classification.py
# Print job completion time
echo "Job finished at: $(date)"