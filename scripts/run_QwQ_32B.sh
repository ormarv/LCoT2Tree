#!/bin/bash
#SBATCH --job-name=testclient         # Name of your job
#SBATCH --output=load_qwq/%x_%j.out            # Output file (%x for job name, %j for job ID)
#SBATCH --error=load_qwq/%x_%j.err             # Error file
#SBATCH --partition=V100              # Partition to submit to (A100, V100, etc.)
#SBATCH --nodes=1       
#SBATCH --gpus=1             
#SBATCH --time=00:05:00               
# Print job details
echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

# Define variables for your job

module load python/3.10.12
echo "echo PATH"
echo $PATH

~/LCoT2Tree/scripts/mywhich -a python
# Activate the environment
source ~/LCoT2Tree/lcot2tree/bin/activate
echo "PATH after venv activation: $PATH"
echo "Homemade which -a python"
chmod +x ~/LCoT2Tree/scripts/mywhich
echo "env"
# env
echo "echo PATH"
echo $PATH
echo "ls -l /home/infres/bjaulmes-22/LCoT2Tree/lcot2tree/bin/py*"
ls -l /home/infres/bjaulmes-22/LCoT2Tree/lcot2tree/bin/py*
echo "which -a python"
which -a python
# Execute the Python script with specific arguments
#srun load_deltabench_gen_reasoning.py
which -a python
echo "which -a pip"
which -a pip
pip list
srun src/cot2tree/QwQ_32B.py
#srun LLM-MindMap/edge_classification.py
rm ~/.cache/libz2_alma_linux/libbz2.so.1.0
# Print job completion time
echo "Job finished at: $(date)"