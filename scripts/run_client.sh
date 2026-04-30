#!/bin/bash
#SBATCH --job-name=create_synth_data         # Name of your job
#SBATCH --output=job_outputs/%x_%j.out            # Output file (%x for job name, %j for job ID)
#SBATCH --error=job_outputs/%x_%j.err             # Error file
#SBATCH --partition=V100              # Partition to submit to (A100, V100, etc.)
#SBATCH --nodes=1
#SBATCH --gpus=1                     
#SBATCH --time=00:30:00               
# Print job details
echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

# Define variables for your job

module load python/3.10.12

# Activate the environment
source ~/TripleCoT/triplecot2/bin/activate
pip list
which python
# Execute the Python script with specific arguments
#srun load_deltabench_gen_reasoning.py
chmod +x src/cot2tree/main.py
chmod +x src/cot2tree/gatv2.py
srun src/cot2tree/gatv2.py
#srun src/cot2tree/main.py train -g -d ~/.local/graphs -v -F nb_parents nb_children node_index distance_to_end nb_words_before nb_nodes_per_depth
#srun LLM-MindMap/edge_classification.py
# Print job completion time
echo "Job finished at: $(date)"