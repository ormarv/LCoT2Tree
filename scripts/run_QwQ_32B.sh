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
echo $PATH
module load python/3.10.12
echo "PATH"
echo $PATH
echo "Python"
which python
# Activate the environment
source ~/LCoT2Tree/lcot2tree/bin/activate
echo "PATH"
echo $PATH
echo "ls -l /home/infres/bjaulmes-22/LCoT2Tree/lcot2tree/bin/py*"
ls -l /home/infres/bjaulmes-22/LCoT2Tree/lcot2tree/bin/py*
which python
ls /lib64/libbz2*
ln -s /lib64/libbz2.so.1 ~/.cache/libz2_alma_linux/libbz2.so.1.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.cache/libbz2_alma_linux/
if test -f /lib64/libbz2.so.1; then
    echo "/lib64/libbz2.so.1 found"
fi
if test -f ~/.cache/libz2_alma_linux/libbz2.so.1.0; then
    echo "~/.cache/libz2_alma_linux/libbz2.so.1.0 found"
fi
# Execute the Python script with specific arguments
#srun load_deltabench_gen_reasoning.py
which python
which pip
pip list
/home/infres/bjaulmes-22/LCoT2Tree/lcot2tree/bin/python src/cot2tree/QwQ_32B.py
#srun LLM-MindMap/edge_classification.py
rm ~/.cache/libz2_alma_linux/libbz2.so.1.0
# Print job completion time
echo "Job finished at: $(date)"