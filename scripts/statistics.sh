#!/bin/bash
#SBATCH --job-name=statistics
#SBATCH --output=/lustre/fswork/projects/rech/rqn/ugy38tw/.local/statistics/%x_%j_%a.out
#SBATCH --error=/lustre/fswork/projects/rech/rqn/ugy38tw/.local/statistics/%x_%j_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH -C v100
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --time=06:00:00
#SBATCH --account=rqn@v100

echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

# 1. Clear the environment (Good practice!)
module purge

# 3. Load Miniforge to restore conda/mamba to your PATH
# (If 24.9.0 is unavailable, run `module avail miniforge` on the login node to find the latest)
module load miniforge/24.9.0
#module load cuda/12.4.1
# 4. Activate your environment
# Note: On Jean Zay, it is generally safer to use `conda activate` even if you install with mamba
conda activate /lustre/fswork/projects/rech/rqn/ugy38tw/triplecot

# 5. Run your script with the '-u' (unbuffered) flag
chmod +x src/cot2tree/make_graph_statistics.py
srun src/cot2tree/make_graph_statistics.py
#echo $LLAMA_CPP_LIB
#export LLAMA_CPP_LIB=/linkhome/rech/genltc01/ugy38tw/triplecot/lib/python3.10/site-packages/llama_cpp/lib/libllama.so
#chmod +x src/cot2tree/language_models.py
#ls /lustre/fswork/projects/rech/rqn/ugy38tw/triplecot/lib/python3.10/site-packages/llama_cpp/lib/
#srun src/cot2tree/language_models.py
