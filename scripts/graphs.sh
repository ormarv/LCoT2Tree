#!/bin/bash
#SBATCH --job-name=makegraphs
#SBATCH --output=/lustre/fswork/projects/rech/rqn/ugy38tw/.local/makegraphs/%x_%j_%a.out
#SBATCH --error=/lustre/fswork/projects/rech/rqn/ugy38tw/.local/makegraphs/%x_%j_%a.err
#SBATCH --array=1,2,3,5,7,8,9,10,11,12,13,14,15,16,18,19,20,21,22,23,24,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,44,45,48,49,50,51,52,53,54,55,57,65,67,68,69,71,72,73,74,75,80,82,83,84,86,87,88,89,90,91,92,93,94,95,96,98,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,144,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,168,169,170,171,172,173,174,178,180,181,182,199,200,201,217,218,219,220,221,224,225,226,227,238,239,240,241,243,246,247,249,264,267,279,281,282,283,284,285,286,288,289,290,291,292,295,299,300,301,304,305,306,307,308,309,310,311,312,313,314,315,316,317,318,319,320,321,322,323,324,325,326,327,328,329,330,331,332,333,334,335,336,337,338,339,340,341,342,343,344,345,346,347,348,349,350,351,352,353,354,355,356,357,358,359,361,362,363,364,365,366,367,368,369,370,371,372,373,374,375,376,377,378,379,380,381,382,383,384,385,386,387,388,389,390,391,392,393,394,395,396,397,398,399,400,401,402,403,404
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -C v100
#SBATCH --gres=gpu:2
#SBATCH --hint=nomultithread
#SBATCH --time=14:00:00
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
srun src/cot2tree/split_lcot_files.py -f $FILE

echo "Job ended at: $(date)"