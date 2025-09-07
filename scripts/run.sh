
###########################################################################################################################
# Run all lighteval tasks

MODEL=DeepSeek-R1-Distill-Qwen-7B
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.9,generation_parameters={max_new_tokens:32768,temperature:1.0,top_p:0.95}"
OUTPUT_DIR=data/evals/$MODEL
TREE_DIR=TREE_DIR
CUDA=4

export VLLM_WORKER_MULTIPROC_METHOD=spawn 

echo "cuda:$CUDA" 


TASK=math_500_multi
CUDA_VISIBLE_DEVICES=$CUDA  python lighteval/src/run.py \
    --model_args $MODEL_ARGS \
    --tasks "custom|$TASK|0|0" \
    --custom_tasks ./src/evaluation/evaluate_multi.py \
    --use_chat_template \
    --save_details \
    --output_dir $OUTPUT_DIR &

###########################################################################################################################
# Preprocess output file

python src/utils/preprocess.py $OUTPUT_DIR

###########################################################################################################################
# Long CoT Analysis
input_path=$OUTPUT_DIR
tree_dir=$TREE_DIR

mkdir -p $tree_dir
mkdir -p $tree_dir/trees

# split cot into multi thoughts

python src/cot2tree/split_thought.py $input_path $tree_dir

# # extract reasoning sketch from long cot

python src/cot2tree/extract_sketch.py $tree_dir atom

# assign thought to sketch

python src/cot2tree/assign_step.py $tree_dir

# assign function to thought

python src/cot2tree/assign_function.py $tree_dir

# build tree from extracted information

python src/cot2tree/build_tree.py $tree_dir


###########################################################################################################################
# Train GNN

python src/gnn/train_gin.py $tree_dir final.json 0,2,5,7,8 111 42 base 

python src/gnn/explain_gin.py $tree_dir final.json 1 0,2,5,7,8 111 42 base 

