
# Long Chain-of-Thought Analysis with GNN

This repository contains code for analyzing long chain-of-thought reasoning using Graph Neural Networks.

## Environment Installation

```bash
pip install -U 'volcengine-python-sdk[ark]'

pip install -r requirements.txt
```

## Pipeline

The pipeline consists of the following steps:

1. **Generate Data with LightEval**
   - Configure model and evaluation tasks:
     ```bash
     MODEL=DeepSeek-R1-Distill-Qwen-7B
     MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.9,generation_parameters={max_new_tokens:32768,temperature:1.0,top_p:0.95}"
     OUTPUT_DIR=data/evals/$MODEL
     TASK=math_500_multi
     ```
   - Run LightEval evaluation:
     ```bash
     CUDA_VISIBLE_DEVICES=$CUDA python lighteval/src/run.py \
         --model_args $MODEL_ARGS \
         --tasks "custom|$TASK|0|0" \
         --custom_tasks ./src/evaluation/evaluate_multi.py \
         --use_chat_template \
         --save_details \
         --output_dir $OUTPUT_DIR
     ```
   - This generates model outputs for mathematical reasoning tasks with chain-of-thought generation

2. **Preprocess Output File**
   ```bash
   python src/utils/preprocess.py $OUTPUT_DIR
   ```

3. **Long CoT Analysis**
   - Split CoT into multiple thoughts:
     ```bash
     python src/cot2tree/split_thought.py $input_path $tree_dir
     ```
   - Extract reasoning sketch:
     ```bash 
     python src/cot2tree/extract_sketch.py $tree_dir atom
     ```
   - Assign thoughts to sketch:
     ```bash
     python src/cot2tree/assign_step.py $tree_dir
     ```
   - Assign functions to thoughts:
     ```bash
     python src/cot2tree/assign_function.py $tree_dir
     ```
   - Build tree from extracted information:
     ```bash
     python src/cot2tree/build_tree.py $tree_dir
     ```

4. **Train GNN Model**
   ```bash
   python src/gnn/train_gin.py $tree_dir final.json 0,2,5,7,8 111 42 base
   ```

5. **Explain GNN Results**
   ```bash
   python src/gnn/explain_gin.py $tree_dir final.json 1 0,2,5,7,8 111 42 base
   ```

## Usage

1. Set environment variables:
   ```bash
   TREE_DIR=<your_tree_dir>
   OUTPUT_DIR=<your_output_dir>
   CUDA=<gpu_id>
   ```

2. Configure LightEval (optional):
   - Modify `scripts/run.sh` to change model, task, or generation parameters
   - Add custom tasks in `src/evaluation/evaluate_multi.py`
   - Adjust model configuration in the MODEL_ARGS variable

3. Run the full pipeline:
   ```bash
   bash scripts/run.sh
   ```

