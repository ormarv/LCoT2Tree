#!/usr/bin/env python3

from vllm import LLM, SamplingParams
import torch
from typing import List
import os
from get_questions import *
#model_id = "/linkhome/rech/genltc01/ugy38tw/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Llama-70B/snapshots/b1c0b44b4369b597ad119a196caf79a9c40e141e"
#model_id = "/linkhome/rech/genltc01/ugy38tw/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-32B/snapshots/711ad2ea6aa40cfca18895e8aca02ab92df1a746/"
parent_dir = "/".join(os.getcwd().split("/")[:-1])
def run_model_with_vLLM(model_id:str, queries:List[str]):
    llm = LLM(
        
    model=model_id,
        
    dtype=torch.bfloat16,
        
    trust_remote_code=True,
        
    quantization="bitsandbytes",
    tensor_parallel_size=2,
    gpu_memory_utilization=0.8
    )

    max_model_len = llm.llm_engine.model_config.max_model_len
    params = SamplingParams(max_tokens=max_model_len)
    outputs = llm.generate(queries, params)
    #answer = outputs[0].outputs[0].text
    answers = [output.outputs[0].text for output in outputs]
    return answers
mmlu_pro = load_MMLU_pro(seed=42, parent_dir=parent_dir)
gpqa =load_GPQA(42, parent_dir)
lcb = load_live_code_bench(42, parent_dir)
math = load_MATH(42, parent_dir)
answers = run_model_with_vLLM(model_id = "/linkhome/rech/genltc01/ugy38tw/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Llama-70B/snapshots/b1c0b44b4369b597ad119a196caf79a9c40e141e", queries=["Who is the oldest living former French president?","If a basketball and a lead weight of the same size are dropped from 50 meters above the ground, which one arrives first?"])
print(answers)