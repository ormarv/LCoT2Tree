#!/usr/bin/env python3

from vllm import LLM, SamplingParams
import torch
from typing import List
#model_id = "/linkhome/rech/genltc01/ugy38tw/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Llama-70B/snapshots/b1c0b44b4369b597ad119a196caf79a9c40e141e"
#model_id = "/linkhome/rech/genltc01/ugy38tw/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-32B/snapshots/711ad2ea6aa40cfca18895e8aca02ab92df1a746/"

def run_model_with_vLLM(model_id:str, queries:List[str]):
    llm = LLM(
        
    model=model_id,
        
    dtype=torch.bfloat16,
        
    trust_remote_code=True,
        
    quantization="bitsandbytes",
    tensor_parallel_size=2,
    )

    max_model_len = llm.llm_engine.model_config.max_model_len
    params = SamplingParams(max_tokens=max_model_len)
    outputs = llm.generate(queries, params)
    #answer = outputs[0].outputs[0].text
    answers = [output.outputs[0].text for output in outputs]
    return answers