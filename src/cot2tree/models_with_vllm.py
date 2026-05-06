#!/usr/bin/env python3

from vllm import LLM, SamplingParams
import torch
model_id = "/linkhome/rech/genltc01/ugy38tw/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Llama-70B/snapshots/b1c0b44b4369b597ad119a196caf79a9c40e141e"
llm = LLM(
    
model=model_id,
    
dtype=torch.bfloat16,
    
trust_remote_code=True,
    
quantization="bitsandbytes",
)

params = SamplingParams(max_tokens=-1)
output = llm.generate("Bob is twice the age that Alice will be in 5 years, and thrice her current age. How old are Bob and Alice?", params)