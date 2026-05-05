#!/usr/bin/env python3
from bitsandbytes import BitsAndBytesConfig
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "models--unsloth--DeepSeek-R1-Distill-Llama-70B-GGUF/snapshots/732dd974083ea5877d7b6d788b36fe7c2e5eab36/"
quantization_config = BitsAndBytesConfig(load_in_4bit=True)

quantized_model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto", torch_dtype=torch.bfloat16, quantization_config=quantization_config)

tokenizer = AutoTokenizer.from_pretrained(model_id)
input_text = "What are we having for dinner?"
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

output = quantized_model.generate(**input_ids, max_new_tokens=10)

print(tokenizer.decode(output[0], skip_special_tokens=True))
