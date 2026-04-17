#!/usr/bin/env python3
# Use a pipeline as a high-level helper

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("unsloth/DeepSeek-R1-Distill-Llama-70B-GGUF")
model = AutoModelForCausalLM.from_pretrained("unsloth/DeepSeek-R1-Distill-Llama-70B-GGUF")
