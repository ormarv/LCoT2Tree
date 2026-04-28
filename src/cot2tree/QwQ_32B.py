#!/usr/bin/env python
from transformers import AutoTokenizer, AutoModelForCausalLM
#model_name = "/lustre/fswork/projects/rech/rqn/ugy38tw/.cache/huggingface/hub/models--bartowski--DeepSeek-R1-Distill-Qwen-32B-GGUF/snapshots/1dc8cf9ffa5dd333057ea1b09ccf4772d8726dec/DeepSeek-R1-Distill-Qwen-32B-Q8_0.gguf"
#model_name = "/lustre/fswork/projects/rech/rqn/ugy38tw/.cache/huggingface/hub/models--unsloth--DeepSeek-R1-Distill-Llama-70B-GGUF/snapshots/732dd974083ea5877d7b6d788b36fe7c2e5eab36/"
model_name = "/lustre/fswork/projects/rech/rqn/ugy38tw/.cache/huggingface/hub/models--Qwen--QwQ-32B/snapshots/976055f8c83f394f35dbd3ab09a285a984907bd0/"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
messages = [
    {"role": "user", "content": "Who are you?"},
]
inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))