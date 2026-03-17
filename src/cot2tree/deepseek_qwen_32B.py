#!/usr/bin/env python3
# !pip install llama-cpp-python

from llama_cpp import Llama

llm = Llama.from_pretrained(
	repo_id="bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF",
	filename="DeepSeek-R1-Distill-Qwen-32B-IQ2_M.gguf",
)
response = llm.create_chat_completion(
	messages = [
		{
			"role": "user",
			"content": "What is the capital of France?"
		}
	]
)

print(response["choices"][0]["message"]["content"])