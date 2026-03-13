#!/usr/bin/env python3
from llama_cpp import Llama

llm = Llama.from_pretrained(
	repo_id="unsloth/DeepSeek-V3.2-GGUF",
	filename="UD-IQ1_S/DeepSeek-V3.2-UD-IQ1_S-00001-of-00004.gguf",
)

llm.create_chat_completion(
	messages = "No input example has been defined for this model task."
)
