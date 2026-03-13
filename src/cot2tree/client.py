# !pip install llama-cpp-python

from llama_cpp import Llama

llm = Llama.from_pretrained(
	repo_id="unsloth/DeepSeek-V3.2-GGUF",
	filename="BF16/DeepSeek-V3.2-BF16-00001-of-00030.gguf",
)

llm.create_chat_completion(
	messages = "No input example has been defined for this model task."
)
