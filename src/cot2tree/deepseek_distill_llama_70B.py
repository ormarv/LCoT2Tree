# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="unsloth/DeepSeek-R1-Distill-Llama-70B-GGUF")
messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe(messages)

