# Load model directly
from transformers import AutoModel
model = AutoModel.from_pretrained("microsoft/deberta-v3-large")
model.save_pretrained("../.cache/huggingface/hub")