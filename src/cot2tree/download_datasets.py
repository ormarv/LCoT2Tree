# Load model directly
from transformers import AutoModelForSequenceClassification, AutoTokenizer
model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.save_pretrained("../.cache/huggingface/hub")