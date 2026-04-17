#!/usr/bin/env python3
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
MODEL_NAME = "cross-encoder/nli-deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class CrossEncoderClient():
    def __init__(self, model_path:str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    def run(self,answer:str, gold_standard:str, threshold:float):
        features = self.tokenizer([answer, gold_standard])
        scores = self.model(**features).logits
        label_mapping = ['contradiction', 'entailment', 'neutral']
        for i, score_max in enumerate(scores.argmax(dim=1)):
            if label_mapping[score_max] == 'entailment' and scores[i][score_max]>=threshold:
                return True
        return False
    
def grade_answers(answers:str, gold_standard:str, model_path:str, threshold:float):
    trimmed_answers = [answer[:-min(max(len(answer)/10,500),len(answer)-1)] for answer in answers]
    cross_client = CrossEncoderClient(model_path=model_path)
    labels = [cross_client.run(answer) for answer in trimmed_answers]
    return labels

        