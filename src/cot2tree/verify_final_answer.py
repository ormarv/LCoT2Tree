#!/usr/bin/env python3
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List
import torch
MODEL_NAME = "/linkhome/rech/genltc01/ugy38tw/.cache/huggingface/hub/models--cross-encoder--nli-deberta-v3-base/snapshots/6c749ce3425cd33b46d187e45b92bbf96ee12ec7/"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class CrossEncoderClient():
    def __init__(self, model_path:str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    def run(self,answer:str, gold_standard:str, threshold:float):
        print(f"Answer: {answer}; type: {type(answer)}")
        print(f"Gold: {gold_standard}; type: {type(gold_standard)}")
        features = self.tokenizer(answer, gold_standard,return_tensors='pt')
        with torch.no_grad():
            scores = self.model(**features).logits
        probs = torch.softmax(scores, dim=1)[0]
        label_mapping = ['contradiction', 'entailment', 'neutral']
        entailment_prob = probs[1].item()
        predicted_idx = probs.argmax().item()

        if predicted_idx == 1 and entailment_prob >= threshold:
            return True
        return False
        """for i, score_max in enumerate(scores.argmax(dim=1)):
            if label_mapping[score_max] == 'entailment' and scores[i][score_max]>=threshold:
                return True
        return False"""
    
def string_matching(answer:str, gold_standard:str):
    print("--------------------------String matching-------------------------------")
    print(f"Answer: {answer}")
    print(f"Gold: {gold_standard}")
    if "\\boxed{" in answer:
        extracted_answer = answer.split("\\boxed{")[1].split("}")[0].lower().replace('\n','').replace(' ','')
    else:
        extracted_answer = answer.lower().replace('\n','').replace(' ','')
    lower_gold = gold_standard.lower().replace('\n','').replace(' ','')
    if gold_standard in extracted_answer:
        return True
    return False

def grade_answers(answers:List[str], gold_standard:List[str], model_path:str, threshold:float, verbose:bool):
    labels = [string_matching(answer, gold) for answer, gold in zip(answers, gold_standard)]
    return labels
    """trimmed_answers = [
        answer[:-int(min(max(len(answer)//10, 500), len(answer)-1))] 
        if len(answer) > 1000 else answer 
        for answer in answers
    ]
    cross_client = CrossEncoderClient(model_path=model_path)
    labels = [cross_client.run(answer, g, threshold) for answer, g in zip(trimmed_answers, gold_standard)]"""
        