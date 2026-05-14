#!/usr/bin/env python3
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict
import torch
import re
from math_verify import LatexExtractionConfig, ExprExtractionConfig, StringExtractionConfig, parse
from math_verify import verify
from lcb_runner.evaluation.testing_util import run_test
from lcb_runner.utils.extraction_utils import extract_test_output_code
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
    
def string_matching(answer:str, gold_standard:str, letter:str):
    print("--------------------------String matching-------------------------------")
    print(f"Answer: {answer}")
    print(f"Gold: {gold_standard}")
    print(f"Letter: {letter}")
    if "boxed{" in answer:
        extracted_answer = answer.split("boxed{")[1].split("}")[0].lower().replace('\n','').replace(' ','')
        lower_gold = gold_standard.lower().replace('\n','').replace(' ','')
        print(f"Extracted answer: {extracted_answer}")
        if lower_gold in extracted_answer:
            return True
        if letter is not None and letter.lower() in extracted_answer:
            return True
    else:
        extracted_answer = answer
        print(f"Extracted answer: {extracted_answer}")
        if gold_standard in extracted_answer:
            return True
        return False
    
def string_matching2(answer:str, gold_standard:str, letter:str):
    if "boxed{" in answer:
        parts = answer.split("boxed{")[-1]
        extracted_answer = parts.rsplit("}", 1)[0]
        lb = extracted_answer.count("{")
        rb = extracted_answer.count("}")
        print(f"Number of left brackets: {lb}")
        print(f"Number of right brackets: {rb}")
    else:
        extracted_answer = answer
    
    def clean(s):
        if not s:
            return ""
        s = s.lower().replace('\n','').replace(' ','')
        return s.strip('.')
    
    clean_extracted = clean(extracted_answer)
    clean_gold = clean(gold_standard)
    clean_letter = letter.lower() if letter else None
    print(f"Extracted: {clean_extracted}")
    print(f"Gold: {clean_gold}")
    print(f"Letter: {clean_letter}")

    if clean_letter:
        if clean_extracted==clean_letter:
            return True
        elif len(answer)>10 and (answer.strip().lower().endswith(f"is{clean_letter}") or answer.strip().lower().endswith(f"answer:{clean_letter}")):
            return True
    
    if clean_extracted==clean_gold:
        return True
    elif clean_gold in clean_extracted and len(clean_gold)>1:
        print("Exception used!")
        return True
    return False

def string_matching3(answer:str, gold_standard:str, letter:str):
    extracted_answer = None
    patterns = [
        r"(?:answer)\s*[:\s]*\(?([A-Z])\)?",
        r"\\boxed\{([A-Z])\}",
        r"\b([A-Z])\b(?:\s*)$"
    ]
    for p in patterns:
        match = re.search(p, answer, re.IGNORECASE)
        if match:
            extracted_answer = match.group(1).upper()
            break
    if extracted_answer and extracted_answer==letter:
        return True
    return False

def grade_math(answer:str, gold_standard:str):
    config = [
        LatexExtractionConfig(),
        ExprExtractionConfig(),
        StringExtractionConfig()
    ]
    result = parse(answer, config)
    is_correct = verify(gold_standard, result)
    return is_correct

def grade_lcb(answer:str, sample):
    code = extract_test_output_code(answer)
    if not code:
        print(f"No code found in {answer}.")
        return False
    results, metadata = run_test(sample, test=code)
    print(f"Results: {results}")
    # Let's assume that results is a list of bools
    for result in results:
        if result is not True:
            return False
    return True
    



def grade_answers(answers:List[str], gold_standard:List[str|Dict], letters:List[str|None], model_path:str, threshold:float, verbose:bool, dataset_n:int):
    if dataset_n<2:
        labels = [string_matching3(answer, gold,letter) for answer, gold, letter in zip(answers, gold_standard, letters)]
    elif dataset_n==2:  #LiveCodeBench
        labels = [grade_lcb(answer, gold) for answer, gold in zip(answers, gold_standard)]
    else:  # MATH
        labels = [grade_math(answer, gold) for answer, gold in zip(answers, gold_standard)]
    return labels
    """trimmed_answers = [
        answer[:-int(min(max(len(answer)//10, 500), len(answer)-1))] 
        if len(answer) > 1000 else answer 
        for answer in answers
    ]
    cross_client = CrossEncoderClient(model_path=model_path)
    labels = [cross_client.run(answer, g, threshold) for answer, g in zip(trimmed_answers, gold_standard)]"""
        