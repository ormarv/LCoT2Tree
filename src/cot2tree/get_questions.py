#!/usr/bin/env python3
from datasets import load_dataset, Dataset
from typing import Dict, List, Tuple
import numpy as np
import itertools
import os
import random
import json
from verify_final_answer import grade_answers
from models_with_vllm import run_model_with_vLLM
from QwQ_32B import run_QwQ32B

def eval_dataset_to_list(dataset:Dataset, nb_samples_per_subj:int, verbose=False)->List[Tuple[str,str]]:
    samples_by_subject = {}
    for sample in dataset:
        subject = sample['subject']
        if subject not in samples_by_subject:
            samples_by_subject[subject] = []
        samples_by_subject[subject].append((sample['question']+"\nPossible answers: "+"\n".join(sample["choices"]), sample['choices'][sample['answer']]))
    indices = [np.random.randint(0,len(samples), nb_samples_per_subj).tolist() for _,samples in samples_by_subject.items()]
    list_samples = []
    #print([subject for subject in samples_by_subject])
    for i, (_, samples) in enumerate(samples_by_subject.items()):
        list_samples.extend(np.array(samples)[indices[i]].tolist())
    return list_samples


def load_MMLU(nb_samples_per_subj:int, parent_dir:str, seed:int=42, verbose=False):
    part_names = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']
    np.random.seed(seed)
    dataset = load_dataset(os.path.join(parent_dir, ".cache/huggingface/hub/datasets--cais--mmlu/"))
    print(dataset)
    train_split = dataset["train"]
    all_train_samples = np.array([(sample['question']+"\nPossible answers: "+"\n".join(sample["choices"]), sample['choices'][sample['answer']]) for sample in train_split])
    indices = np.random.randint(0, len(all_train_samples), nb_samples_per_subj*len(part_names))
    train_samples = all_train_samples[indices].tolist()
    eval_split = dataset["validation"]
    eval_samples = eval_dataset_to_list(eval_split, nb_samples_per_subj)
    test_split = dataset["test"]
    # now we get the list of samples by subject for the test split
    samples_by_subject = {}
    for sample in test_split:
        subject = sample['subject']
        if subject not in samples_by_subject:
            samples_by_subject[subject] = []
        samples_by_subject[subject].append((sample['question']+"\nPossible answers: "+"\n".join(sample["choices"]), sample['choices'][sample['answer']]))
    indices = [np.random.randint(0,len(samples), nb_samples_per_subj).tolist() for _,samples in samples_by_subject.items()]
    #print(len(list(samples_by_subject.items())[0]))
    test_samples = {item[0]:np.array(item[1])[indices[i]].tolist() for i, item in enumerate(samples_by_subject.items())}
    return train_samples, eval_samples, test_samples

def load_live_code_bench(seed:int, parent_dir:str)->List[Tuple[str,str]]:
    np.random.seed(seed)
    dataset = load_dataset(os.path.join(parent_dir, ".cache/huggingface/hub/datasets--PrimeIntellect--LiveCodeBench-v5/"))
    print(dataset)
    train_split = dataset["train"]
    #print(train_split[0])
    print("Prompt")
    print(train_split[0]["prompt"])
    print("Verification_info")
    print(json.loads(train_split[0]["verification_info"]))
    print(type(json.loads(train_split[0]["verification_info"])))
    print("Truth")
    print(json.loads(json.loads(train_split[0]["verification_info"])["ground_truth"])[0]["input"])
    print(type(json.loads(json.loads(train_split[0]["verification_info"])["ground_truth"])[0]["input"]))
    print(json.loads(json.loads(train_split[0]["verification_info"])["ground_truth"])[0]["output"])
    print(type(json.loads(json.loads(train_split[0]["verification_info"])["ground_truth"])[0]["output"]))
    samples = [(sample["prompt"]+"\nInput:"+ json.loads(json.loads(sample["verification_info"])["ground_truth"])[0]["input"], json.loads(json.loads(sample["verification_info"])["ground_truth"])[0]["output"]) for sample in train_split]
    return samples

def load_MMLU_pro(seed:int, parent_dir:str)->List[Tuple[str,str]]:
    np.random.seed(seed)
    dataset = load_dataset(os.path.join(parent_dir, ".cache/huggingface/hub/datasets--TIGER-Lab--MMLU-Pro/"))
    print(dataset)
    test_split = dataset["test"]
    print(test_split[0])
    samples = [(sample['question']+"\nPossible answers: "+"\n".join(sample["options"]), sample["options"][int(sample["answer_index"])])for sample in test_split]
    return samples

def return_shuffle(l):
    random.shuffle(l)
    return l

def load_GPQA(seed:int, parent_dir:str)->List[Tuple[str,str]]:
    np.random.seed(seed)
    main = load_dataset("csv", data_files=os.path.join(parent_dir, ".cache/huggingface/hub/datasets--Idavidrein--gpqa/snapshots/633f5ee89ab8ad4522a9f850766b73f62147ffdd/gpqa_main.csv"))["train"]
    print(main[0])
    samples = [(sample['Question']+"\nPossible answers: "+"\n".join(return_shuffle([sample["Correct Answer"], sample["Incorrect Answer 1"], sample["Incorrect Answer 2"], sample["Incorrect Answer 3"]])), sample["Correct Answer"])for sample in main]
    return samples

def load_MATH(seed:int, parent_dir:str)->List[Tuple[str,str]]:
    np.random.seed(seed)
    dataset = load_dataset(os.path.join(parent_dir, ".cache/huggingface/hub/datasets--simplescaling--openaimath/"))
    print(dataset)
    main = dataset["train"]
    print(main[0])
    samples = [(sample['problem'], sample["answer"])for sample in main]
    return samples

def get_lcots_with_labels(samples, cross_encoder, threshold:float, verbose:bool, nb_samples:int=-1, nb_iterations:int=1):
    if nb_samples != -1:
        nb_samples = min(nb_samples, len(samples))
    print(f"Samples: {samples}")
    # Use choice to avoid the samples[indices] error and get unique samples
    selected_indices = np.random.choice(len(samples), nb_samples, replace=False)
    s = [samples[i] for i in selected_indices]
    print(f"s: {s}")
    lcots = []
    for i, (question, _) in enumerate(s):
        for j in range(nb_iterations):
            lcots.append("1 quadruplet and 2 identical doublets")
            lcots.append("The answer is B.")
            lcots.append("The answer is C.")
            # Model 1
            #lcots.append(run_QwQ32B(question))
            # Model 2
            #lcots.append(run_model_with_vLLM("/linkhome/rech/genltc01/ugy38tw/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Llama-70B/snapshots/b1c0b44b4369b597ad119a196caf79a9c40e141e", query=question))
            # Model 3
            #lcots.append(run_model_with_vLLM(model_id="linkhome/rech/genltc01/ugy38tw/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-32B/snapshots/711ad2ea6aa40cfca18895e8aca02ab92df1a746", query=question))
    
    # Correctly repeat gold answers to match lcots length
    golds = list(itertools.chain.from_iterable([[gold] * 3 * nb_iterations for _, gold in s]))
    print(lcots)
    print(golds)
    labels = grade_answers(answers=lcots, gold_standard=golds, model_path=cross_encoder, threshold=threshold, verbose=verbose)
    
    # Separate the results
    correct_data = [(ans, lab) for ans, lab in zip(lcots, labels) if lab]
    incorrect_data = [(ans, lab) for ans, lab in zip(lcots, labels) if not lab]
    print(f"Correct: {correct_data}")
    print(f"Incorrect: {incorrect_data}")
    # Balancing
    if len(correct_data) > len(incorrect_data):
        np.random.shuffle(correct_data)
        correct_data = correct_data[:len(incorrect_data)]
    elif len(incorrect_data) > len(correct_data):
        np.random.shuffle(incorrect_data)
        incorrect_data = incorrect_data[:len(correct_data)]
    if len(correct_data)>1000:
        correct_data = correct_data[:1000]
    if len(incorrect_data)>1000:
        incorrect_data = incorrect_data[:1000]
    balanced_results = correct_data + incorrect_data
    if not balanced_results:
        return list(zip([], []))
        
    final_lcots, final_labels = zip(*balanced_results)
    return list(zip(final_lcots, final_labels))

def split(data):
    data = list(data)
    random.shuffle(data)
    total_count = len(data)
    test_count = int(total_count * 0.20)
    remainder_count = total_count - test_count
    eval_count = int(remainder_count * 0.10)
    test_set = data[:test_count]
    eval_set = data[test_count : test_count + eval_count]
    train_set = data[test_count + eval_count:]
    return train_set, eval_set, test_set

MODEL_NAME = "/linkhome/rech/genltc01/ugy38tw/.cache/huggingface/hub/models--cross-encoder--nli-deberta-v3-base/snapshots/6c749ce3425cd33b46d187e45b92bbf96ee12ec7/"

parent_dir = "/".join(os.getcwd().split("/")[:-1])
mmlu_pro = load_MMLU_pro(seed=42, parent_dir=parent_dir)
gpqa =load_GPQA(42, parent_dir)
lcb = load_live_code_bench(42, parent_dir)
math = load_MATH(42, parent_dir)
mmlu_lcots = get_lcots_with_labels(mmlu_pro, MODEL_NAME, 0.7, True, nb_samples=2)
math_lcots = get_lcots_with_labels(math, MODEL_NAME, 0.7, True, nb_samples=2)
# for lcb, we need 3 iterations for each model, and 2 for qpqa
lcb_lcots = get_lcots_with_labels(lcb, MODEL_NAME, 0.6, True, nb_samples=2)
gpqa_lcots = get_lcots_with_labels(gpqa, MODEL_NAME, 0.6, True, nb_samples=2)
print(f"MMLU lcots: {mmlu_lcots}")
train_mmlu, eval_mmlu, test_mmlu = split(mmlu_lcots)
train_math, eval_math, test_math = split(math_lcots)
train_lcb, eval_lcb, test_lcb = split(lcb_lcots)
train_gpqa, eval_gpqa, test_gpqa = split(gpqa_lcots)
train_split = train_mmlu+train_math+train_lcb+train_gpqa
eval_split = eval_mmlu+eval_math+eval_lcb+eval_gpqa
test_split = test_mmlu+test_math+test_lcb+test_gpqa
#train_samples, eval_samples, test_samples = load_MMLU(3)
#print(len(train_samples))
#print(train_samples)
#print(len(eval_samples))
#print([len(test_samples[subject]) for subject in test_samples])
#print(len(test_samples['public_relations']))
#print(train_samples[0])
