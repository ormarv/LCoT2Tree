#!/usr/bin/env python3
#from models_with_vllm import run_model_with_vLLM
from vllm import LLM, SamplingParams
from datasets import load_dataset, Dataset
from typing import Dict, List, Tuple
import numpy as np
import itertools
import os
import random
import json
from verify_final_answer import grade_answers
from QwQ_32B import run_QwQ32B
from vllm.distributed.parallel_state import destroy_model_parallel
import torch
import gc
parent_dir = "/".join(os.getcwd().split("/")[:-1])
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

def run_model_with_vLLM(model_id:str, queries:List[str]):
    llm = LLM(
        
    model=model_id,
        
    dtype=torch.bfloat16,
        
    trust_remote_code=True,
        
    quantization="bitsandbytes",
    tensor_parallel_size=4,
    gpu_memory_utilization=0.9,
    kv_cache_dtype="fp8",
    enable_chunked_prefill=True
    )

    max_model_len = llm.llm_engine.model_config.max_model_len
    params = SamplingParams(max_tokens=max_model_len)
    print(f"Calling the model: {model_id}")
    outputs = []
    for query in queries:
        o = llm.generate(query, params)
        outputs.append(o)
    #outputs = llm.generate(queries, params)
    #answer = outputs[0].outputs[0].text
    answers = [output.outputs[0].text for output in outputs]
    return answers

def get_lcots(samples, nb_samples:int=-1, nb_iterations:int=1):
    if nb_samples != -1:
        nb_samples = min(nb_samples, len(samples))
    print(f"Samples: {samples}")
    # Use choice to avoid the samples[indices] error and get unique samples
    selected_indices = np.random.choice(len(samples), nb_samples, replace=False)
    s = [samples[i] for i in selected_indices]
    print(f"s: {s}")
    lcots = []
    # We create a model, then run it nb_iterations times on all samples
    questions = [item[0] for item in s]*nb_iterations
    answers = [item[1] for item in s]*nb_iterations
    
    print(f"Now generating with the Llama 70B.")
    # Model 2
    lcots.extend(run_model_with_vLLM("/linkhome/rech/genltc01/ugy38tw/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Llama-70B/snapshots/b1c0b44b4369b597ad119a196caf79a9c40e141e", queries=questions))

    destroy_model_parallel()
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Now generating with the Qwen32B.")
    # Model 3
    lcots.extend(run_model_with_vLLM(model_id="linkhome/rech/genltc01/ugy38tw/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-32B/snapshots/711ad2ea6aa40cfca18895e8aca02ab92df1a746", queries=questions))
    
    for question in questions:
        lcots.append(run_QwQ32B(question))
    # Correctly repeat gold answers to match lcots length
    golds = answers * 3
    print(lcots)
    print(golds)
    return lcots, golds

def get_labeled_lcots(lcots, golds, cross_encoder, threshold:float, verbose:bool):
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
        print(len(incorrect_data))
        print(incorrect_data)
        print(correct_data)
    if len(correct_data)>1000:
        print("Ping!")
        correct_data = correct_data[:1000]
    if len(incorrect_data)>1000:
        print("Pong!")
        incorrect_data = incorrect_data[:1000]
    balanced_results = correct_data + incorrect_data
    print(f"Balanced results: {balanced_results}")
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

mmlu_pro = load_MMLU_pro(seed=42, parent_dir=parent_dir)
gpqa =load_GPQA(42, parent_dir)
lcb = load_live_code_bench(42, parent_dir)
math = load_MATH(42, parent_dir)
mmlu_lcots, mmlu_answers = get_lcots(mmlu_pro, nb_samples=5)
math_lcots, math_answers = get_lcots(math, nb_samples=5)
# for lcb, we need 3 iterations for each model, and 2 for qpqa
lcb_lcots, lcb_answers = get_lcots(lcb, nb_samples=5)
gpqa_lcots, gpqa_answers = get_lcots(gpqa, nb_samples=5)
"""fin_mmlu_lcots = get_labeled_lcots(mmlu_lcots, mmlu_answers, args.cross_encoder, 0.7, args.verbose, nb_samples=5)
fin_gpqa_lcots = get_labeled_lcots(gpqa_lcots, gpqa_answers, args.cross_encoder, 0.7, args.verbose, nb_samples=5)
fin_lcb_lcots = get_labeled_lcots(lcb_lcots, lcb_answers, args.cross_encoder, 0.7, args.verbose, nb_samples=5)
fin_math_lcots = get_labeled_lcots(math_lcots, math_answers, args.cross_encoder, 0.7, args.verbose, nb_samples=5)
train_mmlu, eval_mmlu, test_mmlu = split(fin_mmlu_lcots)
train_math, eval_math, test_math = split(fin_math_lcots)
train_lcb, eval_lcb, test_lcb = split(fin_lcb_lcots)
train_gpqa, eval_gpqa, test_gpqa = split(fin_gpqa_lcots)
train_samples = train_mmlu+train_math+train_lcb+train_gpqa
eval_samples = eval_mmlu+eval_math+eval_lcb+eval_gpqa
test_samples = {"mmlu":test_mmlu, "gpqa":test_gpqa, "lcb":test_lcb, "math": test_math}
# We save those LCoTs and their labels for potential later use.
if not os.path.isdir(args.lcots_directory):
    if verbose:
        print(f"Did not find directory {args.lcots_directory}. Creating directory.")
    os.mkdir(args.lcots_directory)
path_train = os.path.join(args.lcots_directory,"train.txt")
path_eval = os.path.join(args.lcots_directory, "eval.txt")
path_tests = [os.path.join(args.lcots_directory,"test")+ds+".txt" for ds in ["mmlu","gpqa","lcb","math"]]
with open(path_train, "w+") as f:
    if verbose:
        print(f"Saving train LCoTs to file {path_train}.")
    print("############".join([lcot+"&&&&&&&&&&&&"+str(label) for lcot, label in train_samples]),file=f)
with open(path_eval, "w+") as f:
    if verbose:
        print(f"Saving eval LCoTs to file {path_eval}.")
    print("############".join([lcot+"&&&&&&&&&&&&"+str(label) for lcot, label in eval_samples]),file=f)

with open(path_tests[0], "w+") as f:
    if verbose:
        print(f"Saving MMLU pro test LCoTs in : {f}.")
    print("############".join([lcot+"&&&&&&&&&&&&"+str(label) for lcot, label in test_mmlu]),file=f)
with open(path_tests[1], "w+") as f:
    if verbose:
        print(f"Saving GPQA pro test LCoTs in : {f}.")
    print("############".join([lcot+"&&&&&&&&&&&&"+str(label) for lcot, label in test_gpqa]),file=f)
with open(path_tests[2], "w+") as f:
    if verbose:
        print(f"Saving LCB pro test LCoTs in : {f}.")
    print("############".join([lcot+"&&&&&&&&&&&&"+str(label) for lcot, label in test_lcb]),file=f)
with open(path_tests[3], "w+") as f:
    if verbose:
        print(f"Saving MATH pro test LCoTs in : {f}.")
    print("############".join([lcot+"&&&&&&&&&&&&"+str(label) for lcot, label in test_math]),file=f)
"""

"""MODEL_NAME = "/linkhome/rech/genltc01/ugy38tw/.cache/huggingface/hub/models--cross-encoder--nli-deberta-v3-base/snapshots/6c749ce3425cd33b46d187e45b92bbf96ee12ec7/"

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
print(f"train: {train_split}")
print(f"eval: {eval_split}")
print(f"test: {test_split}")"""
#train_samples, eval_samples, test_samples = load_MMLU(3)
#print(len(train_samples))
#print(train_samples)
#print(len(eval_samples))
#print([len(test_samples[subject]) for subject in test_samples])
#print(len(test_samples['public_relations']))
#print(train_samples[0])
