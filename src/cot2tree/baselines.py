#!/bin/env python3
from vllm import LLM, SamplingParams
from typing import Tuple, List,  Dict
import torch
import torch.nn.functional as F
import numpy as np
import os
from datasets import load_dataset, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
from typing import List
from scipy.stats import entropy
from itertools import combinations, chain
import json
from tqdm import tqdm
import multiprocessing as mp
import shutil
from lcb_runner.evaluation.testing_util import run_test
from lcb_runner.utils.extraction_utils import extract_test_output_code
import tempfile
from argparse import ArgumentParser
from hendryck_cleanup import *
def run_skywork(model, tokenizer, prompt, responses, device):
    conversations = [[{"role":"user", "content":prompt},{"role":"assistant","content":response}] for response in responses]
    tokenized_convs = [tokenizer.apply_chat_template(conv, tokenize=True, return_tensors="pt").to(device) for conv in conversations]
    scores = []
    with torch.no_grad():
        for tok_conv in tokenized_convs:
            scores.append(model(tok_conv).logits[0][0].item())
    return scores

def init_skywork(device):
    model_name = "Skywork/Skywork-Reward-Gemma-2-27B-v0.2"
    rm = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="flash_attention_2",
        num_labels=1,
    )
    rm_tokenizer = AutoTokenizer.from_pretrained(model_name)
    return rm, rm_tokenizer

def init_qwen_prm(device):
    model_name = "Qwen/Qwen2.5-Math-PRM-72B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name, 
        device_map=device, 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval()
    return model, tokenizer

def make_step_rewards(logits, token_masks):
    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1) 
    
    all_scores_res = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i] 
        mask = token_masks[i]
        valid_probs = sample[mask] 
        
        if valid_probs.numel() > 0:
            positive_probs = valid_probs[:, 1]
            non_zero_elements_list = positive_probs.cpu().tolist()
        else:
            non_zero_elements_list = []
            
        all_scores_res.append(non_zero_elements_list)
        
    return all_scores_res


def run_qwen_prm(model, tokenizer, prompt, responses, device):
    all_scores = []
    for response in responses:
        split_response = [step.strip() for step in response.split("\n\n")]
        if not split_response:
            print("No split response!")
            split_response = [""]
        system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
        messages = [
            {"role":"system", "content":system_prompt},
            {"role":"user", "content":prompt},
            {"role":"assistant", "content":"<extra_0>".join(split_response)+"<extra_0>"}
        ]
        conversation_str = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=False
        )

        input_ids = tokenizer.encode(
            conversation_str, 
            return_tensors="pt", 
        ).to(model.device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids)
        step_sep_id = tokenizer.encode("<extra_0>")[0]
        token_masks = (input_ids == step_sep_id)
        extracted_rewards = make_step_rewards(outputs[0], token_masks)
        print(f"Extracted rewards: {extracted_rewards}")
        if extracted_rewards and len(extracted_rewards[0]) > 0:
            score = extracted_rewards[0][-1]
        else:
            print("Scoring problem.")
            score = 0.0
        all_scores.append(score)
    return all_scores

def get_best(scores):
    ind_max = np.argmax(np.array(scores))
    return ind_max

def majority_voting(answers):
    counts = {}
    for answer in answers:
        if answer not in counts:
            counts[answer] = 0
        counts[answer] += 1
    max_answer = max(counts, key=counts.get)
    return max_answer

def weighted_majority_voting(answers, scores):
    added_scores = {}
    for answer, score in zip(answers, scores):
        if answer not in added_scores:
            added_scores[answer] = 0
        added_scores[answer] += score
    max_answer = max(added_scores, key=added_scores.get)
    return max_answer

def length_filtered_vote(answers:List, lengths:List[int], width:int, nb_groups:int, nb_selected_groups:int):
    all_answers = np.unique(np.array(answers)).tolist()
    buckets_combinations = combinations(range(nb_groups), nb_selected_groups)
    entropies = []
    buckets = []
    for j in range(1,nb_groups):
        l = [answer for answer, length in zip(answers, lengths) if length <= width * j and length > width * (j-1)]
        buckets.append(l)
        answer_freq = []
        for answer in all_answers:
            answer_freq.append(len([1 for a in l if a == answer])/len(l))
        entropies.append(entropy(np.array(answer_freq))) 
    sum_entropies = []
    for i, combination in enumerate(buckets_combinations):
        sum_entropies.append(sum([entropies[k] for k in combination]))
    idx_max = np.argmin(np.array(entropies))
    best_combination = buckets_combinations[idx_max]    
    selected_answers = [buckets[bucket_idx] for bucket_idx in best_combination]
    return list(chain.from_iterable(selected_answers))

def laconic(answers:List, lengths:List[int]):
    lac_idx = np.argmin(np.array(lengths))
    return answers[lac_idx]

def tokenize_and_measure(answers:List[str], tokenizer, max_model_len:int):
    batch_tokenize = tokenizer(
        answers,
        padding=True,
        truncation=True,
        max_length=max_model_len,

    )
    return [len(toks) for toks in batch_tokenize]

def load_MATH_500(parent_dir:str)->List[Tuple[str,str]]:
    dataset = load_dataset(os.path.join(parent_dir, ".cache/huggingface/hub/datasets--simplescaling--openaimath/"))
    #print(dataset)
    main = dataset["test"]
    print(main[0])
    samples = [(sample['problem'], sample["answer"])for sample in main]
    return samples

def _load_lcb_split(dataset:Dataset):
    samples = []
    for sample in dataset:
        public_test_cases = sample["public_test_cases"]
        metadata = json.loads(sample["metadata"])
        fn_name = None
        if "func_name" in metadata and metadata["func_name"]!="null":
            fn_name = metadata["func_name"]
        inputs = []
        outputs = []
        for x in public_test_cases:
            y = json.loads(x)
            inputs.append(y["input"])
            outputs.append(y["output"])
        samples.append((sample['question_content'], {'input_output':json.dumps({'inputs':inputs, 'outputs':outputs, 'fn_name':fn_name})}))
    return samples

def load_LCB_v6(parent_dir:str)->List[Tuple[str,str]]:
    dataset = load_dataset(os.path.join(parent_dir, ".cache/huggingface/hub/datasets--drproduck--livecodebench-v6/"))
    ds1 = dataset["train"]
    ds2 = dataset["test"]
    s1 = _load_lcb_split(ds1)
    s2 = _load_lcb_split(ds2)
    samples = s1 + s2
    print(samples[0])
    return samples

def initialize_model_with_vLLM(model_id:str):
    llm = LLM(
        
    model=model_id,
        
    dtype=torch.bfloat16,
        
    trust_remote_code=True,
        
    quantization="fp8",
    tensor_parallel_size=4,
    gpu_memory_utilization=0.9,
    kv_cache_dtype="fp8",
    enable_chunked_prefill=True
    )
    max_model_len = llm.llm_engine.model_config.max_model_len
    params = SamplingParams(max_tokens=max_model_len, temperature=0.6, top_p=0.95, stop=["<|end_of_text|>", "<|eot_id|>"])
    tokenizer = llm.get_tokenizer()
    return llm, params, tokenizer, max_model_len

def run_with_vLLM(llm:LLM, params:SamplingParams, queries:List[str], tokenizer)->List[str]:
    
    formatted_queries = []
    for query in queries:
        message = [{"role":"user","content":query}]
        prompt = tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
        formatted_queries.append(prompt)
    outputs = llm.generate(formatted_queries, params)
    answers = [output.outputs[0].text for output in outputs]
    return answers

def grade_math(answer:str, gold_standard:str):
    cleaned_answer = last_boxed_only_string(answer)
    rm_ans = remove_boxed(cleaned_answer)
    equiv = is_equiv(rm_ans, gold_standard)
    print(f"Answer: {rm_ans}")
    print("\n")
    #print(f"Result: {type(result)}")
    print(f"Gold: {gold_standard}")
    return equiv

def worker(queue, samp, test_code, temp_dir):
    try:
        os.chdir(temp_dir)
        results, metadata = run_test(samp, test=test_code)
        queue.put((results, metadata))
    except Exception as e:
        queue.put(([False], str(e)))

def run_test_isolated(sample, code, timeout=60):
    context = mp.get_context("spawn")

    q = context.Queue()
    temp_dir = tempfile.mkdtemp()
    p = context.Process(target=worker, args=(q, sample, code, temp_dir))
    p.start()
    p.join(timeout=timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        shutil.rmtree(temp_dir, ignore_errors=True)
        return [False], {"error":"Timeout or Infinite Loop"}
    
    if not q.empty():
        return q.get()
    else:
        return [False], {"error":"Process crashed silently"}

def grade_lcb(answer:str, sample):
    code = extract_test_output_code(answer)
    print(f"Code: {code}\n")
    if not code:
        print(f"No code found.")
        return False
    #str_sample = json.dumps(sample)
    results, metadata = run_test_isolated(sample, code)
    print(f"Results: {results}, Metadata: {metadata}")
    # Let's assume that results is a list of bools
    if not results:
        return False
    for result in results:
        if result is not True:
            return False
    return True

def grade_answers(answers:List[str], gold_standard:List[str|Dict], dataset_n:int):
    if dataset_n==0:
        labels = [grade_lcb(answer, gold) for answer, gold in zip(answers, gold_standard)]
    else:  # MATH
        labels = [grade_math(answer, gold) for answer, gold in zip(answers, gold_standard)]
    return labels

def retrieve_split_dataset_samples(file_path:str):
    samples = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                samples.append((data["query"], data["gold"]))
    return samples

def test_baselines(dataset_n:int, lrm_n:int, N:int, filepath:str, split:int):
    ds_names = ["lcb","math"]
    lrm_names = ["llama", "qwen", "qwq"]
    samples = retrieve_split_dataset_samples(filepath)
    if lrm_n==0:
        model, params, tokenizer, max_model_len = initialize_model_with_vLLM("/linkhome/rech/genltc01/ugy38tw/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Llama-70B/snapshots/b1c0b44b4369b597ad119a196caf79a9c40e141e")
    elif lrm_n==1:
        model, params, tokenizer, max_model_len = initialize_model_with_vLLM("/linkhome/rech/genltc01/ugy38tw/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-32B/snapshots/711ad2ea6aa40cfca18895e8aca02ab92df1a746")
    else:
        model, params, tokenizer, max_model_len = initialize_model_with_vLLM("/lustre/fswork/projects/rech/rqn/ugy38tw/.cache/huggingface/hub/models--Qwen--QwQ-32B/snapshots/976055f8c83f394f35dbd3ab09a285a984907bd0/")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    correct_answers_per_model = {"skywork":0,"qwenprm":0,"laconic":0}
    total_samples = 0
    skywork, skywork_tokenizer = init_skywork(device=device)
    qwen_prm, qwen_tokenizer = init_qwen_prm(device=device)
    for sample in tqdm(samples):
        multi_sample = [sample[0]]*N
        multi_gold = [sample[1]]*N
        answers = run_with_vLLM(llm=model, params=params, queries=multi_sample, tokenizer=tokenizer)
        eval = grade_answers(answers=answers, gold_standard=multi_gold, dataset_n=dataset_n)
        if len([1 for e in eval if e])>8:
            continue
        total_samples += 1
        
        #truth_values = grade_answers(answers=answers, gold_standard=multi_gold, dataset_n=dataset_n)
        # for each evaluation technique, we need to do evaluate the N answers and pick one
        
        skywork_scores = run_skywork(model=skywork, tokenizer=skywork_tokenizer, prompt=sample[0], responses=answers, device=device)
        ind_best_skywork = get_best(skywork_scores)
        best_skywork = answers[ind_best_skywork]
        
        qwen_scores = run_qwen_prm(model=qwen_prm, tokenizer=qwen_tokenizer, prompt=sample[0], reponses=answers, device=device)
        ind_best_qwen = get_best(qwen_scores)
        best_qwen = answers[ind_best_qwen]
        # The following function might have a problem
        lengths = tokenize_and_measure(answers=answers, tokenizer=tokenizer, max_model_len=max_model_len)
        ind_min = np.argmin(np.array(lengths))
        best_laconic = answers[ind_min]

        # We evaluate the answers
        
        if eval[ind_best_skywork]:
            correct_answers_per_model["skywork"] += 1
        if eval[ind_best_qwen]:
            correct_answers_per_model["qwenprm"] += 1
        if eval[ind_min]:
            correct_answers_per_model["laconic"] += 1
    accuracy_per_baseline = {}
    for baseline in correct_answers_per_model:
        accuracy_per_baseline[baseline] = correct_answers_per_model[baseline]/total_samples
    row = {"correct_answers_count": correct_answers_per_model, "total_samples":total_samples}
    with open(os.path.join("../.local/baseline_scores",f"{ds_names[dataset_n]}_{lrm_names[lrm_n]}_{split}.json"), "w+") as f:
        f.write(json.dumps(row, ensure_ascii=False))
    return accuracy_per_baseline






pwd = "/".join(os.getcwd().split("/")[:-1])
parser = ArgumentParser()
parser.add_argument("-f", type=str, help="The name of the file from which to read the samples, to be added to the directory name.")
parser.add_argument("-l", type=int)
parser.add_argument("-d", type=int)
parser.add_argument("-N",type=int, default=10)
args = parser.parse_args()
split_dataset_directory = "../.local/split_datasets/"
file_path = os.path.join(split_dataset_directory, args.f)
split_id = int(args.f.split('_')[-1].split('.')[0])
accuracy_per_baseline = test_baselines(args.d, args.l, args.N, file_path, split_id)
print(accuracy_per_baseline)