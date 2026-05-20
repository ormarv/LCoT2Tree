from vllm import LLM, SamplingParams
from typing import Tuple, List
import torch
import torch.nn.functional as F
import numpy as np
import os
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
from typing import List
from scipy.stats import entropy
from itertools import combinations, chain
import json
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
    probabilities = probabilities * token_masks.unsqueeze(-1) # bs, seq_len, num_labels
    
    all_scores_res = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i] # seq_len, num_labels
        positive_probs = sample[sample != 0].view(-1, 2)[:, 1] # valid_tokens, num_labels
        non_zero_elements_list = positive_probs.cpu().tolist()
        all_scores_res.append(non_zero_elements_list)
    return all_scores_res


def run_qwen_prm(model, tokenizer, prompt, reponses, device):
    system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
    messages = [
        {"role":"system", "content":system_prompt},
        {"role":"user", "content":prompt},
        {"role":"assistant", "content":"<extra_0>".join(reponses)+"<extra_0>"}
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

    outputs = model(input_ids=input_ids)
    step_sep_id = tokenizer.encode("<extra_0>")[0]
    token_masks = (input_ids == step_sep_id)
    scores = make_step_rewards(outputs[0], token_masks)[0]
    return scores

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
            added_scores[answer]
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


def tokenize_and_measure(answers:List[str], tokenizer):
    return [len(tokenizer(answer)) for answer in answers]

def load_MATH_500(parent_dir:str)->List[Tuple[str,str]]:
    dataset = load_dataset(os.path.join(parent_dir, ".cache/huggingface/hub/datasets--simplescaling--openaimath/"))
    #print(dataset)
    main = dataset["test"]
    print(main[0])
    samples = [(sample['problem'], sample["answer"])for sample in main]
    return samples

def load_LCB_v6(parent_dir:str)->List[Tuple[str,str]]:
    dataset = load_dataset(os.path.join(parent_dir, ".cache/huggingface/hub/datasets--drproduck--livecodebench-v6/"))
    ds = dataset["train"]+dataset["test"]
    public_test_cases = ds["public_test_cases"]
    metadata = json.loads(ds["metadata"])
    fn_name = None
    if "func_name" in metadata and metadata["func_name"]!="null":
        fn_name = metadata["func_name"]
    inputs = []
    outputs = []
    for x in public_test_cases:
        inputs.append(x["input"])
        outputs.append(x["output"])
    print(ds[0]["starter_code"])
    samples = [(sample['question_content'], {'input_output':json.dumps({'inputs':inputs, 'outputs':outputs, 'fn_name':fn_name})}) for sample in ds]
    return samples

parent_dir = "/".join(os.getcwd().split("/")[:-1])
math_500 = load_MATH_500(parent_dir)
print(math_500[0])
lcb = load_LCB_v6(parent_dir)
print(lcb[0])