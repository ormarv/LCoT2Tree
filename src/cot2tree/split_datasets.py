import os
from datasets import load_dataset, Dataset
from typing import List, Tuple, Dict, Union
import json

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
        ptc = json.loads(public_test_cases)
        for x in ptc:
            """print("TYPE:", type(x), "VALUE:", repr(x))
            try:
                print(f"x: {x}")
                y = json.loads(x)
                print(f"y: {y}")
            except Exception as e:
                print(f"Handled exception {e}, ignoring malformed sample.")
                continue"""
            inputs.append(x["input"])
            outputs.append(x["output"])
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

def split_and_save_dataset(samples:List[Tuple[str, Union[str, Dict]]], output_dir:str, dataset_name:str):
    chunk_size = 100

    for i in range(0, len(samples), chunk_size):
        chunk = samples[i:i+chunk_size]
        part_idx = i//chunk_size

        file_path = os.path.join(output_dir, f"{dataset_name}_{part_idx}.jsonl")

        with open(file_path, "w+") as f:
            for query, gold in chunk:
                row = {"query":query, "gold":gold}
                f.write(json.dumps(row, ensure_ascii=False)+'\n')
        print(f"Finished printing into {file_path}.")

def retrieve_split_dataset_samples(file_path:str):
    samples = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                samples.append((data["query"], data["gold"]))
    return samples

pwd = "/".join(os.getcwd().split("/")[:-1])
#samples_math = load_MATH_500(pwd)
samples_lcb = load_LCB_v6(pwd)
output_dir = "../.local/split_datasets"
#split_and_save_dataset(samples_math, output_dir, "math")
split_and_save_dataset(samples_lcb, output_dir, "lcb")
