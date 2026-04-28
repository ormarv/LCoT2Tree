#!/usr/bin/env python3
from datasets import load_dataset
import numpy as np

dataset = load_dataset("csv", data_files="../.cache/huggingface/hub/datasets--OpenStellarTeam--DeltaBench/snapshots/894d233e5beb06b312b29761cc10e10ca5d2588a/Deltabench_v1.csv")["train"]
train_labels = np.random.randint(0, 2, 10)
eval_labels = np.random.randint(0, 2, 10)
subjects = {'sub1':0, 'sub2':1}
test_labels = {subject:np.random.randint(0,2,2) for subject in subjects}
with open("../.local/lcots/train.txt", "w+") as f:
    print("############".join([dataset[i]["long_cot"]+"&&&&&&&&&&&&"+str(train_labels[i]) for i in range(10)]), file=f)
with open("../.local/lcots/eval.txt", "w+") as f:
    print("############".join([dataset[i]["long_cot"]+"&&&&&&&&&&&&"+str(eval_labels[i]) for i in range(10)]), file=f)
for subject in subjects:
    with open(f"../.local/lcots/test_{subject}.txt", "w+") as f:
        print("############".join([dataset[i]["long_cot"]+"&&&&&&&&&&&&"+str(test_labels[subject][i]) for i in range(2)]), file=f)
  
    

