#!/usr/bin/env python3
from datasets import load_dataset
import numpy as np

dataset = load_dataset("csv", data_files="~/.cache/huggingface/hub/datasets--OpenStellarTeam--DeltaBench/snapshots/894d233e5beb06b312b29761cc10e10ca5d2588a/Deltabench_v1.csv")["train"]
train_labels = np.random.randint(0, 2, 10)
eval_labels = np.random.randint(0, 2, 10)
features = ['nb_parents', 'nb_children', 'node_index', 'distance_to_end', 'nb_words_before', 'nb_nodes_per_depth']
test_labels = {feature:np.random.randint(0,2,2) for feature in features}
with open("~/.local/lcots/train.txt", "w+") as f:
    print("############".join([dataset[i]["long_cot"]+"&&&&&&&&&&&&"+str(train_labels[i]) for i in range(10)]))
with open("~/.local/lcots/eval.txt", "w+") as f:
    print("############".join([dataset[i]["long_cot"]+"&&&&&&&&&&&&"+str(eval_labels[i]) for i in range(10, 20)]))
for feature in features:
    with open(f"~/.local/lcots/test_{feature}.txt", "w+") as f:
        print("############".join([dataset[i]["long_cot"]+"&&&&&&&&&&&&"+str(train_labels[i]) for i in range(10)]))
  
    

