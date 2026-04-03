#!/usr/bin/env python3
from datasets import Dataset, load_dataset
import random
import os

from split_lcot import build_graph_from_chain

#take 30 deltabench samples
dataset = load_dataset("csv", split="train", data_files="../.cache/huggingface/hub/datasets--OpenStellarTeam--DeltaBench/snapshots/894d233e5beb06b312b29761cc10e10ca5d2588a/Deltabench_v1.csv")
print(dataset)
gen_reasoning = dataset.filter(lambda sample: sample['task_l1']=='general')
print(len(gen_reasoning))
random.seed(42)
indices = [random.random() for _ in range(50)]
print(type(gen_reasoning))
print(gen_reasoning[0])
samples = gen_reasoning.select(indices)
print(samples)
#run split_lcot on each and save results in log files
lcots = [sample['long_cot'] for sample in samples]
graphs_no_max_no_t2 = [build_graph_from_chain(lcot,max_path_length_for_nli=None, logfile=open(f"./no_max_no_t2/{i}.txt","w+")) for i, lcot in enumerate(lcots)]
graphs_no_max_t2_0_5 = [build_graph_from_chain(lcot, max_path_length_for_nli=None, t2=0.5, logfile=open(f"./no_max_t2_0_5/{i}.txt","w+")) for i, lcot in enumerate(lcots)]
graphs_max_5_no_t2 = [build_graph_from_chain(lcot, logfile=open(f"./max_5_no_t2/{i}.txt","w+")) for i, lcot in enumerate(lcots)]
graphs_max_5_t2_0_5 = [build_graph_from_chain(lcot, t2=0.5, logfile=open(f"./max_5_t2_0_5/{i}.txt","w+")) for i, lcot in enumerate(lcots)]
graphs_max_1_no_t2 = [build_graph_from_chain(lcot, max_path_length_for_nli=1, logfile=open(f"./max_1_no_t2/{i}.txt","w+")) for i, lcot in enumerate(lcots)]
graphs_max_1_t2_0_5 = [build_graph_from_chain(lcot, max_path_length_for_nli=1, t2=0.5, logfile=open(f"./max_1_t2_0_5/{i}.txt","w+")) for i, lcot in enumerate(lcots)]

#make statistics
# For each directory, get the list of files
# For each file in the list, count the number of "Adding ", "No satisfactory entailment", and "semi-default"
directories = ["./no_max_no_t2", "./no_max_t2_0_5", "./max_5_no_t2", "./max_5_t2_0_5", "./max_1_no_t2", "./max_1_t2_0_5"]
for dir in directories:
    print(f"Directory: {dir}")
    list_edges = []
    list_semi_default_parents = []
    list_default_parents = []
    files = os.listdir(dir)
    for file in files:
        content = file.read()
        list_edges.append(content.count("Adding "))
        list_semi_default_parents.append(content.count("No satisfactory entailment"))
        list_default_parents.append(content.count("semi-default"))
    print(f"Average number of edges: {sum(list_edges)/len(list_edges)}")
    print(f"Average number of semi-default parents: {sum(list_semi_default_parents)/len(list_semi_default_parents)}")
    print(f"Average number of default parents: {sum(list_default_parents)/len(list_default_parents)}")
