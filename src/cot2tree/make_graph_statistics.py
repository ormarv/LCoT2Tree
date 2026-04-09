#!/usr/bin/env python3
from datasets import Dataset, load_dataset
import random
import os

from split_lcot import build_graph_from_chain

#take 30 deltabench samples
#dataset = load_dataset("csv", split="train", data_files="../.cache/huggingface/hub/datasets--OpenStellarTeam--DeltaBench/snapshots/894d233e5beb06b312b29761cc10e10ca5d2588a/Deltabench_v1.csv")
#print(dataset)
#gen_reasoning = dataset.filter(lambda sample: sample['task_l1']=='general')
#print(len(gen_reasoning))
#random.seed(42)
#indices = [int(random.random()*50) for _ in range(50)]
#print(indices)
#print(type(gen_reasoning))
#print(gen_reasoning[0])
#samples = gen_reasoning.select(indices)
#print(samples)
#run split_lcot on each and save results in log files
#lcots = [sample['long_cot'] for sample in samples]
directories = ["/lustre/fswork/projects/rech/rqn/ugy38tw/.local/max_5_t2_0_5", "/lustre/fswork/projects/rech/rqn/ugy38tw/.local/max_1_no_t2", "/lustre/fswork/projects/rech/rqn/ugy38tw/.local/max_1_t2_0_5"]
"""for directory in directories:
    if not os.path.isdir(directory):
        os.makedirs(directory)"""
"""print("The first one.")
print(lcots[0])
print("And the second one")
print(lcots[1])
print(f"Are samples 0 and 1 identical?: {lcots[0]==lcots[1]}")
print(f"Are samples 0 and 2 identical?: {lcots[0]==lcots[2]}")
print(f"Are samples 0 and 3 identical?: {lcots[0]==lcots[3]}")
print(f"Are samples 0 and 4 identical?: {lcots[0]==lcots[4]}")"""
"""graphs_no_max_no_t2 = [build_graph_from_chain(lcot,max_path_length_for_nli=None, logfile=open(f"/lustre/fswork/projects/rech/rqn/ugy38tw/.local/no_max_no_t2/{i}.txt","w+")) for i, lcot in enumerate(lcots)]
print("Done with 0: No max, no t2.")
graphs_no_max_t2_0_5 = [build_graph_from_chain(lcot, max_path_length_for_nli=None, t2=0.5, logfile=open(f"/lustre/fswork/projects/rech/rqn/ugy38tw/.local/no_max_t2_0_5/{i}.txt","w+")) for i, lcot in enumerate(lcots)]
print("Done with 1: No max, t2 = 0.5.")
graphs_max_5_no_t2 = [build_graph_from_chain(lcot, logfile=open(f"/lustre/fswork/projects/rech/rqn/ugy38tw/.local/max_5_no_t2/{i}.txt","w+")) for i, lcot in enumerate(lcots)]
print("Done with 2: max = 5, no t2.")"""
"""graphs_max_5_t2_0_5 = [build_graph_from_chain(lcot, t2=0.5, logfile=open(f"/lustre/fswork/projects/rech/rqn/ugy38tw/.local/max_5_t2_0_5/{i}.txt","w+")) for i, lcot in enumerate(lcots)]
print("Done with 3: max = 5, t2 = 0.5.")
graphs_max_1_no_t2 = [build_graph_from_chain(lcot, max_path_length_for_nli=1, logfile=open(f"/lustre/fswork/projects/rech/rqn/ugy38tw/.local/max_1_no_t2/{i}.txt","w+")) for i, lcot in enumerate(lcots)]
print("Done with 4: max = 1, no t2")
graphs_max_1_t2_0_5 = [build_graph_from_chain(lcot, max_path_length_for_nli=1, t2=0.5, logfile=open(f"/lustre/fswork/projects/rech/rqn/ugy38tw/.local/max_1_t2_0_5/{i}.txt","w+")) for i, lcot in enumerate(lcots)]
print("Done with 5: max = 1, t2 = 0.5.")"""
#make statistics
# For each directory, get the list of files
# For each file in the list, count the number of "Adding ", "No satisfactory entailment", and "semi-default"
for dir in directories:
    print(f"Directory: {dir}")
    list_edges = []
    list_semi_default_parents = []
    list_default_parents = []
    files = os.listdir(dir)
    for file in files:
        path = os.path.join(dir, file)
        f = open(path,"r")
        content = f.read()
        print(f"Content length: {len(content)}")
        n = content.count("Adding ")
        print(f"Counting the \"Adding \" in {file}: {n}")
        list_edges.append(content.count("Adding "))
        list_semi_default_parents.append(content.count("No satisfactory entailment"))
        list_default_parents.append(content.count("semi-default"))
        f.close()
    print(f"Average number of edges: {sum(list_edges)/len(list_edges)}")
    print(f"Average number of semi-default parents: {sum(list_semi_default_parents)/len(list_semi_default_parents)}")
    print(f"Average number of default parents: {sum(list_default_parents)/len(list_default_parents)}")
