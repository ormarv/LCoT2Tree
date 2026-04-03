#!/usr/bin/env python3
from datasets import Dataset, load_from_disk

#take 30 deltabench samples
dataset = load_from_disk("~/.cache/huggingface/hub/datasets--OpenStellarTeam--DeltaBench/")
print(dataset)

#run split_lcot on each and save results in log files

#make statistics
