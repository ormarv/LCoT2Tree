from datasets import load_dataset

ds = load_dataset("PrimeIntellect/LiveCodeBench-v5")
print(ds["train"][0])