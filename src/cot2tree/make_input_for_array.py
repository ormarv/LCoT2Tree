import os

directory = "../.local/split_lcots/"
mapping_file = "scripts/tasks.txt"

# Get a sorted list of all text files to ensure consistent ordering
files = sorted([f for f in os.listdir(directory) if f.endswith(".txt") and not f.startswith('.')])

print(f"Found {len(files)} files. Writing map to {mapping_file}...")

with open(mapping_file, "w", encoding="utf-8") as f:
    for index, filename in enumerate(files, start=1):
        f.write(f"{index}    {filename}\n")

print("Done! You can check the file using: head tasks.txt")