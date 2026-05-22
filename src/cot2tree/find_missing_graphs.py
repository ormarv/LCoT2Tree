import os

def find_missing_indexes(tasks_file, graphs_dir):
    # 1. Get the list of files that already exist in the graphs directory
    try:
        existing_files = set(os.listdir(graphs_dir))
    except FileNotFoundError:
        print(f"Error: The directory '{graphs_dir}' does not exist.")
        return

    missing_indexes = []

    # 2. Parse the tasks file line by line
    with open(tasks_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Split by whitespace to separate the index from the filename
            parts = line.split()
            if len(parts) >= 2:
                idx = parts[0]
                filename = parts[1]
                
                # 3. If the filename isn't in the existing graphs, track its index
                if filename not in existing_files:
                    missing_indexes.append(int(idx))

    # 4. Sort and format the output for Slurm
    missing_indexes.sort()
    
    if not missing_indexes:
        print("All tasks are completed! No missing files found.")
        return

    # Join the individual indexes with commas
    slurm_array_str = ",".join(map(str, missing_indexes))
    
    print("\n--- Copy and paste this into your SLURM script ---")
    print(f"#SBATCH --array={slurm_array_str}")
    print("--------------------------------------------------")
    print(f"Total remaining tasks: {len(missing_indexes)}")

if __name__ == "__main__":
    # Update these paths if you run the script from a different directory
    TASKS_FILE = "scripts/tasks.txt"
    GRAPHS_DIR = "../.local/graphs/"
    
    find_missing_indexes(TASKS_FILE, GRAPHS_DIR)