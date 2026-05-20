from readlog import readlog
import os

def split_file(file_path:str,n:int):
    with open(file_path, "r") as f:
        contents = f.read()
        samples = readlog(contents=contents)
        total_length = len(samples)
        q = total_length//n
        print(f"Quotient: {q}")
        r = total_length - q * n
        print(f"Reste: {r}")
        core_path = os.path.basename(file_path).split(".txt")[0]
        print(f"Core path: {core_path}")
        for i in range(1,n+1):
            path = os.path.join("../.local/split_lcots/",core_path+f"_{i}"+".txt")
            print(f"Path: {path}")
            # print [(i-1)*q:i*q]
            split_samples = samples[(i-1)*q:i*q]
            with open(path, "w+") as g:
                print("############".join([lcot+"&&&&&&&&&&&&"+str(int(label)) for lcot, label in split_samples]),file=g)
        # print [n*q:total_length]
        last_samples = samples[n*q:]
        if len(last_samples)>0:
            with open(os.path.join("../.local/split_lcots/",core_path+f"_{n+1}.txt"), "w+") as h:
                print("############".join([lcot+"&&&&&&&&&&&&"+str(int(label)) for lcot, label in last_samples]),file=h)

directory = "../.local/lcots2/"
files = os.listdir(directory)
for file in files:
    if file.endswith(".txt"):
        if "test" in file:
            n = 5
        elif "train" in file:
            n = 50
        else:
            n = 20
        print(f"File: {file}, n: {n}")
        split_file(file_path=os.path.join(directory,file), n=n)