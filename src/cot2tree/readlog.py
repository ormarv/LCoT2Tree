def readlog(contents:str):
    lines = contents.split("############")
    samples = []
    for line in lines:
        sample = line.split("&&&&&&&&&&&&")
        if len(sample)==2:
            samples.append((sample[0], sample[1]))
            print(f"An a priori correct sample: {sample}")
        elif len(sample)==3:
            print("A side-effect of the append mode.")
            print(f"sample[1][:3]: {sample[1][:3]}")
            print(f"Glued sample: {sample}")
            prev_label, next_lcot = sample[1].split('\n',1)
            samples.append((sample[0], prev_label))
            samples.append((next_lcot, sample[2]))
        else:
            print(f"Something else entirely: length is {len(sample)}.")
            print(f"This is the sample: {sample}")
    return samples