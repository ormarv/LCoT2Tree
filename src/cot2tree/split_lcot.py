#!/usr/bin/env python3
import re
import string
from graph_construction import construct_graph
from lcot_examples import LCOT1, LCOT2
import networkx as nx

def length_regularity(steps):
    lengths = {0:0, 10:0, 20:0, 30:0, 40:0, 50:0, 60:0, 70:0, 80:0, 90:0, 100:0, 110: 0, 120:0, 130:0, 140:0, 150:0, 160:0, 170:0, 180:0, 190:0, 200:0}
    for i,step in steps.items():
        q = len(step.split(' '))//10
        if q>=20:
            #print(f"Huge step:{step}")
            lengths[200]+=1
        else:
            lengths[q*10]+=1
    print(lengths)

def contains_alphanumeric(separator:str)->bool:
    alnum = set(string.ascii_letters+string.digits)
    intersection = alnum.intersection(set(separator))
    if len(intersection)>0:
        return True
    return False

def contains_letters(separator:str)->bool:
    intersection = set(string.ascii_letters).intersection(set(separator))
    if len(intersection)>0:
        return True
    return False

def intelligent_split(lcot:str, n_first:int):
    first_words = {}
    raw_steps = lcot.split("\n\n")
    print(f"Number of raw steps: {len(raw_steps)}")
    #length_regularity(raw_steps)
    for raw_step in raw_steps:
        words = raw_step.split(' ')
        if contains_letters(words[0]):
            fw = words[0].replace(',',' ')
            if fw.strip()=="I":
                fw = r'[\.;:\?\!\n]\s+I'
            if fw not in first_words:
                first_words[fw] = 0
            first_words[fw] += 1
    print(f"There are {len(first_words)} prefixes.")
    print(first_words)
    sorted_words = [k for k,_ in sorted(first_words.items(), key=lambda item: item[1], reverse=True)]
    #print(sorted_words)
    keywords = sorted_words[:n_first]
    #print(f"Keywords: {keywords}")
    augmented_keywords = []
    for keyword in keywords:
        augmented_keywords.append(keyword+" ")
        augmented_keywords.append(keyword+",")
        capitalized = keyword.capitalize()
        augmented_keywords.append(capitalized+" ")
        augmented_keywords.append(capitalized+",")
    print(augmented_keywords)
    string = '|'.join(augmented_keywords)
    steps = re.finditer(string,lcot)
    split_indices = []
    for match in steps:
        start = match.start()
        if " I," in match.group() or " I " in match.group() or "\nI " in match.group() or "\nI," in match.group() and match.start()<len(string)-1:
            start+=1
        split_indices.append(start)
    #split_indices = [match.start() for match in steps]
    start_indices = [0]+split_indices[:-1]
    end_indices = split_indices+[len(split_indices)]
    all_indices = zip(start_indices,end_indices)
    full_steps = []
    for (i,j) in all_indices:
        full_steps.append(lcot[i:j])
    return full_steps



steps = intelligent_split(LCOT2,8)
print(len(steps))
steps = {i:step for i,step in enumerate(steps)}
length_regularity(steps)
#graph = construct_graph(steps=steps)
#dict_graph = nx.to_dict_of_dicts(graph)
#print(dict_graph)