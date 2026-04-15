#!/usr/bin/env python3
import re
import string
from io import TextIOWrapper
from graph_construction import construct_graph
from lcot_examples import LCOT1, LCOT2
import networkx as nx
from argparse import ArgumentParser

def length_regularity(steps):
    lengths = {0:0, 10:0, 20:0, 30:0, 40:0, 50:0, 60:0, 70:0, 80:0, 90:0, 100:0, 110: 0, 120:0, 130:0, 140:0, 150:0, 160:0, 170:0, 180:0, 190:0, 200:0}
    for i,step in steps.items():
        q = len(step.split(' '))//10
        if q>=20:
            # print(f"Huge step:{step}")
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

def intelligent_split(lcot:str, n_first:int, logfile:TextIOWrapper):
    first_words = {}
    raw_steps = lcot.split("\n\n")
    print(f"Number of raw steps: {len(raw_steps)}", file=logfile)
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
    print(f"There are {len(first_words)} prefixes.", file=logfile)
    print(first_words, file=logfile)
    sorted_words = [k for k,_ in sorted(first_words.items(), key=lambda item: item[1], reverse=True)]
    #print(sorted_words)
    keywords = sorted_words[:n_first]
    #print(f"Keywords: {keywords}")
    augmented_keywords = []
    for keyword in keywords:
        #augmented_keywords.append(re.escape(keyword)+" ")
        #augmented_keywords.append(re.escape(keyword+","))
        capitalized = keyword.capitalize()
        augmented_keywords.append(re.escape(capitalized+" "))
        augmented_keywords.append(re.escape(capitalized+","))
    print(augmented_keywords, file=logfile)
    string = '|'.join(augmented_keywords)
    #print(f"Regex string: {string}")
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
    current_step = ""
    list_mid_sentence = [',',';',':']
    for (i,j) in all_indices:
        step = lcot[i:j]
        if len(step.split(' '))<= 10:
            
            if step[-1].islower() or step[-1] in list_mid_sentence:
                current_step+=step
                
            elif step[0].islower() or step[0] in list_mid_sentence:
                if current_step=="":
                    full_steps[-1]+=step
                else:
                    current_step+=step
            else:
                if current_step=="":
                    full_steps[-1]+=step
                else:
                    current_step+=step
                print(current_step)
        else:
            current_step+=step
            full_steps.append(current_step)
            current_step = ""
        if current_step!="":
            full_steps.append(current_step)
    return full_steps


def build_graph_from_chain(lcot:str,nb_keywords:int=8,max_path_length_for_nli:int=5, t2:float=None, logfile:TextIOWrapper=None):
    steps = intelligent_split(lcot,nb_keywords,logfile)
    print(type(steps))
    #print(steps[0])
    steps = {i:step for i,step in enumerate(steps)}
    #print(steps[0])
    length_regularity(steps)
    graph = construct_graph(steps=steps, max_path_length_for_nli=max_path_length_for_nli, t2=t2, logfile=logfile)
    dict_graph = nx.to_dict_of_dicts(graph)
    return dict_graph

parser = ArgumentParser(prog="Reasoning graph construction", description="Builds a reasoning graph from a reasoning chain.")
parser.add_argument("-m","--max_path_length_for_nli", type=int, default=None)
parser.add_argument("-t2","--secondary_threshold", type=float, default=None)
args = parser.parse_args()
max_path_length_for_nli = args.max_path_length_for_nli
t2 = args.secondary_threshold