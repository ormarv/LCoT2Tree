#!/usr/bin/env python3
from readlog import readlog
import os
from tqdm import tqdm
from split_lcot import build_graph_from_chain
from gatv2 import build_features
import networkx as nx
from argparse import ArgumentParser

def split_file(file_path:str,n:int):
    with open(file_path, "r") as f:
        contents = f.read()
        samples = readlog(contents=contents)
        total_length = len(samples)
        print(f"Total: {total_length}")
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

def read_one_file_and_make_graphs(file_path:str, graph_directory:str):
    filename = os.path.basename(file_path)
    graph_filename = os.path.join(graph_directory,filename)
    with open(file_path,"r") as f:
        contents = f.read()
        samples = [(iteration.split("&&&&&&&&&&&&")[0], iteration.split("&&&&&&&&&&&&")[1]) for iteration in contents.split("############")]
    lcots = [lcot for lcot,_ in samples]
    features = ['nb_parents', 'nb_children', 'node_index', 'distance_to_end', 'nb_words_before', 'nb_nodes_per_depth']
    wanted_features = {feature:i for i, feature in enumerate(features)}
    logfile = open("../.local/construction_log_file.txt","a+")
    graphs =  [build_graph_from_chain(lcot=lcot, nb_keywords=8, max_path_length_for_nli=None, logfile=logfile, wanted_features=wanted_features) for lcot in lcots]
    logfile.close()
    graphs_full_features = [(graph, build_features(graph=graph, all_features=features, wanted_features=wanted_features), eval(label)) for (graph,features),(_, label) in zip(graphs,samples)]
    print(f"LCoTs from: {file_path}, printing graphs in {graph_filename}.")
    with open(graph_filename, "w+") as g:
        print("############".join([str(nx.to_dict_of_dicts(graph))+"&&&&&&&&&&&&"+str(features.tolist())+"&&&&&&&&&&&&"+str(label) for graph, features, label in graphs_full_features]),file=g)

parser = ArgumentParser()
parser.add_argument("-f", type=str)
directory = "../.local/split_lcots/"
args = parser.parse_args()
graph_directory = "../.local/graphs/"
file = args.f
#os.path.join(directory, file)
read_one_file_and_make_graphs(file_path=file, graph_directory=graph_directory)