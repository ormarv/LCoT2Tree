from typing import List, Dict
import networkx as nx
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm

from cot2tree.language_models import  NLI_client

MODEL_ID = "/lustre/fswork/projects/rech/rqn/ugy38tw/.cache/huggingface/hub/models--MoritzLaurer--DeBERTa-v3-base-mnli-fever-docnli-ling-2c/snapshots/eff31bcd5e3d26a4246264878a14e937cc5d7fc0"


def get_all_new_paths(graph:nx.DiGraph, node:int, logfile=None):
    # suboptimal, we could just add the nodes to the paths of their parents
    #TODO: rewrite
    if node==0:
        return [[0]]
    new_paths = list(nx.all_simple_paths(graph, source=0, target=node))
    print(f"New paths: {new_paths}",file=logfile) 
    return new_paths

def get_path_content(path:List[int],steps:Dict[int,str]):
    path_content = ""
    for node in path:
        path_content = path_content + steps[node]
    return path_content

def get_attachment_pool(new_paths:Dict[int,Dict],last_node:int,leaves, main_branch, logfile=None):
    # leaves are a set of integers
    attachment_pool = set()
    #print(f"Starting attachment pool, the new paths are {new_paths}, the former leaves are {leaves}.")
    for i,node in enumerate(main_branch):
        # the main branch doesn't contain a leaf
        attachment_pool.add(node)
    # we find and take out the former leaves
    intersection = attachment_pool.intersection(leaves)
    for node in intersection:
        if node in leaves:
            leaves.remove(node)
    #print(f"The new leaves are {leaves}.")
    for node in leaves:
        attachment_pool.add(node)
    # we add the current node to the leaves, for next time
    leaves.add(last_node)
    print('\n',file=logfile)
    print(f"The attachment pool for {last_node} is {list(attachment_pool)}",file=logfile)
    print('\n',file=logfile)
    attachment_pool = list(attachment_pool)
    attachment_pool.sort(reverse=True)
    return attachment_pool


def construct_graph(steps:Dict[int,str], threshold:float = 0.7, max_path_length_for_nli=None, k1:float=0.01, k2:float=0.02, t2:float=None, logfile=None, wanted_features:Dict[str,int]=[])->Dict[str,List[str]]:
    """
    Construct a reasoning graph from the list of the steps.
    Params:
    steps: A dictionary that associates the name of the step (i.e. s_i) to its content

    Return:
    A dictionary giving a list of children for each step.
    """
    q1 = int(k1*len(steps))
    q2 = int(k2*len(steps))
    graph = nx.DiGraph()
    paths = {}  # key: int (node index); value: List of paths
    new_paths = []
    main_branch = []
    leaves = set()
    nli_client = NLI_client(MODEL_ID)
    print(f"There are {len(steps)} steps.")
    graph_features = []
    total_nb_words = sum([len(steps[step].split(' ')) for step in steps])
    cumulative_tokens = 0  # nb of words, actually: we separate on whitespace
    for step in tqdm(steps):
        # create the empty features list
        features = [None]*len(wanted_features)
        if 'node_index' in wanted_features:
            features[wanted_features['node_index']] = step
        if 'distance_to_end' in wanted_features:
            distance = (total_nb_words - cumulative_tokens)/total_nb_words
            features[wanted_features['distance_to_end']] = distance
        if 'nb_words_before' in wanted_features:
                features[wanted_features['nb_words_before']] = cumulative_tokens
                cumulative_tokens+=len(steps[step].split(' '))
        print('\n',file=logfile)
        print(f"---------------------------------Inserting step {step}---------------------------------",file=logfile)
        print(f"The step's content is {steps[step]}",file=logfile)
        print('\n',file=logfile)
        graph.add_node(step)
        branch_scores = {}
        attachment_pool = get_attachment_pool(new_paths, step, leaves, main_branch, logfile=logfile)
        while len(attachment_pool)>0:
            node = attachment_pool[0]
            relevant_paths = paths[node]
            is_parent = False
            #print(f"Relevant paths: {relevant_paths}")
            for path in relevant_paths:
                # run NLI model
                if max_path_length_for_nli is not None and len(path)>max_path_length_for_nli:
                    path_content = get_path_content(path[:max_path_length_for_nli],steps)
                else:
                    path_content = get_path_content(path, steps)
                prediction = nli_client.run(premise=path_content, hypothesis=steps[step])
                # get entailment probability
                # add to branch_scores
                branch_scores[tuple(path)] = prediction
                if prediction>=threshold:
                    is_parent = True
                    break
            if is_parent:
                ascendants = set(itertools.chain.from_iterable(relevant_paths))
                for ascendant in list(ascendants):
                    if ascendant in attachment_pool:
                        attachment_pool.remove(ascendant)
            if node in attachment_pool:
                attachment_pool.remove(node)
        #print(f"Branch scores: {branch_scores}")
        # get three highest scored paths (if there are at least three paths)
        sorted_scores = [(key,value) for key,value in sorted(branch_scores.items(), key=lambda item: item[1], reverse=True)]
        if step<q1:
            nb_parents = 1
        elif step<q2:
            nb_parents = 2
        else:
            nb_parents = 3
        if len(sorted_scores)>nb_parents:
            sorted_scores = sorted_scores[:nb_parents]
        # compare their scores to a threshold
        has_parent = False
        for k,v in sorted_scores:
            if v>=threshold:
                has_parent = True
                print(k,file=logfile)
                parent = list(k)[len(k)-1]
                print('\n',file=logfile)
                print(f"Adding edge between {parent} and {step}.",file=logfile)
                print(f"Content of parent ({parent}): {steps[parent]}",file=logfile)
                print('\n',file=logfile)
                graph.add_edge(parent, step)
                # add the number of parents
                if 'nb_parents' in wanted_features:
                    if features[wanted_features['nb_parents']]==None:
                        features[wanted_features['nb_parents']]=0
                    features[wanted_features['nb_parents']] += 1
        if not has_parent and step!=0:
            if 'nb_parents' in wanted_features:
                features[wanted_features['nb_parents']]=1
            print(f"Sorted_scores: {sorted_scores}",file=logfile)
            if t2 is not None and sorted_scores[0][1]>=t2:
                graph.add_edge(sorted_scores[0][0][-1],step)
                print('\n',file=logfile)
                print(f"Adding edge to a semi-default parent ({sorted_scores[0][0][-1]})",file=logfile)
                print(f"Content of semi-default parent ({sorted_scores[0][0][-1]}): {steps[sorted_scores[0][0][-1]]}",file=logfile)
                print('\n',file=logfile)
            else:
                graph.add_edge(sorted_scores[0][0][0], step)
                print('\n',file=logfile)
                print(f"No satisfactory entailment. Adding {sorted_scores[0][0][0]} as parent of {step}",file=logfile)
                print(f"This is the content of the default parent: {steps[sorted_scores[0][0][0]]}",file=logfile)
                print('\n',file=logfile)
        elif not has_parent:
            # this is the case of the root node
            if 'nb_parents' in wanted_features:
                features[wanted_features['nb_parents']]=0
            
            
        # add to highest: what if no path gives satisfactory results?
        dict_graph = nx.to_dict_of_dicts(graph)
        print('\n',file=logfile)
        print(f"The new graph is: {dict_graph}",file=logfile)
        new_paths = get_all_new_paths(graph, step, logfile=logfile)
        #print(f"Printing new paths again: {new_paths}")
        if step not in paths:
            paths[step] = []
        paths[step] += new_paths
        #print(f"Sanity check: {paths[step]}")
        #print(f"New state of paths: {paths}")
        # Need to find the best new path that will be the new active branch for the attachment pool
        if len(sorted_scores)>0:
            main_branch, _ = sorted_scores[0]
        else:
            main_branch = []
        main_branch = list(main_branch)
        graph_features.append(features)
    nx.draw(graph)
    plt.savefig(f'graph/graph_no_max.png')
        
    return graph, graph_features

