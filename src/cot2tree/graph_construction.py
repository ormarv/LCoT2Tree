from typing import List, Dict
import networkx as nx

from download_datasets import  NLI_client

MODEL_ID = "/lustre/fswork/projects/rech/rqn/ugy38tw/.cache/huggingface/hub/models--MoritzLaurer--DeBERTa-v3-base-mnli-fever-docnli-ling-2c/snapshots/eff31bcd5e3d26a4246264878a14e937cc5d7fc0"


def get_all_new_paths(graph:nx.DiGraph, node:int):
    # suboptimal, we could just add the nodes to the paths of their parents
    #TODO: rewrite
    if node==0:
        return [[0]]
    new_paths = list(nx.all_simple_paths(graph, source=0, target=node))
    print(f"New paths: {new_paths}") 
    return new_paths

def get_path_content(path:List[int],steps:Dict[int,str]):
    path_content = ""
    for node in path:
        path_content = path_content + steps[node]
    return path_content

def get_attachment_pool(new_paths:Dict[int,Dict],last_node:int,leaves):
    # leaves are a set of integers
    attachment_pool = set()
    print(f"Starting attachment pool, the new paths are {new_paths}, the former leaves are {leaves}.")
    for path in new_paths:
        for i,node in enumerate(path):
            # we exclude the leaf of each path
            if i!=len(path)-1:
                attachment_pool.add(node)
    # we find and take out the former leaves
    intersection = attachment_pool.intersection(leaves)
    for node in intersection:
        leaves.remove(node)
    print(f"The new leaves are {leaves}.")
    for node in leaves:
        attachment_pool.add(node)
    # we add the current node to the leaves, for next time
    leaves.add(last_node)
    print(f"The attachment pool for {last_node} is {list(attachment_pool)}")
    return list(attachment_pool)


def construct_graph(steps:Dict[int,str], threshold:float = 0.7)->Dict[str,List[str]]:
    """
    Construct a reasoning graph from the list of the steps.
    Params:
    steps: A dictionary that associates the name of the step (i.e. s_i) to its content

    Return:
    A dictionary giving a list of children for each step.
    """
    graph = nx.DiGraph()
    paths = {}  # key: int (node index); value: List of paths
    new_paths = []
    leaves = set()
    nli_client = NLI_client(MODEL_ID)
    for step in steps:
        print(f"Inserting step {step}")
        graph.add_node(step)
        branch_scores = {}
        attachment_pool = get_attachment_pool(new_paths, step, leaves)
        for node in attachment_pool:
            relevant_paths = paths[node]
            print(f"Relevant paths: {relevant_paths}")
            for path in relevant_paths:
                # run NLI model
                path_content = get_path_content(path, steps)
                prediction = nli_client.run(premise=path_content, hypothesis=steps[step])
                # get entailment probability
                # add to branch_scores
                branch_scores[tuple(path)] = prediction
        print(f"Branch scores: {branch_scores}")
        # get three highest scored paths (if there are at least three paths)
        sorted_scores = [(key,value) for key,value in sorted(branch_scores.items(), key=lambda item: item[1], reverse=True)]
        if len(sorted_scores)>3:
            sorted_scores = sorted_scores[:3]
        # compare their scores to a threshold
        has_parent = False
        for k,v in sorted_scores:
            if v>=threshold:
                has_parent = True
                print(k)
                parent = list(k)[len(k)-1]
                print(f"Adding edge between {parent} and {step}.")
                graph.add_edge(parent, step)
        if not has_parent and step!=0:
            print(f"Sorted_scores: {sorted_scores}")
            graph.add_edge(sorted_scores[0][0][0], step)
            print(f"No satisfactory entailment. Adding {sorted_scores[0][0][0]} as parent of {step}")
        # add to highest: what if no path gives satisfactory results?
        dict_graph = nx.to_dict_of_dicts(graph)
        print(f"The new graph is: {dict_graph}")
        new_paths = get_all_new_paths(graph, step)
        print(f"Printing new paths again: {new_paths}")
        if step not in paths:
            paths[step] = []
        paths[step] += new_paths
        print(f"Sanity check: {paths[step]}")
        print(f"New state of paths: {paths}")
    return graph

