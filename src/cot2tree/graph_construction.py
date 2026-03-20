from typing import List, Dict
import networkx as nx

from download_datasets import  NLI_client

MODEL_ID = "/lustre/fswork/projects/rech/rqn/ugy38tw/.cache/huggingface/hub/models--MoritzLaurer--DeBERTa-v3-base-mnli-fever-docnli-ling-2c/snapshots/eff31bcd5e3d26a4246264878a14e937cc5d7fc0"


def get_all_new_paths(graph:nx.DiGraph, node:int):
    # suboptimal, we could just add the nodes to the paths of their parents
    #TODO: rewrite
    new_paths = nx.all_simple_paths(graph, source=0, target=node)
    print(new_paths) 
    return new_paths  


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
    nli_client = NLI_client(MODEL_ID)
    for step in steps:
        print(f"Inserting step {step}")
        graph.add_node(step)
        branch_scores = {}
        for node in list(graph.nodes):
            if node!=step:
                relevant_paths = paths[node]
                print(f"Relevant paths: {paths}")
                for path in relevant_paths:
                    # run NLI model
                    prediction = NLI_client.run(premise=path, hypothesis=step)
                    # get entailment probability
                    # add to branch_scores
                    branch_scores[tuple(path)] = prediction
        # get three highest scored paths (if there are at least three paths)
        sorted_scores = {key:value for key,value in sorted(branch_scores.items(), key=lambda item: item[1], reverse=True)}
        if len(sorted_scores)>3:
            sorted_scores = sorted_scores[:3]
        # compare their scores to a threshold
        has_parent = False
        for k,v in sorted_scores.items():
            if v>=threshold:
                has_parent = True
                parent = list(k)[:-1]
                graph.add_edge(parent, step)
        if not has_parent and step!=0:
            graph.add_edge(list(sorted_scores.items())[0][0], step)
            print(f"No satisfactory entailment. Adding {sorted_scores.items[0][0]} as parent of {step}")
        # add to highest: what if no path gives satisfactory results?
        new_paths = get_all_new_paths(graph, step)
        paths[node] = new_paths
    return graph

