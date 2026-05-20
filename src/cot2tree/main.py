#!/bin/env python3
import ast
from argparse import ArgumentParser
import networkx as nx
import os
import json
import torch
#from get_questions import load_MMLU, get_lcots, get_labeled_lcots, load_GPQA, load_live_code_bench, load_MATH, load_MMLU_pro, split
from language_models import *
from split_lcot import build_graph_from_chain
from gatv2 import build_features, train, test, build_dataloader
from torch_geometric.explain import Explainer, GNNExplainer, GraphMaskExplainer, PGExplainer, AttentionExplainer
from toy_explainer import get_explanations
from tqdm import tqdm
from readlog import readlog
# This file aggregates all the functions from the other files
# It is the one that collects all the parameters from the user

parser = ArgumentParser(prog="LCoT2Graph")
parser.add_argument("actions",type=str, nargs='+', choices=["train", "test"], help="Whether to train or test the model. Choose \"train\" or \"test\".")
parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity.")
parser.add_argument("-f", "--fusion", action="store_true", help="Whether to fuse the smallest steps with bigger ones.")
parser.add_argument("-c", "--checkpoint", action="store_true", help="Whether to use checkpoints during training.")
parser.add_argument("-t", "--threshold", type=float, default=0.7, help="The threshold for a node to be attached to a parent using the NLI model.")
parser.add_argument("-R", "--resume-training", type=int, default=0, help="Epoch at which to resume training.")
parser.add_argument("-k", type=float, nargs=2, default=[0.01, 0.02], help="Thresholds before which only 1 or 2 parents (respectively) are allowed for a new node.")
parser.add_argument("-m", "--max-context-nli", type=int, default=None, help="The maximum number of steps given to the NLI model. Use None or -1 for no maximum.")
parser.add_argument("-n", "--nb-samples-subject", type=int, default=30, help="The number of samples taken from each subject in MMLU when creating training dataset.")
parser.add_argument("-L", "--use-existing-lcots", action="store_true", help="Whether to use pre-existing LCoTs to build the graph. This requires setting --lcots-directory (-D).")
parser.add_argument("-D", "--lcots-directory", type=str, help="The directory where new LCoTs are stored, and existing ones read.")
parser.add_argument("-g", "--use-existing-graphs", action="store_true", help="Whether to use pre-existing graphs to train the model. This requires setting --graphs-directory (-d).")
parser.add_argument("-d", "--graphs-directory", type=str, help="The directory where new graphs are stored, and existing ones read.")
parser.add_argument("-s", "--dataset-seed", type=int, default=42, help="The seed to use for the random selection of samples in MMLU.")
# Should we keep the argument -G or simply make it a part of -v?
parser.add_argument("-G", "--graph-construction-logfile", type=str, help="The file where the details of the graph construction process are printed.")
parser.add_argument("-p", "--paths-lrms", type=str, nargs='+', help="The paths to the LRMs for the production of the LCoTs.")
parser.add_argument("-b", "--batch-size", type=int, default=32, help="The batch size to train and test the model.")
parser.add_argument("-e", "--epochs", type=int, default=100, help="The number of epochs to train the model.")
parser.add_argument("-r", "--learning-rate", type=float, default=1e-3, help="The learning rate when training the model.")
parser.add_argument("-N", "--nli-model-path", type=str, help="Path to the NLI model to build the graph.")
parser.add_argument("-F", "--wanted-features", type=str, nargs='+', choices=['nb_parents', 'nb_children', 'node_index', 'distance_to_end', 'nb_words_before', 'nb_nodes_per_depth'], help="The list of wanted features for the graph nodes.")
parser.add_argument("-C", "--cross-encoder", type=str, help="The cross-encoder model that evaluates the answers of the LCoTs.")
parser.add_argument("-w", "--nb-keywords", type=int, default=8, help="The number of keywords to use when splitting the LCoTs.")
#parser.add_argument("-i", "--in-channels", type=int, help="The input dimension of the graph model.")
parser.add_argument("-o", "--out-channels", type=int, default=2, help="The output dimension of the graph model.")
parser.add_argument("-H", "--hidden-channels", type=int, default=64, help="The hidden dimension of the graph model.")
parser.add_argument("-M", "--trained-model-path",type=str, help="The path to the file containing the model to use, either for training or to continue training from a checkpoint.")

args = parser.parse_args()
actions = args.actions
verbose = args.verbose
train_samples = None
eval_samples = None
test_samples = None
test_split = None
trained_model = None
wanted_features = {feature:i for i, feature in enumerate(args.wanted_features)}
parent_dir = "/".join(os.getcwd().split("/")[:-1])
if verbose:
    print(f"The given arguments are:{args}")
if "train" in actions:
    if args.use_existing_graphs:  # If we use pre-existing graphs.
        # We read from the files where the graphs are saved.
        if verbose:
            print(f"Loading existing graphs for training and evaluation from directory {args.graphs_directory}.")
        files = os.listdir(args.graphs_directory)
        for file in files:
            path = os.path.join(args.graphs_directory, file)
            with open(path, "r") as f:
                contents = f.read()
                if "train" in file:
                    if verbose:
                        print(f"Loading train graphs from file {path}.")
                    """a, b, c = f.read().split("############")[0].split("&&&&&&&&&&&&")
                    print(f"Graph: {a}")
                    print(f"Trying ast.literal_eval: {ast.literal_eval(a)}")
                    print(f"Features: {b}")
                    print(f"Label: {c}")"""
                    train_graphs_with_full_features = [(nx.from_dict_of_dicts(ast.literal_eval(content.split("&&&&&&&&&&&&")[0])), torch.tensor(ast.literal_eval(content.split("&&&&&&&&&&&&")[1])), eval(content.split("&&&&&&&&&&&&")[2])) for content in contents.split("############")]
                    # For each graph, we need a ast.literal_eval, for the features a split on "," twice, and for the labels a transformation to boolean form.
                if "eval" in file:
                    if verbose:
                        print(f"Loading eval graphs from file {path}.")
                    eval_graphs_with_full_features = [(nx.from_dict_of_dicts(ast.literal_eval(content.split("&&&&&&&&&&&&")[0])), torch.tensor(ast.literal_eval(content.split("&&&&&&&&&&&&")[1])), eval(content.split("&&&&&&&&&&&&")[2])) for content in contents.split("############")]
    
    else:
        if args.use_existing_lcots:  # If we use pre-existing LCoTs
            if verbose:
                print(f"Loading existing LCoTs for training and evaluation from directory {args.lcots_directory}.")
            files = os.listdir(args.lcots_directory)
            for file in files:
                path = os.path.join(args.lcots_directory, file)
                with open(path, "r") as f:
                    print(f"File name: {path}")
                    contents = f.read()
                    if "train" in file:
                        if verbose:
                            print(f"Loading train LCoTs from file {path}.")
                        #train_samples = [(iteration.split("&&&&&&&&&&&&")[0], iteration.split("&&&&&&&&&&&&")[1]) for iteration in contents.split("############")]
                        train_samples = readlog(contents=contents)
                        print(f"len(train_samples): {len(train_samples)}")
                    if "eval" in file:
                        if verbose:
                            print(f"Loading eval LCoTs from file {path}.")
                        eval_samples = readlog(contents=contents)
                        #eval_samples = [(iteration.split("&&&&&&&&&&&&")[0], iteration.split("&&&&&&&&&&&&")[1]) for iteration in contents.split("############")]
        else:
            if verbose:
                print("No existing graphs or LCoTs given, using default.")
                print("Loading the datasets.")
            #train_split, eval_split, test_split = load_MMLU(args.nb_samples_subject, parent_dir=parent_dir, seed=args.dataset_seed,)
            #train_samples = get_lcots_with_labels(samples=train_split, cross_encoder=args.cross_encoder, lrms=args.paths_lrms)
            #eval_samples = get_lcots_with_labels(samples=eval_split, cross_encoder=args.cross_encoder, lrms=args.paths_lrms)
            mmlu_pro = load_MMLU_pro(seed=42, parent_dir=parent_dir)
            gpqa =load_GPQA(42, parent_dir)
            lcb = load_live_code_bench(42, parent_dir)
            math = load_MATH(42, parent_dir)
            mmlu_lcots, mmlu_answers = get_lcots(mmlu_pro, nb_samples=5)
            math_lcots, math_answers = get_lcots(math, nb_samples=5)
            # for lcb, we need 3 iterations for each model, and 2 for qpqa
            lcb_lcots, lcb_answers = get_lcots(lcb, nb_samples=5)
            gpqa_lcots, gpqa_answers = get_lcots(gpqa, nb_samples=5)
            fin_mmlu_lcots = get_labeled_lcots(mmlu_lcots, mmlu_answers, args.cross_encoder, 0.7, args.verbose, nb_samples=5)
            fin_gpqa_lcots = get_labeled_lcots(gpqa_lcots, gpqa_answers, args.cross_encoder, 0.7, args.verbose, nb_samples=5)
            fin_lcb_lcots = get_labeled_lcots(lcb_lcots, lcb_answers, args.cross_encoder, 0.7, args.verbose, nb_samples=5)
            fin_math_lcots = get_labeled_lcots(math_lcots, math_answers, args.cross_encoder, 0.7, args.verbose, nb_samples=5)
            train_mmlu, eval_mmlu, test_mmlu = split(fin_mmlu_lcots)
            train_math, eval_math, test_math = split(fin_math_lcots)
            train_lcb, eval_lcb, test_lcb = split(fin_lcb_lcots)
            train_gpqa, eval_gpqa, test_gpqa = split(fin_gpqa_lcots)
            train_samples = train_mmlu+train_math+train_lcb+train_gpqa
            eval_samples = eval_mmlu+eval_math+eval_lcb+eval_gpqa
            test_samples = {"mmlu":test_mmlu, "gpqa":test_gpqa, "lcb":test_lcb, "math": test_math}
            # We save those LCoTs and their labels for potential later use.
            if not os.path.isdir(args.lcots_directory):
                if verbose:
                    print(f"Did not find directory {args.lcots_directory}. Creating directory.")
                os.mkdir(args.lcots_directory)
            path_train = os.path.join(args.lcots_directory,"train.txt")
            path_eval = os.path.join(args.lcots_directory, "eval.txt")
            path_tests = [os.path.join(args.lcots_directory,"test")+ds+".txt" for ds in ["mmlu","gpqa","lcb","math"]]
            with open(path_train, "w+") as f:
                if verbose:
                    print(f"Saving train LCoTs to file {path_train}.")
                print("############".join([lcot+"&&&&&&&&&&&&"+str(label) for lcot, label in train_samples]),file=f)
            with open(path_eval, "w+") as f:
                if verbose:
                    print(f"Saving eval LCoTs to file {path_eval}.")
                print("############".join([lcot+"&&&&&&&&&&&&"+str(label) for lcot, label in eval_samples]),file=f)

            with open(path_tests[0], "w+") as f:
                if verbose:
                    print(f"Saving MMLU pro test LCoTs in : {f}.")
                print("############".join([lcot+"&&&&&&&&&&&&"+str(label) for lcot, label in test_mmlu]),file=f)
            with open(path_tests[1], "w+") as f:
                if verbose:
                    print(f"Saving GPQA pro test LCoTs in : {f}.")
                print("############".join([lcot+"&&&&&&&&&&&&"+str(label) for lcot, label in test_gpqa]),file=f)
            with open(path_tests[2], "w+") as f:
                if verbose:
                    print(f"Saving LCB pro test LCoTs in : {f}.")
                print("############".join([lcot+"&&&&&&&&&&&&"+str(label) for lcot, label in test_lcb]),file=f)
            with open(path_tests[3], "w+") as f:
                if verbose:
                    print(f"Saving MATH pro test LCoTs in : {f}.")
                print("############".join([lcot+"&&&&&&&&&&&&"+str(label) for lcot, label in test_math]),file=f)
   
        # We make the graphs and features from the LCoTs
        train_lcots = [lcot for lcot, _ in train_samples]
        print(f"len(train_lcots): {len(train_lcots)}")
        print(f"Length of each lcot: {[len(lcot) for lcot in train_lcots]}")
        eval_lcots = [lcot for lcot, _ in eval_samples]
        logfile = open(args.graph_construction_logfile, "w+")
        train_graphs_features = [build_graph_from_chain(lcot=lcot, nb_keywords=args.nb_keywords, max_path_length_for_nli=args.max_context_nli, logfile=logfile, wanted_features=wanted_features) for lcot in tqdm(train_lcots)]
        eval_graphs_features = [build_graph_from_chain(lcot=lcot, nb_keywords=args.nb_keywords, max_path_length_for_nli=args.max_context_nli, logfile=logfile, wanted_features=wanted_features) for lcot in tqdm(eval_lcots)]
        logfile.close()
        # These two lines might cause trouble, I am not sure about the way this zip unfolds.
        train_graphs_with_full_features = [(graph, build_features(graph=graph, all_features=features, wanted_features=wanted_features), eval(label)) for (graph,features),(_, label) in zip(train_graphs_features,train_samples)]
        eval_graphs_with_full_features = [(graph, build_features(graph=graph, all_features=features, wanted_features=wanted_features), eval(label)) for (graph,features),(_, label) in zip(eval_graphs_features, eval_samples)]

        # We save those graphs, their features, and their labels for potential future use.
        if not os.path.isdir(args.graphs_directory):
                if verbose:
                    print(f"Did not find directory {args.graphs_directory}. Creating directory.")
                os.mkdir(args.graphs_directory)
        path_train = os.path.join(args.graphs_directory,"train.txt")
        path_eval = os.path.join(args.graphs_directory, "eval.txt")
        with open(path_train, "w+") as f:
            if verbose:
                    print(f"Saving train graphs to file {path_train}.")
            print("############".join([str(nx.to_dict_of_dicts(graph))+"&&&&&&&&&&&&"+str(features.tolist())+"&&&&&&&&&&&&"+str(label) for graph, features, label in train_graphs_with_full_features]),file=f)
        with open(path_eval, "w+") as f:
            if verbose:
                    print(f"Saving train graphs to file {path_eval}.")
            print("############".join([str(nx.to_dict_of_dicts(graph))+"&&&&&&&&&&&&"+str(features.tolist())+"&&&&&&&&&&&&"+str(label) for graph, features, label in eval_graphs_with_full_features]),file=f)
    
    # Now we create the DataLoaders
    train_graphs, train_features, train_labels = zip(*train_graphs_with_full_features)
    print(f"train_graphs_with_full_features: {train_graphs_with_full_features}")
    print(f"train_features: {train_features}")
    print(train_features[0])
    print(type(train_features[0][0]))
    eval_graphs, eval_features, eval_labels = zip(*eval_graphs_with_full_features)
    print(f"Eval_graphs: {eval_graphs}")
    train_loader = build_dataloader(list(train_features), list(train_graphs), list(train_labels), batch_size=args.batch_size)
    print(f"Train loader: {train_loader}")
    eval_loader = build_dataloader(list(eval_features), list(eval_graphs), list(eval_labels), batch_size=args.batch_size)
    print(f"Eval loader: {eval_loader}")
    trained_model = train(train_dataloader=train_loader, val_loader=eval_loader, in_channels=len(wanted_features), out_channels=args.out_channels, hidden=args.hidden_channels, parent_dir=parent_dir, epochs=args.epochs, lr=args.learning_rate)

    # We save the trained model in the specified path.
    if verbose:
        print(f"Saving the trained model to file {args.trained_model_path}.")
    torch.save(trained_model.state_dict(), args.trained_model_path)
    

if "test" in actions:
    if args.use_existing_graphs:  # If we use pre-existing graphs.
        # We read from the files where the graphs are saved.
        files = os.listdir(args.graphs_directory)
        if verbose:
            print(f"Loading existing graphs for training and evaluation from directory {args.graphs_directory}.")
        test_graphs_with_full_features = {}
        for file in files:
            if "test" in file:
                subject = file.split('_')[1].split('.')[0]  # file is of the shape test_subject.txt
                path = os.path.join(args.graphs_directory, file)
                with open(path, "r") as f:
                    if verbose:
                        print(f"Loading test graphs on subject {subject} from file {path}.")
                    contents = f.read()
                    test_graphs_with_full_features[subject] = [(nx.from_dict_of_dicts(ast.literal_eval(content.split("&&&&&&&&&&&&")[0])), torch.tensor(ast.literal_eval(content.split("&&&&&&&&&&&&")[1])), eval(content.split("&&&&&&&&&&&&")[2])) for content in contents.split("############")]
    else:
        if args.use_existing_lcots:  # If we use pre-existing LCoTs
            if fin_mmlu_lcots and fin_math_lcots and fin_lcb_lcots and fin_gpqa_lcots:
                test_samples = {'mmlu':fin_mmlu_lcots, 'gpqa':fin_gpqa_lcots, 'lcb':fin_lcb_lcots, 'math':fin_math_lcots}
                if verbose:
                    print("Using existing LCoTs in memory.")
            else:
                if verbose:
                    print(f"Loading existing LCoTs for testing from directory {args.lcots_directory}.")
                test_samples_true = {}
                test_samples_false = {}
                for ds in ["mmlu", "gpqa", "lcb", "math"]:
                    ds_samples_true = {}
                    ds_samples_false = {}
                    for m in ["llama", "qwen", "qwq"]:
                        file_true = os.path.join(args.lcots_directory,"test_"+ds+"_"+m+"_true"+".txt")
                        file_false = os.path.join(args.lcots_directory,"test_"+ds+"_"+m+"_false"+".txt")
                        with open(file_true, "r+") as f:
                            contents = f.read()
                            ds_samples_true[m] = [(iteration.split("&&&&&&&&&&&&")[0], iteration.split("&&&&&&&&&&&&")[1]) for iteration in contents.split("############")]
                        with open(file_false, "r+") as f:
                            contents = f.read()
                            ds_samples_false[m] = [(iteration.split("&&&&&&&&&&&&")[0], iteration.split("&&&&&&&&&&&&")[1]) for iteration in contents.split("############")]
                    
                    test_samples_true[ds] = ds_samples_true
                    test_samples_false[ds] = ds_samples_false
        else:  # Won't be used
            if verbose:
                print("No existing graphs or LCoTs given, using default.")
            test_samples = {}
            if test_split is None:
                if verbose:
                    print("Loading MMLU test split.")
                _, _, test_split = load_MMLU(args.nb_samples_subject, parent_dir=parent_dir, seed=args.dataset_seed)
            else:
                if verbose:
                    print("Using already loaded MMLU test split.")
            if not os.path.isdir(args.lcots_directory):
                if verbose:
                    print(f"Did not find directory {args.lcots_directory}. Creating directory.")
                os.mkdir(args.lcots_directory)
            for subject in test_split:
                test_samples[subject] = get_lcots(samples=test_split[subject], cross_encoder=args.cross_encoder, lrms=args.paths_lrms)
                path_test = os.path.join(args.lcots_directory,f"test_{subject}.txt")
                with open(path_test, "w+") as f:
                    if verbose:
                        print(f"Saving test LCoTs for subject {subject} to file {path_test}.")
                    print("############".join([lcot+"&&&&&&&&&&&&"+label for lcot, label in test_samples[subject]]),file=f)
        
        # We produce the test graphs.
        if not os.path.isdir(args.graphs_directory):
                if verbose:
                    print(f"Did not find directory {args.graphs_directory}. Creating directory.")
                os.mkdir(args.graphs_directory)
        test_lcots_true = {}  # a dict with just the lcots, no labels
        test_lcots_false = {}
        for subject in test_samples_true:
            subj_lcots = {}
            for m in test_samples_true[subject]:
                subj_lcots[m] = [lcot for lcot, _ in test_samples_true[subject][m]]
            test_lcots_true[subject] = subj_lcots
        for subject in test_samples_false:
            subj_lcots = {}
            for m in test_samples_false[subject]:
                subj_lcots[m] = [lcot for lcot, _ in test_samples_false[subject][m]]
            test_lcots_false[subject] = subj_lcots
        test_graphs_features_true = {}
        test_graphs_features_false = {}
        test_graphs_with_full_features_true = {}
        test_graphs_with_full_features_false = {}
        log = open(args.graph_construction_logfile, "w+")
        if trained_model is None:
            if verbose:
                print(f"Loading the trained model from file {args.trained_model_path}.")
            trained_model = torch.load(args.trained_model_path, weights_only=False)
        for subject in test_lcots_true:
            subj_graphs = {}
            subj_full_features = {}
            for m in test_lcots_true[subject]:
                subj_graphs[m] = [build_graph_from_chain(lcot=lcot, nb_keywords=args.nb_keywords, max_path_length_for_nli=args.max_context_nli, logfile=log) for lcot in test_lcots_true[subject][m]]
            
                # These two lines might cause trouble, I am not sure about the way this zip unfolds.
                subj_full_features[m] = [(graph, build_features(graph=graph, all_features=features, wanted_features=wanted_features), label) for (graph,features),(_, label) in zip(subj_graphs[m],test_samples_true[subject][m])]
                path_save = os.path.join(args.graphs_directory,f"test_{subject}_{m}.txt")
                if verbose:
                    print(f"Saving graphs for subject {subject} in file {path_save}")
                with open(path_save) as f:
                    print("############".join([str(nx.to_dict_of_dicts(graph))+"&&&&&&&&&&&&"+str(features.tolist())+"&&&&&&&&&&&&"+str(label) for graph, features, label in subj_full_features[m]]),file=f)
                test_graphs, test_features, test_labels = zip(*subj_full_features[m])
                test_loader = build_dataloader(test_features, test_graphs, test_labels, batch_size=args.batch_size)
                test(test_dataloader=test_loader, model=trained_model)
                explainer = Explainer(
                                model=trained_model, 
                                algorithm=GNNExplainer(epochs=200), 
                                explanation_type='model', 
                                edge_mask_type='object', 
                                model_config=dict(mode="multiclass_classification", task_level="graph", return_type="log_probs")
                            )
                patterns = get_explanations(explainer, test_loader, subject, m, parent_dir)
            test_graphs_features_true[subject] = subj_graphs 
            test_graphs_with_full_features_true[subject] = subj_full_features   
        for subject in test_lcots_false:
            subj_graphs = {}
            subj_full_features = {}
            for m in test_lcots_false[subject]:
                subj_graphs[m] = [build_graph_from_chain(lcot=lcot, nb_keywords=args.nb_keywords, max_path_length_for_nli=args.max_context_nli, logfile=log) for lcot in test_lcots_false[subject][m]]
            
                # These two lines might cause trouble, I am not sure about the way this zip unfolds.
                subj_full_features[m] = [(graph, build_features(graph=graph, all_features=features, wanted_features=wanted_features), label) for (graph,features),(_, label) in zip(subj_graphs[m],test_samples_false[subject][m])]
                path_save = os.path.join(args.graphs_directory,f"test_{subject}_{m}.txt")
                if verbose:
                    print(f"Saving graphs for subject {subject} in file {path_save}")
                with open(path_save) as f:
                    print("############".join([str(nx.to_dict_of_dicts(graph))+"&&&&&&&&&&&&"+str(features.tolist())+"&&&&&&&&&&&&"+str(label) for graph, features, label in subj_full_features[m]]),file=f)
                test_graphs, test_features, test_labels = zip(*subj_full_features[m])
                test_loader = build_dataloader(test_features, test_graphs, test_labels, batch_size=args.batch_size)
                test(test_dataloader=test_loader, model=trained_model)
                explainer = Explainer(
                                model=trained_model, 
                                algorithm=GNNExplainer(epochs=200), 
                                explanation_type='model', 
                                edge_mask_type='object', 
                                model_config=dict(mode="multiclass_classification", task_level="graph", return_type="log_probs")
                            )
                patterns = get_explanations(explainer, test_loader, subject, m, parent_dir)
            test_graphs_features_false[subject] = subj_graphs 
            test_graphs_with_full_features_false[subject] = subj_full_features
    
    


    # build the explainer
    explainer = Explainer(
        model=trained_model, 
        algorithm=GNNExplainer(epochs=200), 
        explanation_type='model', 
        edge_mask_type='object', 
        model_config=dict(mode="multiclass_classification", task_level="graph", return_type="log_probs")
    )

    # now we explain
    #TODO: we need to find a way to average feature importance
    print('\n\n')
    print("----------Now we explain the decisions----------")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device', device)
    trained_model.to(device)
    trained_model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            explanation = explainer(x=data.x, edge_index=data.edge_index)
            
   

    
    
        