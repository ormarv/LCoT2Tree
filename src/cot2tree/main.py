#!/bin/env python3
from argparse import ArgumentParser
import os
import json
import torch
from get_questions import load_MMLU, get_lcots_with_labels
from language_models import *
from split_lcot import build_graph_from_chain
from gatv2 import build_features, train, test, build_dataloader

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
parser.add_argument("-d", "--graphs-directory", type=str, help="The directory where new graphs are sotred, and existing ones read.")
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
            with open(path, "w+") as f:
                if "train" in file:
                    if verbose:
                        print(f"Loading train graphs from file {path}.")
                    c = f.read().split("############")
                    print(f"Content: {c}")
                    train_graphs_with_full_features = [(json.loads(content.split("&&&&&&&&&&&&")[0]), [feature.split(',') for feature in content.split("&&&&&&&&&&&&")[1].split(',')], eval(content.split("&&&&&&&&&&&&")[2])) for content in f.read().split("############")]
                    # For each graph, we need a json.loads, for the features a split on "," twice, and for the labels a transformation to boolean form.
                if "eval" in file:
                    if verbose:
                        print(f"Loading eval graphs from file {path}.")
                    eval_graphs_with_full_features = [(json.loads(content.split("&&&&&&&&&&&&")[0]), [feature.split(',') for feature in content.split("&&&&&&&&&&&&")[1].split(',')], eval(content.split("&&&&&&&&&&&&")[2])) for content in f.read().split("############")]
    
    else:
        if args.use_existing_lcots:  # If we use pre-existing LCoTs
            if verbose:
                print(f"Loading existing LCoTs for training and evaluation from directory {args.lcots_directory}.")
            files = os.listdir(args.lcots_directory)
            for file in files:
                path = os.path.join(args.lcots_directory, file)
                with open(path, "w+") as f:
                    if "train" in file:
                        if verbose:
                            print(f"Loading train LCoTs from file {path}.")
                        train_samples = [(iteration.split("&&&&&&&&&&&&")[0], iteration.split("&&&&&&&&&&&&")[1]) for iteration in f.read().split("############")]
                    if "eval" in file:
                        if verbose:
                            print(f"Loading eval LCoTs from file {path}.")
                        eval_samples = [(iteration.split("&&&&&&&&&&&&")[0], iteration.split("&&&&&&&&&&&&")[1]) for iteration in f.read().split("############")]
        else:
            if verbose:
                print("No existing graphs or LCoTs given, using default.")
                print("Loading MMLU.")
            train_split, eval_split, test_split = load_MMLU(args.nb_samples_subject, parent_dir=parent_dir, seed=args.dataset_seed,)
            train_samples = get_lcots_with_labels(samples=train_split, cross_encoder=args.cross_encoder, lrms=args.paths_lrms)
            eval_samples = get_lcots_with_labels(samples=eval_split, cross_encoder=args.cross_encoder, lrms=args.paths_lrms)
            # We save those LCoTs and their labels for potential later use.
            if not os.path.isdir(args.lcots_directory):
                if verbose:
                    print(f"Did not find directory {args.lcots_directory}. Creating directory.")
                os.mkdir(args.lcots_directory)
            path_train = os.path.join(args.lcots_directory,"train.txt")
            path_eval = os.path.join(args.lcots_directory, "eval.txt")
            with open(path_train, "w+") as f:
                if verbose:
                    print(f"Saving train LCoTs to file {path_train}.")
                print("############".join([lcot+"&&&&&&&&&&&&"+str(label) for lcot, label in train_samples]),file=f)
            with open(path_eval, "w+") as f:
                if verbose:
                    print(f"Saving eval LCoTs to file {path_eval}.")
                print("############".join([lcot+"&&&&&&&&&&&&"+str(label) for lcot, label in eval_samples]),file=f)
        
        # We make the graphs and features from the LCoTs
        train_lcots = [lcot for lcot, _ in train_samples]
        eval_lcots = [lcot for lcot, _ in eval_samples]
        train_graphs_features = [build_graph_from_chain(lcot=lcot, nb_keywords=args.nb_keywords, max_path_length_for_nli=args.max_context_nli, logfile=open(args.graph_construction_logfile)) for lcot in train_lcots]
        eval_graphs_features = [build_graph_from_chain(lcot=lcot, nb_keywords=args.nb_keywords, max_path_length_for_nli=args.max_context_nli, logfile=open(args.graph_construction_logfile)) for lcot in eval_lcots]
        # These two lines might cause trouble, I am not sure about the way this zip unfolds.
        train_graphs_with_full_features = [(graph, build_features(graph=graph, all_features=features, wanted_features=args.wanted_features), eval(label)) for (graph,features),(_, label) in zip(train_graphs_features,train_samples)]
        eval_graphs_with_full_features = [(graph, build_features(graph=graph, all_features=features, wanted_features=args.wanted_features), eval(label)) for (graph,features),(_, label) in zip(eval_graphs_features, eval_samples)]

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
            print("############".join([graph+"&&&&&&&&&&&&"+features+"&&&&&&&&&&&&"+str(label) for graph, features, label in train_graphs_with_full_features]),file=f)
        with open(path_eval, "w+") as f:
            if verbose:
                    print(f"Saving train graphs to file {path_eval}.")
            print("############".join([graph+"&&&&&&&&&&&&"+features+"&&&&&&&&&&&&"+str(label) for graph, features, label in eval_graphs_with_full_features]),file=f)
    
    # Now we create the DataLoaders
    train_graphs, train_features, train_labels = zip(*train_graphs_with_full_features)
    eval_graphs, eval_features, eval_labels = zip(*eval_graphs_with_full_features)
    train_loader = build_dataloader(train_features, train_graphs, train_labels, batch_size=args.batch_size)
    eval_loader = build_dataloader(eval_features, eval_graphs, eval_labels, batch_size=args.batch_size)
    trained_model = train(train_dataloader=train_loader, val_loader=eval_loader, in_channels=len(wanted_features), out_channels=args.o, hidden=args.hidden_channels, epochs=args.epochs, lr=args.learning_rate)

    # We save the trained model in the specified path.
    if verbose:
        print(f"Saving the trained model to file {args.threshold}.")
    

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
                with open(path, "w+") as f:
                    if verbose:
                        print(f"Loading test graphs on subject {subject} from file {path}.")
                    test_graphs_with_full_features[subject] = [(json.loads(content.split("&&&&&&&&&&&&")[0]), [feature.split(',') for feature in content.split("&&&&&&&&&&&&")[1].split(',')], eval(content.split("&&&&&&&&&&&&")[2])) for content in f.read().split("############")]
    else:
        if args.use_existing_lcots:  # If we use pre-existing LCoTs
            if verbose:
                print(f"Loading existing LCoTs for testing from directory {args.lcots_directory}.")
            files = os.listdir(args.lcots_directory)
            test_samples = {}
            for file in files:
                if "test" in file:
                    subject = file.split('_')[1].split('.')[0]
                    path = os.path.join(args.lcots_directory, file)
                    with open(path, "w+") as f:
                        test_samples[subject] = [(iteration.split("&&&&&&&&&&&&")[0], iteration.split("&&&&&&&&&&&&")[1]) for iteration in f.read().split("############")]
        else:
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
                test_samples[subject] = get_lcots_with_labels(samples=test_split[subject], cross_encoder=args.cross_encoder, lrms=args.paths_lrms)
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
        test_lcots = {}
        for subject in test_samples:
            test_lcots[subject] = [lcot for lcot, _ in test_samples]
        test_graphs_features = {}
        test_graphs_with_full_features = {}
        for subject in test_lcots:
            test_graphs_features[subject] = [build_graph_from_chain(lcot=lcot, nb_keywords=args.nb_keywords, max_path_length_for_nli=args.max_context_nli, logfile=open(args.graph_construction_logfile)) for lcot in test_lcots[subject]]
            # These two lines might cause trouble, I am not sure about the way this zip unfolds.
            test_graphs_with_full_features[subject] = [(graph, build_features(graph=graph, all_features=features, wanted_features=args.wanted_features), label) for (graph,features),(_, label) in zip(test_graphs_features[subject],test_samples[subject])]
    # Now we need a trained GAT model
    if trained_model is None:
        if verbose:
            print(f"Loading the trained model from file {args.threshold}.")
        trained_model = torch.load(args.threshold, weights_only=False)
    for subject in test_graphs_with_full_features:
        path_test = os.path.join(args.graphs_directory,f"test_{subject}.txt")
        if verbose:
            print(f"Saving graphs for subject {subject} in file {path_test}")
        print("############".join([graph+"&&&&&&&&&&&&"+features+"&&&&&&&&&&&&"+str(label) for graph, features, label in test_graphs_with_full_features[subject]]),file=f)
        test_graphs, test_features, test_labels = zip(*test_graphs_with_full_features[subject])
        test_loader = build_dataloader(test_features, test_graphs, test_labels, batch_size=args.batch_size)
        test(test_dataloader=test_loader, model=trained_model)

    
        