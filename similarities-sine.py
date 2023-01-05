"""
Script to run experiments with a single algorithm of choice.
The design allows for user input and flexibility. 

Command line options are:
------------------------
runs : int, optional
    Number of experiments to perform (using different random seeds)
    (default = 1)
N : int, optional
    Number of classes per task
k : int
    Number of examples in the support sets of tasks
k_test : int
    Number of examples in query sets of meta-validation and meta-test tasks
T : int
    Number of optimization steps to perform on a given task
train_batch_size : int, optional
    Size of minibatches to sample from META-TRAIN tasks (or size of flat minibatches
    when the model requires flat data and batch size > k)
    Default = k (no minibatching, simply use entire set)
test_batch_size : int, optional
    Size of minibatches to sample from META-[VAL/TEST] tasks (or size of flat minibatches
    when the model requires flat data and batch size > k)
    Default = k (no minibatching, simply use entire set)
logfile : str
    File name to write results in (does not have to exist, but the containing dir does)
seed : int, optional
    Random seed to use
cpu : boolean, optional
    Whether to use cpu

Usage:
---------------
python main.py --arg=value --arg2=value2 ...
"""

import argparse
import csv
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import json
import random
import torchmeta
import torchmeta.datasets as datasets
import matplotlib.pyplot as plt
import pandas as pd

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
from torchmeta.transforms import Categorical, ClassSplitter
from torchmeta.utils.data import BatchMetaDataLoader
from tqdm import tqdm #Progress bars
from networks import SineNetwork, Conv4, BoostedConv4, ConvX, ResNet, LinearNet
from algorithms.metalearner_lstm import LSTMMetaLearner  
from algorithms.train_from_scratch import TrainFromScratch
from algorithms.finetuning import FineTuning
from algorithms.moso import MOSO
from algorithms.turtle import Turtle
from algorithms.reptile import Reptile
from algorithms.maml import MAML
from algorithms.ownlstm import LSTM
from algorithms.modules.utils import get_init_score_and_operator
from sine_loader import SineLoader
from image_loader import ImageLoader
from linear_loader import LinearLoader
from misc import BANNER, NAMETAG
from configs import TFS_CONF, FT_CONF, CFT_CONF, LSTM_CONF,\
                    MAML_CONF, MOSO_CONF, TURTLE_CONF, LSTM_CONF2,\
                    REPTILE_CONF
from batch_loader import BatchDataset, cycle

FLAGS = argparse.ArgumentParser()

# Required arguments
FLAGS.add_argument("--problem", choices=["sine", "min", "cub", "linear"], required=True,
                   help="Which problem to address?")

FLAGS.add_argument("--k", type=int, required=True,
                   help="Number examples per task set during meta-validation and meta-testing."+\
                   "Also the number of examples per class in every support set.")

FLAGS.add_argument("--k_train", type=int, default=None,
                   help="Number examples per task set during meta-training."+\
                   "Also the number of examples per class in every support set.")

FLAGS.add_argument("--k_test", type=int, required=True,
                   help="Number examples per class in query set")

FLAGS.add_argument("--model", choices=["tfs", "finetuning", "centroidft", 
                   "lstm", "maml", "moso", "lstm2", "turtle", "reptile"], required=True,
                   help="Which model to use?")

# Optional arguments
FLAGS.add_argument("--N", type=int, default=None,
                   help="Number of classes (only applicable when doing classification)")   

FLAGS.add_argument("--meta_batch_size", type=int, default=1,
                   help="Number of tasks to compute outer-update")   

FLAGS.add_argument("--val_after", type=int, default=None,
                   help="After how many episodes should we perform meta-validation?")

FLAGS.add_argument("--decouple", type=int, default=None,
                   help="After how many train tasks switch from meta-mode to base-mode?")

FLAGS.add_argument("--lr", type=float, default=None,
                   help="Learning rate for (meta-)optimizer")

FLAGS.add_argument("--cpe", type=float, default=0.5,
                   help="#Times best weights get reconsidered per episode (only for baselines)")

FLAGS.add_argument("--T", type=int, default=None,
                   help="Number of weight updates per training set")

FLAGS.add_argument("--T_val", type=int, default=None,
                   help="Number of weight updates at validation time")

FLAGS.add_argument("--T_test", type=int, default=None,
                   help="Number of weight updates at test time")

FLAGS.add_argument("--history", choices=["none", "grads", "updates"], default="none",
                   help="Historical information to use (only applicable for TURTLE): none/grads/updates")

FLAGS.add_argument("--beta", type=float, default=None,
                   help="Beta value to use (only applies when model=TURTLE)")

FLAGS.add_argument("--train_batch_size", type=int, default=None,
                   help="Size of minibatches for training "+\
                         "only applies for flat batch models")

FLAGS.add_argument("--test_batch_size", type=int, default=None,
                   help="Size of minibatches for testing (default = None) "+\
                   "only applies for flat-batch models")

FLAGS.add_argument("--activation", type=str, choices=["relu", "tanh", "sigmoid"],
                   default=None, help="Activation function to use for TURTLE/MOSO")

FLAGS.add_argument("--runs", type=int, default=30, 
                   help="Number of runs to perform")

FLAGS.add_argument("--devid", type=int, default=None, 
                   help="CUDA device identifier")

FLAGS.add_argument("--second_order", action="store_true", default=False,
                   help="Use second-order gradient information for TURTLE")

FLAGS.add_argument("--batching_eps", action="store_true", default=False,
                   help="Batching from episodic data")

FLAGS.add_argument("--input_type", choices=["raw_grads", "raw_loss_grads", 
                   "proc_grads", "proc_loss_grads", "maml"], default=None, 
                   help="Input type to the network (only for MOSO and TURTLE"+\
                   " choices = raw_grads, raw_loss_grads, proc_grads, proc_loss_grads, maml")

FLAGS.add_argument("--layer_wise", action="store_true", default=False,
                   help="Whether TURTLE should use multiple meta-learner networks: one for every layer in the base-learner")

FLAGS.add_argument("--param_lr", action="store_true", default=False,
                   help="Whether TURTLE should learn a learning rate per parameter")

FLAGS.add_argument("--base_lr", type=float, default=None,
                   help="Inner level learning rate")

FLAGS.add_argument("--train_iters", type=int, default=None,
                    help="Number of meta-training iterations")

FLAGS.add_argument("--model_spec", type=str, default=None,
                   help="Store results in file ./results/problem/k<k>test<k_test>/<model_spec>/")

FLAGS.add_argument("--layers", type=str, default=None,
                   help="Neurons per hidden/output layer split by comma (e.g., '10,10,1')")

FLAGS.add_argument("--cross_eval", default=False, action="store_true",
                   help="Evaluate on tasks from different dataset (cub if problem=min, else min)")

FLAGS.add_argument("--backbone", type=str, default=None,
                    help="Backbone to use (format: convX)")

FLAGS.add_argument("--seed", type=int, default=1337,
                   help="Random seed to use")

FLAGS.add_argument("--single_run", action="store_true", default=False,
                   help="Whether the script is run independently of others for paralellization. This only affects the storage technique.")

FLAGS.add_argument("--no_annealing", action="store_true", default=False,
                   help="Whether to not anneal the meta learning rate for reptile")

FLAGS.add_argument("--cpu", action="store_true",
                   help="Use CPU instead of GPU")

FLAGS.add_argument("--time_input", action="store_true", default=False,
                   help="Add a timestamp as input to TURTLE")                   

FLAGS.add_argument("--validate", action="store_true", default=False,
                   help="Validate performance on meta-validation tasks")


FLAGS.add_argument("--no_freeze", action="store_true", default=False,
                   help="Whether to freeze the weights in the finetuning model of earlier layers")

FLAGS.add_argument("--eval_on_train", action="store_true", default=False,
                    help="Whether to also evaluate performance on training tasks")

FLAGS.add_argument("--test_adam", action="store_true", default=False,
                   help="Optimize weights with Adam, LR = 0.001 at test time.")

FLAGS.add_argument("--sim_problem", choices=["min", "cub", "sine"], required=True,
                   help="Which problem to test the adaptation on?")

FLAGS.add_argument("--avg_runs", action="store_true", default=False,
                   help="Not only use the best model, but average across all models")

FLAGS.add_argument("--random_init", action="store_true", default=False,
                   help="Dont load stored models but use random initialization")


RESULT_DIR = "./results/"

def create_dir(dirname):
    """
    Create directory <dirname> if not exists
    """
    
    if not os.path.exists(dirname):
        print(f"[*] Creating directory: {dirname}")
        try:
            os.mkdir(dirname)
        except FileExistsError:
            # Dir created by other parallel process so continue
            pass

def print_conf(conf):
    """Print the given configuration
    
    Parameters
    -----------
    conf : dictionary
        Dictionary filled with (argument names, values) 
    """
    
    print(f"[*] Configuration dump:")
    for k in conf.keys():
        print(f"\t{k} : {conf[k]}")

def set_batch_size(conf, args, arg_str):
    value = getattr(args, arg_str)
    # If value for argument provided, set it in configuration
    if not value is None:
        conf[arg_str] = value
    else:
        try:
            # Else, try to fetch it from the configuration
            setattr(args, arg_str, conf[arg_str]) 
            args.train_batch_size = conf["train_batch_size"]
        except:
            # In last case (nothing provided in arguments or config), 
            # set batch size to N*k
            num = args.k
            if not args.N is None:
                num *= args.N
            setattr(args, arg_str, num)
            conf[arg_str] = num             

def overwrite_conf(conf, args, arg_str):
    # If value provided in arguments, overwrite the config with it
    value = getattr(args, arg_str)
    if not value is None:
        conf[arg_str] = value
    else:
        # Try to fetch argument from config, if it isnt there, then the model
        # doesn't need it
        try:
            setattr(args, arg_str, conf[arg_str])
        except:
            return
        
def setup(args):
    """Process arguments and create configurations
        
    Process the parsed arguments in order to create corerct model configurations
    depending on the specified user-input. Load the standard configuration for a 
    given algorithm first, and overwrite with explicitly provided input by the user.

    Parameters
    ----------
    args : cmd arguments
        Set of parsed arguments from command line
    
    Returns
    ----------
    args : cmd arguments
        The processed command-line arguments
    conf : dictionary
        Dictionary defining the meta-learning algorithm and base-learner
    data_loader
        Data loader object, responsible for loading data
    """
    
    if args.k_train is None:
        args.k_train = args.k

    # Mapping from model names to configurations
    mod_to_conf = {
        "tfs": (TrainFromScratch, TFS_CONF),
        "finetuning": (FineTuning, FT_CONF),
        "centroidft": (FineTuning, CFT_CONF), 
        "lstm": (LSTMMetaLearner, LSTM_CONF),
        "lstm2": (LSTM, LSTM_CONF2),
        "maml": (MAML, MAML_CONF),
        "moso": (MOSO, MOSO_CONF),
        "turtle": (Turtle, TURTLE_CONF),
        "reptile": (Reptile, REPTILE_CONF)
    }

    baselines = {"tfs", "finetuning", "centroidft"}
    
    # Get model constructor and config for the specified algorithm
    model_constr, conf = mod_to_conf[args.model]

    # Set batch sizes
    set_batch_size(conf, args, "train_batch_size")
    set_batch_size(conf, args, "test_batch_size")
        
    # Set values of T, lr, and input type
    overwrite_conf(conf, args, "T")
    overwrite_conf(conf, args, "lr")
    overwrite_conf(conf, args, "input_type")
    overwrite_conf(conf, args, "beta")
    overwrite_conf(conf, args, "meta_batch_size")
    overwrite_conf(conf, args, "time_input")
    conf["no_annealing"] = args.no_annealing
    conf["test_adam"] = args.test_adam
    
    # Parse the 'layers' argument
    if not args.layers is None:
        try:
            layers = [int(x) for x in args.layers.split(',')]
        except:
            raise ValueError(f"Error while parsing layers argument {args.layers}")
        conf["layers"] = layers
    
    # Make sure argument 'val_after' is specified when 'validate'=True
    if args.validate:
        assert not args.val_after is None,\
                    "Please specify val_after (number of episodes after which to perform validation)"
    
    # If using multi-step maml, perform gradient clipping with -10, +10
    if not conf["T"] is None:
        if conf["T"] > 1 and (args.model=="maml" or args.model=="turtle"):# or args.model=="reptile"):
            conf["grad_clip"] = 10
        elif args.model == "lstm" or args.model == "lstm2":
            conf["grad_clip"] = 0.25 # it does norm clipping
        else:
            conf["grad_clip"] = None
    
    # If MOSO or TURTLE is selected, set the activation function
    if args.activation:
        act_dict = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(), 
            "sigmoid": nn.Sigmoid()
        }
        conf["act"] = act_dict[args.activation]
    
    # Set the number of reconsiderations of best weights during meta-training episodes,
    # and the device to run the algorithms on 
    conf["cpe"] = args.cpe
    conf["dev"] = args.dev
    conf["second_order"] = args.second_order
    conf["history"] = args.history
    conf["layer_wise"] = args.layer_wise
    conf["param_lr"] = args.param_lr
    conf["decouple"] = args.decouple
    conf["batching_eps"] = args.batching_eps
    conf["freeze"] = not args.no_freeze

    if args.T_test is None:
        conf["T_test"] = conf["T"]
    else:
        conf["T_test"] = args.T_test
    
    if args.T_val is None:
        conf["T_val"] = conf["T"]
    else:
        conf["T_val"] = args.T_val

    if not args.base_lr is None:
        conf["base_lr"] = args.base_lr

    assert not (args.input_type == "maml" and args.history != "none"), "input type 'maml' and history != none are not compatible"
    assert not (conf["T"] == 1 and args.history != "none"), "Historical information cannot be used when T == 1" 

    # Different data set loader to test domain shift robustness
    cross_loader = None
    
    # Pick appropriate base-learner model for the chosen problem [sine/image]
    # and create corresponding data loader obejct
    if args.problem == "linear":
        data_loader = LinearLoader(k=args.k, k_test=args.k_test, seed=args.seed)
        conf["baselearner_fn"] = LinearNet
        conf["baselearner_args"] = {"criterion":nn.MSELoss(), "dev":args.dev}
        conf["generator_args"] = {
            "batch_size": args.train_batch_size, # Only applies for baselines
        }
        train_loader, val_loader, test_loader, cross_loader = data_loader, None, None, None
    elif args.problem == "sine":
        data_loader = SineLoader(k=args.k, k_test=args.k_test, seed=args.seed)
        conf["baselearner_fn"] = SineNetwork
        conf["baselearner_args"] = {"criterion":nn.MSELoss(), "dev":args.dev}
        conf["generator_args"] = {
            "batch_size": args.train_batch_size, # Only applies for baselines
            "reset_ptr": True,
        }
        args.data_loader = data_loader
        train_loader, val_loader, test_loader, cross_loader = data_loader, None, None, None
    else:
        assert not args.N is None, "Please provide the number of classes N per set"
        
        # Image problem
        if args.backbone is None:
            args.backbone = "conv4"
            if args.model == "centroidft":
                conf["baselearner_fn"] = BoostedConv4
                lowerstr = "Bconv4"
            else:    
                conf["baselearner_fn"] = ConvX
                lowerstr = "conv4"
            img_size = (84,84)
        else:
            lowerstr = args.backbone.lower()    
            args.backbone = lowerstr        
            if "resnet" in lowerstr:
                modelstr = "resnet"
                constr = ResNet
                img_size = (224,224)
            elif "conv" in lowerstr:
                modelstr = "conv"
                constr = ConvX
                img_size = (84,84)
            else:
                raise ValueError("Could not parse the provided backbone argument")
            
            num_blocks = int(lowerstr.split(modelstr)[1])
            print(f"Using backbone: {modelstr}{num_blocks}")
            conf["baselearner_fn"] = constr

        if args.train_iters is None:
            if args.k >= 5:
                train_iters = 40000
            else:
                train_iters = 60000
        else:
            train_iters = args.train_iters

        eval_iters = 600
        args.eval_iters = 600
        args.train_iters = train_iters

        if not "sine" in args.sim_problem:
            if "min" in args.sim_problem:
                ds = datasets.MiniImagenet
                cds = datasets.CUB
            elif "cub" in args.sim_problem:
                ds = datasets.CUB
                cds = datasets.MiniImagenet

            val_loader = ds(root="./data/", num_classes_per_task=args.N, meta_train=False, 
                            meta_val=True, meta_test=False, meta_split="val",
                            transform=Compose([Resize(size=img_size), ToTensor()]),
                            target_transform=Compose([Categorical(args.N)]),
                            download=True)
            val_loader = ClassSplitter(val_loader, shuffle=True, num_train_per_class=args.k, num_test_per_class=args.k_test)
            val_loader = BatchMetaDataLoader(val_loader, batch_size=1, num_workers=4, shuffle=True)


            test_loader = ds(root="./data/", num_classes_per_task=args.N, meta_train=False, 
                            meta_val=False, meta_test=True, meta_split="test",
                            transform=Compose([Resize(size=img_size), ToTensor()]),
                            target_transform=Compose([Categorical(args.N)]),
                            download=True)
            test_loader = ClassSplitter(test_loader, shuffle=True, num_train_per_class=args.k, num_test_per_class=args.k_test)
            test_loader = BatchMetaDataLoader(test_loader, batch_size=1, num_workers=4, shuffle=True)


            cross_loader = None
            if args.cross_eval:
                cross_loader = cds(root="./data/", num_classes_per_task=args.N, meta_train=False, 
                                meta_val=False, meta_test=True, meta_split="test",
                                transform=Compose([Resize(size=img_size), ToTensor()]),
                                target_transform=Compose([Categorical(args.N)]),
                                download=True)
                cross_loader = ClassSplitter(cross_loader, shuffle=True, num_train_per_class=args.k, num_test_per_class=args.k_test)
                cross_loader = BatchMetaDataLoader(cross_loader, batch_size=1, num_workers=4, shuffle=True)


        train_class_per_problem = {
            "min": 64,
            "cub": 140
        }

        problem_to_root = {
            "min": "./data/miniimagenet/",
            "cub": "./data/cub/"
        }

        if args.model in baselines:
            if not args.model == "tfs":
                train_classes = train_class_per_problem[args.problem.lower()]
            else:
                train_classes = args.N # TFS does not train, so this enforces the model to have the correct output dim. directly

            train_loader = BatchDataset(root_dir=problem_to_root[args.problem],
                                        transform=Compose([Resize(size=img_size), ToTensor()]))
            train_loader = iter(cycle(DataLoader(train_loader, batch_size=conf["train_batch_size"], shuffle=True, num_workers=4)))
            args.batchmode = True
            print("Using custom made BatchDataset")
        else:
            train_classes = args.N

            train_loader = ds(root="./data/", num_classes_per_task=args.N, meta_train=True, 
                          meta_val=False, meta_test=False, meta_split="train",
                          transform=Compose([Resize(size=img_size), ToTensor()]),
                          target_transform=Compose([Categorical(args.N)]),
                          download=True)
            train_loader = ClassSplitter(train_loader, shuffle=True, num_train_per_class=args.k_train, num_test_per_class=args.k_test)
            train_loader = BatchMetaDataLoader(train_loader, batch_size=1, num_workers=4, shuffle=True)
            args.batchmode = False
            
        conf["baselearner_args"] = {
            "train_classes": train_classes,
            "eval_classes": args.N, 
            "criterion": nn.CrossEntropyLoss(),
            "dev":args.dev
        }

        if not args.backbone is None:
            conf["baselearner_args"]["num_blocks"] = num_blocks
        
        args.backbone = lowerstr
        
    # Print the configuration for confirmation
    print_conf(conf)
    

    if args.problem == "linear" or args.problem == "sine":
        episodic = True
        args.batchmode = False
        if args.model in baselines:
            episodic = False
            args.batchmode = True
        
        print(args.train_batch_size)
        train_loader = train_loader.generator(episodic=episodic, batch_size=args.train_batch_size, mode="train")
        args.linear = True
        args.sine = True
    else:
        args.linear = False
        args.sine = False



    
    args.resdir = RESULT_DIR
    bstr = args.backbone if not args.backbone is None else ""
    # Ensure that ./results directory exists
    #create_dir(args.resdir)
    args.resdir += args.problem + '/'
    # Ensure ./results/<problem> exists
    #create_dir(args.resdir)
    if args.N:
        args.resdir += 'N' + str(args.N) + 'k' + str(args.k) + "test" + str(args.k_test) + '/' 
    else:
        args.resdir += 'k' + str(args.k) + "test" + str(args.k_test) + '/' 
    # Ensure ./results/<problem>/k<k>test<k_test> exists
    #create_dir(args.resdir)
    if args.model_spec is None:
        args.resdir += args.model + '/'
    else:
        args.resdir += args.model_spec + '/'
    # Ensure ./results/<problem>/k<k>test<k_test>/<model>/ exists
    #create_dir(args.resdir)

    
    #args.resdir += f"{bstr}-runs/"


    test_loaders = [test_loader]
    filenames = [args.resdir+f"{args.backbone}-test_scores.csv"]
    loss_filenames = [args.resdir+f"{args.backbone}-test_losses-T{conf['T_test']}.csv"]

    if args.eval_on_train:
        train_classes = args.N

        loader = ds(root="./data/", num_classes_per_task=args.N, meta_train=True, 
                        meta_val=False, meta_test=False, meta_split="train",
                        transform=Compose([Resize(size=img_size), ToTensor()]),
                        target_transform=Compose([Categorical(args.N)]),
                        download=True)
        loader = ClassSplitter(loader, shuffle=True, num_train_per_class=args.k_train, num_test_per_class=args.k_test)
        loader = BatchMetaDataLoader(loader, batch_size=1, num_workers=4, shuffle=True)
        test_loaders.append(loader)
        filenames.append(args.resdir+f"{args.backbone}-train_scores.csv")
        loss_filenames.append(args.resdir+f"{args.backbone}-train_losses-T{conf['T_test']}.csv")
    if args.cross_eval:
        test_loaders.append(cross_loader)
        filenames.append(args.resdir+f"{args.backbone}-cross_scores.csv")
        loss_filenames.append(args.resdir+f"{args.backbone}-cross_losses-T{conf['T_test']}.csv")        

    return args, conf, train_loader, val_loader, test_loaders, [filenames, loss_filenames], model_constr

def get_save_paths(resdir):
    fn = "test_scores"
    files = [resdir+x for x in os.listdir(resdir) if fn in x]

    print("obtained resdir:", resdir)

    if len(files) == 0 :
        print("Could not find results file.")
        import sys; sys.exit()
    elif len(files) > 1:
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("@ WARNING: MORE THAN 1 RESULT FILES FOUND. MERGING THEM.")
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

    prefixes = ["/".join(x.split('/')[:-1])+'/'+x.split('/')[-1].split('-')[0][:-4] for x in files]
    seeds = set([x.split('/')[-1].split('-')[0] for x in files])

    print(files, prefixes)
    dfs = []
    lens = []
    for file in files:
        df = pd.read_csv(file)
        print(df)
        dfs.append(df)
        lens.append(len(df))
        
    df = pd.concat(dfs)
    print("here:")
    print(df)
    perfs = df["mean_loss"] 
    q1 = perfs.quantile(0.25)
    q3 = perfs.quantile(0.75)
    iqr = q3 - q1

    sub = (perfs >= q1 - 1.5*iqr) & (perfs <= q3 + 1.5*iqr)
    models_to_use = np.where(sub)[0]

    if len(seeds) > 1:
        print("Multiple seeds detected, loading best model for each")
        models_to_load = np.array(prefixes)[models_to_use]
        print("Will load:", [p+'model.pkl' for p in prefixes])
        return [p+'model.pkl' for p in prefixes]


    # model paths that should be locked and loaded
    mfiles = []
    # partition counter
    pc = 0
    # global counter, model id
    for gc, mid in enumerate(models_to_use):
        if mid > sum(lens[:pc+1]) - 1:
            pc += 1
        name = prefixes[pc]+f"model-{mid-sum(lens[:pc])}.pkl"
        mfiles.append(name)
        print("Will load", name)
    
    return mfiles


     
def body(args, conf, train_loader, val_loader, test_loaders, files, model_constr):
    """Create and apply the meta-learning algorithm to the chosen data
    
    Backbone of all experiments. Responsible for:
    1. Creating the user-specified model
    2. Performing meta-training
    3. Performing meta-validation
    4. Performing meta-testing
    5. Logging and writing results to output channels
    
    Parameters
    -----------
    args : arguments
        Parsed command-line arguments
    conf : dictionary
        Configuration dictionary with all model arguments required for construction
    data_loader : DataLoader
        Data loder object which acts as access point to the problem data
    model_const : constructor fn
        Constructor function for the meta-learning algorithm to use
    
    """
    

    create_dir("similarities")
    create_dir(f"similarities/{args.problem}2{args.sim_problem}")
    create_dir(f"similarities/{args.problem}2{args.sim_problem}/N{args.N}k{args.k}test{args.k_test}/") 
    create_dir(f"similarities/{args.problem}2{args.sim_problem}/N{args.N}k{args.k}test{args.k_test}/{args.backbone}/") 

    if not args.avg_runs:
        models = [model_constr(**conf)]
        save_paths = [args.resdir+"model.pkl"]
        if not args.random_init:
            models[0].read_file(save_paths[0])
            models[0].sine=True
    else:
        save_paths = get_save_paths(args.resdir) #[args.resdir+x for x in os.listdir(args.resdir) if "model-" in x]
        models = [model_constr(**conf) for _ in save_paths]
        if not args.random_init:
            for mid, model in enumerate(models):
                print("Loading model from", save_paths[mid])
                model.read_file(save_paths[mid])
                model.sine = True
        else:
            for model in models:
                model.sine=True
            
    # Set seed and next test seed to ensure test diversity
    set_seed(args.test_seed)        

    CKAS = [[] for _ in range(len(save_paths))]
    ACCS = [[] for _ in range(len(save_paths))]
    DISTS = [[] for _ in range(len(save_paths))]

    # Just test, and call evaluate with argument cka=True
    if args.sine:
        test_loader = args.data_loader.generator(reset_ptr=True, episodic=True, mode="test", batch_size=args.test_batch_size)
    for eid, epoch in tqdm(enumerate(test_loader)):
        print(f"Episode {eid}")
        for mid, model in enumerate(models):
            #model.to(torch.cuda.current_device())
            train_x, train_y, test_x, test_y  = epoch
            acc, ckas, dists = model.evaluate(
                    train_x = train_x, 
                    train_y = train_y, 
                    test_x = test_x, 
                    test_y = test_y, 
                    val=False, #real test! no validation anymore
                    compute_cka=True
            )
            #model.to("cpu")
            #torch.cuda.empty_cache()
            CKAS[mid].append(ckas)
            ACCS[mid].append(acc)
            DISTS[mid].append(dists)

    
    for mid, (model_CKAS, model_DISTS) in enumerate(zip(CKAS, DISTS)):
        mCKAS = np.array(model_CKAS)
        averaged = mCKAS.mean(axis=0)
        std = mCKAS.std(axis=0)
        mDISTS = np.array(model_DISTS)
        averaged_dist = mDISTS.mean(axis=0)
        dist_std = mDISTS.std(axis=0)

        base = f"similarities/{args.problem}2{args.sim_problem}/N{args.N}k{args.k}test{args.k_test}/{args.backbone}/"
        if args.random_init:
            save_path = base + args.model_spec + f"-randominit-model{mid}.cka"
            dist_save_path = base + args.model_spec + f"-randominit-model{mid}.dist"
        else:
            save_path = base + args.model_spec + f"-model{mid}.cka"
            dist_save_path = base + args.model_spec + f"-model{mid}.dist"

        with open(save_path, "w+") as f:
            f.writelines([",".join([str(x) for x in averaged])+"\n", ",".join([str(x) for x in std])+"\n"])
        
        with open(dist_save_path, "w+") as f:
            f.writelines([",".join([str(x) for x in averaged_dist])+"\n", ",".join([str(x) for x in dist_std])+"\n"])
        
        print(f"Model {mid} accuracy: {np.mean(ACCS[mid]):.3f}")


def set_seed(seed):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

if __name__ == "__main__":
    # Parse command line arguments
    args, unparsed = FLAGS.parse_known_args()

    # If there is still some unparsed argument, raise error
    if len(unparsed) != 0:
        raise ValueError(f"Argument {unparsed} not recognized")
    
    # Set device to cpu if --cpu was specified
    if args.cpu:
        args.dev="cpu"
    
    # If cpu argument wasn't given, check access to CUDA GPU
    # defualt device is cuda:1, if that raises an exception
    # cuda:0 is used
    if not args.cpu:
        print("Current device:", torch.cuda.current_device())
        print("Available devices:", torch.cuda.device_count())
        if not args.devid is None:
            torch.cuda.set_device(args.devid)
            args.dev = f"cuda:{args.devid}"
            print("Using cuda device: ", args.dev)
        else:
            if not torch.cuda.is_available():
                raise RuntimeError("GPU unavailable.")
            
            try:
                torch.cuda.set_device(1)
                args.dev="cuda:1"
            except:
                torch.cuda.set_device(0)
                args.dev="cuda:0"

    # Let there be reproducibility!
    set_seed(args.seed)
    print("Chosen seed:", args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    args.test_seed = random.randint(0,100000)

    # Let there be recognizability!
    print(BANNER)
    print(NAMETAG)

    # Let there be structure!
    pargs, conf, train_loader, val_loader, test_loaders, files, model_constr = setup(args)


    # Let there be beauty!
    body(pargs, conf, train_loader, val_loader, test_loaders, files, model_constr)