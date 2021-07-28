#!/usr/bin/env python
import os
import json
import torch
import pprint
import argparse
import importlib
import numpy as np
import shutil
import matplotlib

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

matplotlib.use("Agg")

from config import system_configs
from net.network import NetworkFactory
from db.datasets import datasets
torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Test CenterNet")
    parser.add_argument("--cfg_file",default="cnn",  help="config file", type=str)
    parser.add_argument("--testiter", dest="testiter",
                        help="test at iteration i",
                        default=
                        6000, type=int)
    parser.add_argument("--split", dest="split",
                        help="which split to use",
                        default="validation", type=str)
    parser.add_argument("--suffix", dest="suffix", default=None, type=str)
    parser.add_argument("--debug", default=True,action="store_true")

    args = parser.parse_args()
    return args


def make_dirs(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)


def ttest(split, testiter, debug=False, suffix=None):
    result_dir = system_configs.result_dir
    result_dir = os.path.join(result_dir, str(testiter), split)

    if suffix is not None:
        result_dir = os.path.join(result_dir, suffix)

    make_dirs([result_dir])

    test_iter = system_configs.max_iter if testiter is None else testiter
    print("loading parameters at iteration: {}".format(test_iter))
    nnet = None
    print("building neural network...")
    nnet = NetworkFactory()
    print("loading parameters...")
    nnet.load_params(test_iter)



    nnet.cuda()
    nnet.eval_mode()
    # test_file = "test.{}".format('remote_test_ngt')
    # test_file = "test.{}".format('remote')
    # test_file = "test.{}".format('remote_center')

    # test_file = "test.{}".format('remote_offset_ngt')
    # test_file = "test.{}".format('remote_ngt_crn')
    test_file = "test.{}".format('remote_ngt')
    # test_file = "test.{}".format('remote_mpn_ngt')
    testing = importlib.import_module(test_file).testing
    testing(nnet, result_dir, debug=debug)



if __name__ == "__main__":
    args = parse_args()

    if args.suffix is None:
        cfg_file = os.path.join(system_configs.config_dir, args.cfg_file + ".json")
    else:
        cfg_file = os.path.join(system_configs.config_dir, args.cfg_file + "-{}.json".format(args.suffix))
    print("cfg_file: {}".format(cfg_file))

    with open(cfg_file, "r") as f:
        configs = json.load(f)

    configs["system"]["snapshot_name"] = args.cfg_file
    system_configs.update_config(configs["system"])

    train_split = system_configs.train_split
    val_split = system_configs.val_split
    test_split = system_configs.test_split

    split = {
        "training": train_split,
        "validation": val_split,
        "testing": test_split
    }[args.split]

    print("loading all datasets...")
    dataset = system_configs.dataset
    print("split: {}".format(split))
    # testing_db = datasets[dataset](configs["db"], split)
    #
    # print("system config...")
    # pprint.pprint(system_configs.full)
    #
    # print("db config...")
    # pprint.pprint(testing_db.configs)



    ttest(args.split, args.testiter, args.debug, args.suffix)
