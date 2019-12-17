#!/usr/bin/env python

# globals
import os
import argparse
import yaml
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from apex.parallel import DistributedDataParallel as DDP
from apex import amp

# locals
from loader import Loader
from data import Data
from config import Config

def train(gpu_num, model):

    args = model.args
    config = model.config
    loader = model.loader
    rank = args.local_rank * config['gpus'] + gpu
    print('Hello from rank:', rank)



    # data = Data(loader, config)

    # dnn = model.getDNN()



class Model:
    def __init__(self):
        self.initialize()

    def getDNN(self):
        sys.path.insert(1, os.getcwd())
        try:
            from dnn import DNN
            print('Successfully imported your model.')
        except:
            print('Could not import your model. Exiting.')
            exit(-1)
        return Net(self.loader.getXShape())

    def initialize(self):
        '''
        Step 1: Read in all of the configuration. This will come from argparse and YAML
        '''
        self.Conf = Config()
        self.args = self.Conf.getArgs()
        self.config = self.Conf.getConfig()

        '''
        Step 2: Load HDF5 data from train and test directories.
        '''
        self.loader = Loader(self.args, self.config)



    def train(self):
        mp.spawn(train, nprocs=self.config['gpus'], args=(self,))


