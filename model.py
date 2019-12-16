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
from data import Data, DataFetcher



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
        self.config = Config(self.parser)

        # '''
        # Step 2: Load HDF5 data from train and test directories.
        # '''
        # self.loader = Loader(self.parser, self.config)
        # self.data = Data(self.loader, self.config)
        # self.train_fetcher = DataFetcher(self.data.getTrainingSet())
        # self.test_fetcher = DataFetcher(self.data.getTestingSet())

        # '''
        # Step 3: Load in the network
        # '''
        # self.dnn = self.getDNN()


    def train(self):
        pass


