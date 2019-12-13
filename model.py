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
from torch.nn.parallel import DistributedDataParallel as DDP

# locals
from loader import Loader
from data import Data


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Explicitly setting seed to make sure that models created in two processes
    # start from same random weights and biases.
    torch.manual_seed(42)


def cleanup():
    dist.destroy_process_group()

def initDDP(rank, world_size, dnn, config):
    setup(rank, world_size)
    n_gpus_per_task = torch.cuda.device_count() // world_size
    device_ids = list(range(rank * n_gpus_per_task, (rank + 1) * n_gpus_per_task))

    model = dnn.to(device_ids[0])
    ddp_model = DDP(model, device_ids=device_ids)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(ddp_model.parameters(), lr=config['learning_rate'])

    return model, loss_fn, optimizer

class Model:
    def __init__(self):
        self.initialize()

    def parseArgs(self):
        parser = argparse.ArgumentParser(description='Regression using Pytorch. Change model.py to modify the deep neural network.')
        parser.add_argument('--train', help='Start training with data stored in "train" directory.', action='store_true', dest='train')
        return parser.parse_args()

    def parseConfig(self):
        if not os.path.isfile('./input.yaml'):
            print('Could not find "input.yaml". Please make sure this is in your working directory.')
            exit(-1)
        return yaml.safe_load(open('input.yaml'))

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
        self.parser = self.parseArgs()
        self.config = self.parseConfig()

        '''
        Step 2: Load HDF5 data from train and test directories.
        '''
        self.loader = Loader(self.parser, self.config)
        self.data = Data(self.loader, self.config)

        '''
        Step 3: Load in the network
        '''
        self.dnn = self.getDNN()

        '''
        Step 4: Set up DistributedDataParallel
        '''
        self.model, self.loss_fn, self.optimizer = mp.spawn(initDDP, args=(self.config['world_size'], self.dnn, self.config), nprocs=self.config['world_size'], join=True)

    def train(self):



    def getTrainingData(self):
        return self.data.getTrainingData()

    def getTestingData(self):
        return self.data.getTestingData()