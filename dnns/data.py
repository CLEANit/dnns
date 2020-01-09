#!/usr/bin/env python
import torch
import h5py
import os
import logging
import numpy as np
import time


class HDF5Dataset(torch.utils.data.Dataset):
    def __init__(self, filename, x_label, y_label, rank):
        super(HDF5Dataset, self).__init__()
        self.filename = filename
        self.x_label = x_label
        self.y_label = y_label
        self.rank = rank
        self.h5_file = h5py.File(filename, 'r')
        self.length = self.h5_file[x_label].shape[0]
        self.checkDataSize()

    def checkDataSize(self):
        # print('Rank', self.rank,'Trying to load dataset (' + self.filename + ') into memory...', end='')
        #try:
        #    self.X = self.h5_file[self.x_label][:]
        #    self.Y = self.h5_file[self.y_label][:]
            # print('successful.')
        # except:
            # print('unsuccessful. Dataset is too large to fit in memory.')
        self.X = self.h5_file[self.x_label]
        self.Y = self.h5_file[self.y_label]

    def __getitem__(self, index):
        item_x, item_y = self.X[index], self.Y[index]
        return item_x.astype('float32'), item_y.astype('float32')

    def __len__(self):
        return self.length

class Data:
    def __init__(self, loader, config, args):
        self.loader = loader
        self.config = config
        self.args = args

        self.training_dataset = HDF5Dataset(
            self.loader.getTrainingFiles()[0],
            self.config['input_label'],
            self.config['output_label'],
            self.args.local_rank
        )

        self.testing_dataset = HDF5Dataset(
            self.loader.getTestingFiles()[0],
            self.config['input_label'],
            self.config['output_label'],
            self.args.local_rank
        )

        self.train_sampler = torch.utils.data.distributed.DistributedSampler(
            self.training_dataset,
            num_replicas=self.args.world_size,
            rank=self.args.local_rank
        )

        self.test_sampler = torch.utils.data.distributed.DistributedSampler(
            self.testing_dataset,
            num_replicas=self.args.world_size,
            rank=self.args.local_rank
        )

    def getTrainingData(self):
        if self.args.world_size > 1:
            return torch.utils.data.DataLoader(
                self.training_dataset,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=self.config['cpus_per_task'],
                pin_memory=True,
                sampler=self.train_sampler
            )
        else:
            return torch.utils.data.DataLoader(
                self.training_dataset,
                batch_size=self.config['batch_size'],
                shuffle=True,
                num_workers=self.config['cpus_per_task']
            )

    def getTestingData(self):
        if self.args.world_size > 1:
            return torch.utils.data.DataLoader(self.testing_dataset,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=self.config['cpus_per_task'],
                pin_memory=True,
                sampler=self.test_sampler
            )
        else:
            return torch.utils.data.DataLoader(
                self.testing_dataset,
                batch_size=self.config['batch_size'],
                shuffle=True,
                num_workers=self.config['cpus_per_task']
            )
