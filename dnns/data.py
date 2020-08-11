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
        max_size = 32 * 1e9 # roughly 32 GB
        if np.prod(self.h5_file[self.x_label].shape) * 8 >  max_size:
            if self.rank == 0:
                print('Data size is too large (> 32 GB), will read from disk.')
            self.X = self.h5_file[self.x_label]
            self.Y = self.h5_file[self.y_label]
        else:
            if self.rank == 0:
                print('Loading data into memory.')
            self.X = self.h5_file[self.x_label][:]
            self.Y = self.h5_file[self.y_label][:]

    def __getitem__(self, index):
        item_x, item_y = self.X[index], self.Y[index]
        return torch.from_numpy(item_x.astype('float32')), torch.from_numpy(item_y.astype('float32'))

    def __len__(self):
        return self.length

class TwinHDF5Dataset(torch.utils.data.Dataset):
    def __init__(self, filename, x_label, y_label, n_samples, rank):
        super(TwinHDF5Dataset, self).__init__()
        self.filename = filename
        self.x_label = x_label
        self.y_label = y_label
        self.rank = rank
        self.h5_file = h5py.File(filename, 'r')
        self.max_len = self.h5_file[x_label].shape[0]
        self.length = self.max_len // 2
        # self.indices = np.indices((self.max_len, self.max_len)).T.reshape(self.length, 2)
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
        item_x1, item_y1 = self.X[2*index], self.Y[2*index]
        item_x2, item_y2 = self.X[2*index + 1 ], self.Y[2*index + 1]
        return np.array([item_x1.astype('float32'), item_x2.astype('float32')]), item_y1.astype('float32') - item_y2.astype('float32')


    def __len__(self):
        return self.length

class TwinData:
    def __init__(self, loader, config, args):
        self.loader = loader
        self.config = config
        self.args = args

        self.training_dataset = TwinHDF5Dataset(
            self.loader.getTrainingFiles()[0],
            self.config['input_label'],
            self.config['output_label'],
            self.config['n_training_samples'],
            self.args.local_rank
        )

        self.testing_dataset = TwinHDF5Dataset(
            self.loader.getTestingFiles()[0],
            self.config['input_label'],
            self.config['output_label'],
            self.config['n_testing_samples'],
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
            rank=self.args.rank
        )

        self.test_sampler = torch.utils.data.distributed.DistributedSampler(
            self.testing_dataset,
            num_replicas=self.args.world_size,
            rank=self.args.rank
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
