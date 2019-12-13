#!/usr/bin/env python
import torch
import h5py
import os
import logging
import numpy as np
import time

class HDF5Dataset(torch.utils.data.Dataset):
    def __init__(self, filename, x_label, y_label):
        super(HDF5Dataset, self).__init__()
        self.filename = filename
        self.x_label = x_label
        self.y_label = y_label
        self.h5_file = h5py.File(filename, 'r')
        self.length = self.h5_file[x_label].shape[0]
        self.checkDataSize()

    def checkDataSize(self):
        print('Trying to load dataset (' + self.filename + ') into memory...', end='')
        try:
            self.X = self.h5_file[self.x_label][:]
            self.Y = self.h5_file[self.y_label][:]
            print('successful.')
        except:
            print('unsuccessful. Dataset is too large to fit in memory.')
            self.X = self.h5_file[self.x_label]
            self.Y = self.h5_file[self.y_label]

    def __getitem__(self, index):
        item_x, item_y = self.X[index], self.Y[index]
        return item_x.astype('float32'), item_y.astype('float32')

    def __len__(self):
        return self.length

class Data:
	self __init__(self, loader, config):
		self.loader = loader
		self.config = config

	def getTrainingData(self):
	    training_set = torch.utils.data.DataLoader(HDF5Dataset(
	                                                        self.loader.getTrainingFiles()[0],
	                                                        self.config['x_label'],
	                                                        self.config['y_label']),
	                                                        batch_size=self.config['batch_size'],
	                                                        shuffle=True,
	                                                        num_workers=self.config['n_workers'])
	  	return training_set
	def getTestingData(self):
	    test_set = torch.utils.data.DataLoader(HDF5Dataset(
	                                                        self.loader.getTestingFiles()[0],
	                                                        self.config['x_label'],
	                                                        self.config['y_label']),
	                                                        batch_size=self.config['batch_size'],
	                                                        shuffle=False,
	                                                        num_workers=self.config['n_workers'])
	    return test_set
