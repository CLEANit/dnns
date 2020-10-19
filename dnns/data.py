#!/usr/bin/env python
import torch
import h5py
import os
import logging
import numpy as np
import time
import sys
import matplotlib.pyplot as plt
import subprocess
import hashlib

def dataSplit(fname, test_pct, hash_on_key):
    """
    Split a HDF5 dataset into a training file and testing file.

    Parameters
    ----------
    fname (string): Name of file to be written in the train and test directory.
    test_pct (float): Percentage of test.
    hash_on_key (string): Select which dataset to perform a hash on. This hash allows us to know
    for sure that no element in the test set is in the training set.

    """
    file = h5py.File(fname, 'r')
    subprocess.call(['mkdir -p train test'], shell=True)
    training_file = h5py.File('train/' + fname , 'w')
    testing_file = h5py.File('test/' + fname , 'w')

    for key, val in file.attrs.items():
        testing_file.attrs[key] = val
        training_file.attrs[key] = val

    n_images = file[list(file.keys())[0]].shape[0]

    n_testing = int(n_images * test_pct)
    n_training = n_images - n_testing
    for key, val in file.items():
        training_file.create_dataset(key, (n_training,) + file[key].shape[1:])
        testing_file.create_dataset(key, (n_testing,) + file[key].shape[1:])

    hashes = []
    print('Hashing dataset:', hash_on_key)
    for elem in getProgressBar()(file[hash_on_key]):
        h = hashlib.sha256(elem).hexdigest()
        hashes.append(h)

    hashes = np.array(hashes)
    sorted_inds = np.argsort(hashes)
    sorted_hashes = hashes[sorted_inds]

    test_indices = sorted_inds[:n_testing]

    # we have to check and make sure the last element is not a duplicate
    duplicate = True
    while duplicate:
        for elem in sorted_inds[n_testing:]:
            if elem != test_indices[-1]:
                duplicate = False
            else:
                n_testing += 1
                n_training -= 1

    print('Test pct:', float(n_testing / n_images))
    print('Train pct:', float(n_training / n_images))

    assert float(n_testing / n_images) + float(n_training / n_images) == 1.0, 'Test and Train Percentages should add up to 1.0.'
    test_indices = np.sort(sorted_inds[:n_testing])
    train_indices = np.sort(sorted_inds[n_testing:])
    



    # this can be made to fit in a certain amount of memory
    # 1024 for now
    write_batch_size = 1024
    n_test_batches = int(test_indices.shape[0] / write_batch_size) + 1
    n_train_batches = int(train_indices.shape[0] / write_batch_size) + 1
    for key in file.keys():
        print('Dataset name:', key)
        print('Writing test data...')
        for i in getProgressBar()(range(n_test_batches)):
            start = i * write_batch_size
            end = (i + 1) * write_batch_size
            if end > len(test_indices):
                end = len(test_indices)
            testing_file[key][start:end] = file[key][test_indices[start:end]]
        print('Writing train data...')
        for i in getProgressBar()(range(n_train_batches)):
            start = i * write_batch_size
            end = (i + 1) * write_batch_size
            if end > len(train_indices):
                end = len(train_indices)
            training_file[key][start:end] = file[key][train_indices[start:end]]

    training_file.close()
    testing_file.close()
    file.close()

class HDF5Dataset(torch.utils.data.Dataset):
    """
    HDF5 Dataset class which wraps the torch.utils.data.Dataset class.

    Parameters
    ----------
    filename (string): HDF5 filename.
    x_label (string): Dataset label for the input data.
    y_label (string): Dataset label for the output data.
    rank (int): Rank of the process that is creating this object.
    use_hist (bool): Generate a histogram and use metropolis sampling to select training examples. This is experimental.
    """
    def __init__(self, filename, x_label, y_label, rank, use_hist=False):
        """
        Initialization of the class. 
        """
        super(HDF5Dataset, self).__init__()
        self.filename = filename
        self.x_label = x_label
        self.y_label = y_label
        self.rank = rank
        self.use_hist = use_hist
        self.h5_file = h5py.File(filename, 'r')
        self.length = self.h5_file[x_label].shape[0]
        self.checkDataSize()

    def checkDataSize(self):
        """
        Check the dataset size and if larger than 32 GB than read from disk, else load into memory.
        """

        max_size = 32 * 1e9 # roughly 32 GB
        if np.prod(self.h5_file[self.x_label].shape) * 8 >  max_size:
            if self.rank == 0:
                print('Data from file ' + self.filename + ' is too large (> 32 GB), will read from disk on the fly.')
            self.X = self.h5_file[self.x_label]
            self.Y = self.h5_file[self.y_label]
        else:
            if self.rank == 0:
                print('Loading file ' + self.filename + ' into memory.')
            self.X = self.h5_file[self.x_label][:]
            self.Y = self.h5_file[self.y_label][:]

        if self.use_hist:
            print('Preparing histogram for uniform data sampling.')
            n_bins = 64
            self.hist, self.bins = np.histogram(np.linalg.norm(self.Y, axis=-1), bins=n_bins, density=True)
            self.hist_indices = np.arange(self.hist.shape[0])
            self.hist /= n_bins
            self.distro = 1 - self.hist
            self.distro /= self.distro.sum()
    
    def __getitem__(self, index):
        if self.use_hist:
            good_choice = False
            while not good_choice:
                try:
                    bin_select = np.random.choice(self.hist_indices, p=self.distro)
                    left = self.bins[bin_select]
                    right = self.bins[bin_select + 1]
                    vals = np.linalg.norm(self.Y, axis=-1)
                    index = np.random.choice(np.argwhere(np.logical_and(vals > left, vals <= right))[0])
                    good_choice = True
                except:
                    pass
        item_x, item_y = self.X[index], self.Y[index]
        return torch.from_numpy(item_x.astype('float32')), torch.from_numpy(item_y.astype('float32'))

    def __len__(self):
        return self.length

class TwinHDF5Dataset(torch.utils.data.Dataset):
    def __init__(self, filename, x_label, y_label, n_samples, rank, use_hist=False):
        super(TwinHDF5Dataset, self).__init__()
        self.filename = filename
        self.x_label = x_label
        self.y_label = y_label
        self.rank = rank
        self.h5_file = h5py.File(filename, 'r')
        self.max_len = self.h5_file[x_label].shape[0]
        self.length = n_samples
        self.use_hist = use_hist
        # self.indices = np.indices((self.max_len, self.max_len)).T.reshape(self.length, 2)
        self.checkDataSize()

    def checkDataSize(self):
        max_size = 32 * 1e9 # roughly 32 GB
        if np.prod(self.h5_file[self.x_label].shape) * 8 >  max_size:
            if self.rank == 0:
                print('Data from file ' + self.filename + ' is too large (> 32 GB), will read from disk on the fly.')
            self.X = self.h5_file[self.x_label]
            self.Y = self.h5_file[self.y_label]
        else:
            if self.rank == 0:
                print('Loading file ' + self.filename + ' into memory.')
            self.X = self.h5_file[self.x_label][:]
            self.Y = self.h5_file[self.y_label][:]

        if self.use_hist:
            n_bins = 256
            diffs = np.zeros(self.length)
            indices = np.zeros((self.length, 2), dtype=np.int64)
            for i in range(self.length):
                index1 = np.random.randint(self.max_len)
                index2 = np.random.randint(self.max_len)
                diffs[i] = self.Y[index1] - self.Y[index2]
                indices[i][0] = index1
                indices[i][1] = index2
            self.diffs = diffs
            self.indices = indices
            # plt.hist(diffs, bins=256, density=True)
            # plt.show()
            self.hist, self.bins = np.histogram(diffs, bins=n_bins, density=True)
            self.hist_indices = np.arange(self.hist.shape[0])
            self.hist /= n_bins 
            self.distro = 1 - self.hist
            self.distro /= self.distro.sum()
            # plt.plot(self.distro)
            # plt.show()

    def __getitem__(self, index):
        if self.use_hist:
            good_choice = False
            while not good_choice:
                try:
                    bin_select = np.random.choice(self.hist_indices, p=self.distro)
                    left = self.bins[bin_select]
                    right = self.bins[bin_select + 1]
                    index = np.random.choice(np.argwhere(np.logical_and(self.diffs > left, self.diffs <= right))[0])
                    index1, index2 = self.indices[index, 0], self.indices[index, 1]
                    good_choice = True
                except:
                    pass
        else:
            index1 = np.random.randint(self.max_len)
            index2 = np.random.randint(self.max_len)
        item_x1, item_y1 = self.X[index1], self.Y[index1]
        item_x2, item_y2 = self.X[index2], self.Y[index2]
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
            self.args.local_rank,
            use_hist=self.config['use_hist']
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
    """
    The main data class for training. Set's up the HDF5 torch Dataset classes for parallel or non-parallel training. 

    Parameters
    ----------
    loader (class): The class that loads the data from disk. See loader module for more information.
    config (class): The class that holds the YAML configuration. See config module for more information.
    args (argparse object): Argparse object that holds the command line arguments.
    """
    def __init__(self, loader, config, args):
        self.loader = loader
        self.config = config
        self.args = args

        self.training_dataset = HDF5Dataset(
            self.loader.getTrainingFiles()[0],
            self.config['input_label'],
            self.config['output_label'],
            self.args.rank,
            use_hist=self.config['use_hist']
        )

        self.testing_dataset = HDF5Dataset(
            self.loader.getTestingFiles()[0],
            self.config['input_label'],
            self.config['output_label'],
            self.args.rank,
            use_hist=False
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
        """
        Get the dataloader for training data. If the world size is greater than 1, a training sampler is used.

        Returns
        -------
        A torch DataLoader.
        """
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
        """
        Get the dataloader for testing (or validation) data. If the world size is greater than 1, a testing sampler is used.

        Returns
        -------
        A torch DataLoader.
        """
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
