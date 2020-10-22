#!/usr/bin/env

import h5py
import os
import numpy as np
import time
import random

class Loader():
    """
    Read in HDF5 files from train and test directory and perform some checks to make sure everything is okay.

    Parameters
    ----------
    parser (argparse object): Command line arguments handled by argparse.
    config (dict): Dictionary of configuration which was made from YAML. 

    """
    def __init__(self, parser, config):
        start = time.time()
        self.parser = parser
        self.config = config
        # print('Initializing Loader...', end='')
        self.mapping = {}

        # get the training files
        self.train_files = self.readDir(os.getcwd() + '/train')
        self.train_h5_files = self.prepareData(self.train_files)

        # get the testing files
        self.test_files = self.readDir(os.getcwd() + '/test')
        self.test_h5_files = self.prepareData(self.test_files, test=True)

        # print('done. (%5.5f s.)' % (time.time() - start))

    def readDir(self, dir_name):
        """
        Read all files in a directory. This is called in the __init__ function.

        Parameters
        ----------
        dir_name (str): Path in which files will be collected.

        Returns
        -------
        A list of files in the supplied path.
        """
        if os.path.isdir(dir_name):
            files = os.listdir(dir_name)
            if len(files) == 0:
                print('No files found in dir: %s' % dir_name)
                exit(-1)
            else:
                return [dir_name + '/' + elem for elem in  os.listdir(dir_name)]
        else:
            print('No dir called: %s, please put your h5 files in there.' % dir_name)


    def prepareData(self, files, test=False):
        """
        Check the shapes across all data sets to make sure all is good.

        Parameters
        ----------
        files (list): list of files to check.
        test (bool): Set to true if they are testing files. Default: False.

        Returns
        -------
        A list of h5py file objects.
        """

        h5_files = []
        filenames = []
        for fname in files:
            filenames.append(fname)
            h5_files.append(h5py.File(fname, 'r'))
            # self.mapping[fname] = h5py.File(fname, 'r')

        self.min = np.inf
        self.max = -np.inf
        self.x_shape = None
        self.y_shape = None
        if not test:
            self.total = 0
            self.image_counts = {}
        self.filenames = filenames

        for i, h5file in enumerate(h5_files):
            x_shape = h5file[self.config['input_label']].shape
            y_shape = h5file[self.config['output_label']].shape
            
            if x_shape[0] != y_shape[0]:
                print('The datasets X and Y must have the same length!')
                exit(-1)

            if not test:
                self.image_counts[filenames[i]] = y_shape[0]
                self.total += y_shape[0]

            min_y = np.min(h5file[self.config['output_label']])
            
            if min_y < self.min:
                self.min = min_y

            max_y = np.max(h5file[self.config['output_label']])
            if max_y > self.max:
                self.max = max_y

            if i > 0:
                if self.x_shape != x_shape[1:]:
                    print('All of the X datasets must have the same shape!')
                    exit(-1)
            else:
                self.x_shape = x_shape[1:]

            if i > 0:
                if self.y_shape != y_shape[1:]:
                    print('All of the Y datasets must have the same shape!')
                    exit(-1)
            else:
                self.y_shape = y_shape[1:]

            # h5file.close()

        return h5_files

    def getTotalImages(self):
        """
        Returns
        -------
        The total number of images used in training. 
        """
        return self.total

    def getImageCountsPerFile(self):
        """
        Returns
        -------
        A dict with training filenames as keys and number of images as values.
        """
        return self.image_counts

    def getMin(self):
        """
        Returns
        -------
        The minimum value of training and testing. 
        """
        return self.min

    def getMax(self):
        """
        Returns
        -------
        The maximim value of training and testing. 
        """
        return self.max

    def getXShape(self):
        """
        Returns
        -------
        The shape of the input data. 
        """
        return self.x_shape

    def getYShape(self):
        """
        Returns
        -------
        The shape of the output data. 
        """
        return self.y_shape

    def getTrainingH5Files(self):
        """
        Returns
        -------
        The list of training h5py file objects.
        """
        return self.train_h5_files

    def getTrainingFiles(self):
        """
        Returns
        -------
        The list of training filenames.
        """
        return self.train_files

    def getTestingH5Files(self):
        """
        Returns
        -------
        The list of testing h5py file objects.
        """
        return self.test_h5_files

    def getTestingFiles(self):
        """
        Returns
        -------
        The list of testing filenames.
        """
        return self.test_files
