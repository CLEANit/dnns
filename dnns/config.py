#!/usr/bin/env python

import yaml
import argparse
import pprint
import time
import os

class Config:
    """
    Class to handle configuration from input file and command line args.
    """
    def __init__(self):
        self.parseArgs()
        # set the default config
        self.defaultConfig()
        
        self.parseConfig()

    def defaultConfig(self):
        """
        This method generates default config variables. Any new variables meant to go in the YAML
        config file should go here as well. One should use the set() method.
        """
        self.config = {}
        self.config['cpus_per_task'] = 1
        self.config['batch_size'] = 128
        self.config['learning_rate'] = 1e-4
        self.config['l2_regularization'] = 0.
        self.config['input_label'] = 'X'
        self.config['output_label'] = 'Y'
        self.config['model'] = 'dnn.py'
        self.config['mixed_precision'] = True
        self.config['twin'] = False
        self.config['n_training_samples'] = 0
        self.config['n_testing_samples'] = 0
        self.config['use_hist'] = False

    def parseArgs(self):
        """
        Set up the argument parser with a few additions to handle multi-node training.
        """
        parser = argparse.ArgumentParser(description='Machine learning with Pytorch. Change dnn.py and your YAML input file to modify training.')
        parser.add_argument('-lr', '--local_rank', default=0, type=int,
                            help='ranking within the nodes')
        parser.add_argument('-nr', '--node_rank', default=0, type=int, help='Node number')
        parser.add_argument('-ng', '--gpus_per_node', default=1, type=int, help='Number of GPUs on node.')
        parser.add_argument('-i', '--input', default='input.yaml', type=str)
        parser.add_argument('-cp','--checkpoint_path', default='./', type=str )
        self.args = parser.parse_args()

    def parseConfig(self):
        """
        Parse the configuration from the input yaml file.
        """
        yaml_config = yaml.safe_load(open(self.args.input, 'r'))
        for key in yaml_config:
            self.config[key] = yaml_config[key]

    def printConfig(self):
        """
        Print out the configuration.
        """
        pp = pprint.PrettyPrinter(indent=4)
        print()
        print('Here is your configuration:')
        pp.pprint(self.config)
        print()

    def getConfig(self):
        """
        Get the input configuration.

        Returns
        -------
        A Dictionary which holds the configuration.
        """
        return self.config

    def getArgs(self):
        """
        Get the command line configuration.

        Returns
        -------
        A argparse object.
        """
        return self.args

    def set(self, key, val):
        """
        Add new variables to the configuration.
        """
        self.config[key] = val

        
