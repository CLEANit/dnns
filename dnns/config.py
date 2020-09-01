#!/usr/bin/env python

import yaml
import argparse
import pprint
import time
import os

class Config:
    def __init__(self):
        self.parseArgs()
        # set the default config
        self.defaultConfig()
        
        self.parseConfig()

    def defaultConfig(self):
        self.config = {}
        self.config['cpus_per_task'] = 1
        self.config['batch_size'] = 128
        self.config['learning_rate'] = 1e-3
        self.config['input_label'] = 'X'
        self.config['output_label'] = 'Y'
        self.config['model'] = 'dnn.py'
        self.config['mixed_precision'] = True
        self.config['twin'] = False
        self.config['n_training_samples'] = 0
        self.config['n_testing_samples'] = 0
        self.config['use_hist'] = False

    def parseArgs(self):
        parser = argparse.ArgumentParser(description='Machine learning with Pytorch. Change dnn.py and your YAML input file to modify training.')
        parser.add_argument('-lr', '--local_rank', default=0, type=int,
                            help='ranking within the nodes')
        parser.add_argument('-nr', '--node_rank', default=0, type=int, help='Node number')
        parser.add_argument('-ng', '--gpus_per_node', default=1, type=int, help='Number of GPUs on node.')
        parser.add_argument('-i', '--input', default='input.yaml', type=str)
        parser.add_argument('-cp','--checkpoint_path', default='./', type=str )
        self.args = parser.parse_args()

    def parseConfig(self):
        yaml_config = yaml.safe_load(open(self.args.input, 'r'))
        for key in yaml_config:
            self.config[key] = yaml_config[key]

    def printConfig(self):
        pp = pprint.PrettyPrinter(indent=4)
        print()
        print('Here is your configuration:')
        pp.pprint(self.config)
        print()

    def getConfig(self):
        return self.config

    def getArgs(self):
        return self.args

    def set(self, key, val):
        self.config[key] = val

        
