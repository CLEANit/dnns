#!/usr/bin/env python

import yaml
import argparse
import pprint
import time

class Config:
    def __init__(self):
        
        print('███ Initializing Config...', end='')
        start = time.time()
        # set the default config
        self.defaultConfig()
        
        self.parseArgs()
        self.parseConfig()
        print('done. Elapsed time:', time.time() - start, '███')
        self.printConfig()




    def defaultConfig(self):
        self.config = {}
        self.config['tasks'] = 1
        self.config['cpus_per_task'] = 1
        self.config['gpus'] = 1
        self.config['epochs'] = 10
        self.config['batch_size'] = 128
        self.config['learning_rate'] = 1e-3
        self.config['training_label'] = 'X'
        self.config['testing_label'] = 'Y'

    def parseArgs(self):
        parser = argparse.ArgumentParser(description='Machine learning with Pytorch. Change dnn.py and your YAML input file to modify training.')
        parser.add_argument('-lr', '--local_rank', default=0, type=int,
                            help='ranking within the nodes')
        parser.add_argument('-i', '--input', default='input.yaml', type=str)
        self.args = parser.parse_args()

    def parseConfig(self):
        yaml_config = yaml.safe_load(open(self.args.input, 'r'))
        for key in yaml_config:
            self.config[key] = yaml_config[key]

    def printConfig(self):
        pp = pprint.PrettyPrinter(indent=4)
        print('███ Here is your configuration: ███')
        print('███████████████████████████████████')
        pp.pprint(self.config)
        print('███████████████████████████████████')

        