# dnns - general purpose package for training deep neural networks with custom HDF5 datasets. 

After many iterations of deep learning scripts, I have finally decided to have one package - with data loading tools to do deep learning.

## Some information about the package:
  1. This package utilizes pytorch.
  2. This package can also utilize mixed-precision training using NVIDIA's apex package.
  3. There is an executable titled 'worker.py' that is installed with this package. If you are not interested in going through the process of writing a script that does training (or you have done this already and don't want to do it again), then you can just use it (see steps of how to use it below).
  4. This script can be used for multi-node, multi-gpu training (see below for more information).
  
## Quickstart:
You **must** have 2 things (1 other thing is recommended). 

  1) A network (using the ordinary class structure that pytorch uses) written to a file (default is dnn.py) with a network class defined to be DNN. 
  2) A train and test directory with your HDF5 data sitting in it. The default dataset labels that the loader will read are 'X' and 'Y', which represent input and output data.

  3) (Recommended) A YAML file (default name is input.yaml) where you can configure the training protocol. 
  
To run, you can simply run:
```
python -m torch.distributed.launch -nnodes <n_nodes> --nproc_per_node <n_gpus_per_node> worker.py
```

## Slowstart:
Let's say you are in some directory called `some_dir`. We type `ls` and see:
```
~some_dir $ ls
input.yaml  dnn.py  train/  test/
```
Let's say you have HDF5 files called `training.h5` and `testing.h5` located in `train/` and `test/` with dataset labels `input_data` and `output_data`. Our configuration file `input.yaml` could look something like this:
```
~some_dir $ more input.yaml
# the number of epochs to run
n_epochs: 2000

# batch size of images
batch_size: 512

# learning rate for model
learning_rate: 0.00001

# number of threads for each GPU to use for data queuing
n_workers: 6 

# labels in the HDF5 files
x_label: 'input_data'
y_label: 'output_data'

mixed_precision: false
```
Here we have defined the number of epochs, batch size, learning rate, the number of worker threads, dataset labels, and we have turned mixed precision off. Now let's look at `dnn.py`. 
```
~some_dir $ more dnn.py
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

class DNN(nn.Module):

    def __init__(self, input_shape):
        super().__init__()
        layers = OrderedDict()
        layers['conv_red_1'] = nn.Conv3d(1, 64, 5, padding=2, stride=2)
        layers['conv_red_1_elu'] = nn.ELU()
        layers['conv_red_2'] = nn.Conv3d(64, 64, 5, padding=2, stride=1)
        layers['conv_red_2_elu'] = nn.ELU()
	
        layers['conv_nonred_3'] = nn.Conv3d(64, 16, 5, padding=2)
        layers['conv_nonred_3_elu'] = nn.ELU()
        for i in range(4, 9):
            layers['conv_nonred_' + str(i)] = nn.Conv3d(16, 16, 5, padding=2)
            layers['conv_nonred_' + str(i) + '_elu'] = nn.ELU()

        layers['conv_red_3'] = nn.Conv3d(16, 64, 5, padding=2, stride=1)
        layers['conv_red_3_elu'] = nn.ELU()

        layers['conv_nonred_9'] = nn.Conv3d(64, 32, 5, padding=2, stride=1)
        layers['conv_red_9_elu'] = nn.ELU()
        for i in range(10, 14):
            layers['conv_nonred_' + str(i)] = nn.Conv3d(32, 32, 5, padding=2)
            layers['conv_nonred_' + str(i) + '_elu'] = nn.ELU()
        
        layers['flatten'] = nn.Flatten()
        layers['fc1'] = nn.Linear((input_shape[0] //2 + 1) * (input_shape[1] //2 + 1) * (input_shape[2] //2 + 1) * input_shape[3] * 32, 1024 )
        layers['fc1_relu'] = nn.ELU()
        layers['fc2'] = nn.Linear(1024, 1)
        self.model = nn.Sequential(layers)


    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[-1], x.shape[1], x.shape[2], x.shape[3])  
        return self.model(x)
```
With these defined you can simply run:
```
python -m torch.distributed.launch -nnodes <n_nodes> --nproc_per_node <n_gpus_per_node> worker.py
```
Afterwards a checkpoint file `checkpoint.torch`, and a data file `loss_vs_epoch.dat` is created. 

## Limitations
1) Currently, you can only have one HDF5 file for training/testing.
2) There has not been a lot of testing of this repo. I expect there will be critical issues that must be taken care of in future releases.

## Contact
Please contact me at `kryczko@uottawa.ca` for any issues.
