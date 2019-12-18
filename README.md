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
Here we have defined the number of epochs, batch size, learning rate, the number of worker threads, dataset labels, and we have turned mixed precision off.
