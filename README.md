# dnns - general purpose package for training deep neural networks with custom HDF5 datasets. 

After many iterations of deep learning scripts, I have finally decided to have one package - with data loading tools to do deep learning.

## Some information about the package:
  1. This package utilizes pytorch.
  2. This package can also utilize mixed-precision training using NVIDIA's apex package.
  3. There is an executable titled 'worker.py' that is installed with this package. If you are not interested in going through the process of writing a script that does training (or you have done this already and don't want to do it again), then you can just use it (see steps of how to use it below).
  4. This script can be used for multi-node, multi-gpu training (see below for more information).
  
## Using the package:
