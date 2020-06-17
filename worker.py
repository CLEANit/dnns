#!/usr/bin/env python

# globals
import os
import sys
import time
import torch
import torch.distributed as dist
from apex.parallel import DistributedDataParallel as DDP
from apex import amp

# locals
from dnns.loader import Loader
from dnns.data import Data
from dnns.config import Config


def getDNN(loader):
    sys.path.insert(1, os.getcwd())
    try:
        from dnn import DNN
        # print('Successfully imported your model.')
    except:
        # print('Could not import your model. Exiting.')
        exit(-1)
    return DNN(loader.getXShape())

def getModel(loader, config, args):
    model = getDNN(loader).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    if config['mixed_precision']:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O2')
    
    if args.world_size > 1:
        model = DDP(model)

    loss_fn = torch.nn.MSELoss().cuda()
    return model, optimizer, loss_fn

def train(training_set, model, optimizer, loss_fn, args, config):
    model.train()
    epoch_loss = 0.0
    counter = 0
    start = time.time()
    for i, data in enumerate(training_set, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        if config['mixed_precision']:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        # print statistics
        if args.world_size > 1:
            current_loss = reduce_tensor(loss.data)
        else:
            current_loss = loss.data
        epoch_loss += current_loss
        counter += 1
        # batch += 1
        if args.local_rank == 0:
            print('batch: %3.0i | current_loss: %0.3e | time: %0.3e' % (i, current_loss, time.time() - start)) 
            sys.stdout.flush()

    return epoch_loss / counter

def validate(testing_set, model, loss_fn, args, epoch):
    model.eval()
    val_loss = 0
    val_counter = 0
    if args.local_rank == 0:
        f = open('true_vs_pred_epoch_%04d.dat' % (epoch), 'w')

    for data in testing_set:
        inputs, labels = data
        inputs = inputs.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        if args.world_size > 1:
            val_loss += reduce_tensor(loss.data)
        else:
            val_loss = loss.data
        val_counter += 1

        # outputs = reduce_tensor(outputs)

        if args.local_rank == 0:
            for elem_t, elem_p in zip(labels, outputs):
                for t, p in zip(elem_t.data.cpu().numpy(), elem_p.data.cpu().numpy()):
                    f.write('%1.20e\t%1.20e\t' %(t, p))
                f.write('\n')

    if args.local_rank == 0:
        f.close()

    return val_loss / val_counter

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= int(os.environ['WORLD_SIZE'])
    return rt

def checkpoint(epoch, model, optimizer, checkpoint_path):
    torch.save({
        'epoch' : epoch,
        'model_state_dict' : model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict()
    }, checkpoint_path)

def tryToResume(model, optimizer, checkpoint_path, args):

    try:
        checkpoint = torch.load(checkpoint_path, map_location = lambda storage, loc: storage.cuda(args.local_rank))
        if args.local_rank == 0:
            print('The checkpoint was loaded successfully. Continuing training.')
    except FileNotFoundError:
        if args.local_rank == 0:
            print('There was no checkpoint found. Training from scratch.')
        checkpoint = None

    if checkpoint is None:
        start_epoch = 0
        # batch = 0
        # loss_vs_batch = open('loss_vs_batch.dat', 'w')
        loss_vs_epoch = open('loss_vs_epoch.dat', 'w')
    else:
        model.load_state_dict((checkpoint['model_state_dict']))
        optimizer.load_state_dict((checkpoint['optimizer_state_dict']))
        start_epoch = checkpoint['epoch']
        # batch = int(np.loadtxt('loss_vs_batch.dat')[-1][0])
        # loss_vs_batch = open('loss_vs_batch.dat', 'a')
        loss_vs_epoch = open('loss_vs_epoch.dat', 'a')
    return model, optimizer, start_epoch, loss_vs_epoch

def main():

    # get config
    Conf = Config()
    config = Conf.getConfig()
    args = Conf.getArgs()
    args.world_size = int(os.environ['WORLD_SIZE'])

    # get data
    loader = Loader(args, config)
    if config['twin']:
        data = TwinData(loader, config, args)
    else:
        data = Data(loader, config, args)

    training_set = data.getTrainingData()
    testing_set = data.getTestingData()

    # get model
    if args.world_size > 1:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')

    model, optimizer, loss_fn = getModel(loader, config, args)

    checkpoint_path ='./checkpoint.torch'

    model, optimizer, start_epoch, loss_vs_epoch = tryToResume(model, optimizer, checkpoint_path, args)

    for epoch in range(start_epoch, config['n_epochs']):
        start = time.time()
        training_loss = train(training_set, model, optimizer, loss_fn, args, config)
        validation_loss = validate(testing_set, model, loss_fn, args, epoch)
        if args.local_rank == 0:
            print('epoch: %3.0i | training loss: %0.3e | validation loss: %0.3e | time(s): %0.3e' %
                      (epoch + 1, training_loss, validation_loss, time.time() - start))
            loss_vs_epoch.write('%10.20e\t%10.20e\t%10.20e\t%10.20e\n' % (epoch + 1, training_loss, validation_loss, time.time() - start))
            loss_vs_epoch.flush()
            checkpoint(epoch, model, optimizer, checkpoint_path)

if __name__ == '__main__':
    main()
