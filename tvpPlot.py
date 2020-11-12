#!/usr/bin/env python

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import sys
from cmocean import cm
import argparse

def parseArgs():
    """
    Set up the argument parser.
    """
    parser = argparse.ArgumentParser(description='Plot true versus predicted for a given epoch.')
    parser.add_argument('-f', '--file', default='', type=str,
                            help='file name in which to plot')
    parser.add_argument('-c', '--columns', help='Columns in which to plot', dest='columns', default=[0, 1], type=int, nargs='+')
    parser.add_argument('-s', '--save', dest='save', action='store_true', help='Save plot to file.')
    parser.add_argument('-l', '--latex', dest='latex', action='store_true', help='Use latex in text.')
    return parser.parse_args()

def main():
    """
    Plot true verus predicted for selected columns.
    """
    args = parseArgs()
    d = np.loadtxt(args.file)
    assert len(args.columns) == 2, 'Number of columns must be 2.'
    pred = d[:, args.columns[1]]
    true = d[:, args.columns[0]]

    if args.latex:
        font = {'family' : 'CMU Serif',
        #         'weight' : 'light',
        'size'   : 18}

        plt.rc('font', **font)
        plt.rc('text', usetex=True)

    print ("Median error            :", np.median(abs(pred - true)))
    print ("Median squared error    :", np.median((pred - true)**2))
    print ("Mean absolute error     :", np.mean(abs(pred - true)))
    print ("Max absolute error      :", np.max(abs(pred - true)))

    # color_distro = np.linspace(np.min(abs(true- pred)),np.max(abs(true- pred)), 100)
    color_distro, cumul = np.histogram(true, 100, density=True)
    truemax, truemin =  np.max(true), np.min(true)
    h = (truemax - truemin) / (100 - 1)
    colors = np.empty(len(true))
    x = np.linspace(np.min(true), np.max(true), 100)

    for i, t in enumerate(true):
    	colors[i] = color_distro[int((t - truemin) / h)]
    	
    fig, axes  = plt.subplots(1, 2, figsize=(12, 5))

    im = axes[0].scatter(true, pred, s=10, c=colors, cmap=cm.thermal_r)
    axes[0].plot(x,x, '--', color='k', markersize=20)
    # plt.colorbar(im, ax=axes[0])
    axes[0].grid(True)


    # plt.title('True vs. predicted distances for $H_2$', {'family': 'serif','fontsize': 15})
    axes[0].set_xlabel('True', {'family': 'serif','fontsize': 12})
    axes[0].set_ylabel('Predicted', {'family': 'serif','fontsize': 12})

    diffs = true - pred
    im = axes[1].scatter(true, true - pred, s=10, c=colors, cmap=cm.thermal_r)
    # axes[0].plot(x,x, '--', color='k', markersize=20)
    cbar = plt.colorbar(im, ax=axes[1])
    cbar.set_label('Distribution of Data')
    axes[1].grid(True)


    # plt.title('True vs. predicted distances for $H_2$', {'family': 'serif','fontsize': 15})
    axes[1].set_xlabel('True', {'family': 'serif','fontsize': 12})
    axes[1].set_ylabel('True - Predicted', {'family': 'serif','fontsize': 12})
    axes[0].set_aspect('equal', 'box')
    axes[1].set_ylim([np.mean(diffs) - 3 * np.std(diffs), np.mean(diffs) + 3 * np.std(diffs)])
    plt.tight_layout()
    if args.save:
        plt.savefig('true_vs_pred.pdf')
    plt.show()

if __name__ == '__main__':
    main()
