#!/usr/bin/env python

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import to_rgba_array
import numpy as np
import sys
from cmocean import cm
import argparse
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def parseArgs():
    """
    Set up the argument parser.
    """
    parser = argparse.ArgumentParser(description='Plot true versus predicted for a given epoch.')
    parser.add_argument('-f', '--file', default='', type=str,
                            help='file name in which to plot')
    parser.add_argument('-s', '--save', dest='save', action='store_true', help='Save plot to file.')
    parser.add_argument('-l', '--latex', dest='latex', action='store_true', help='Use latex in text.')
    parser.add_argument('--maxs', dest='maxs', default=None, nargs="+" , help='Maxs of data, default: 1.0', type=float)
    parser.add_argument('--mins', dest='mins', default=None, nargs="+", help='Mins of data, default: 0.0', type=float)
    parser.add_argument('--unit', dest='unit', default='', help='Units of values')
    parser.add_argument('--xlabel', dest='xlabel', default='')
    parser.add_argument('--ylabel', dest='ylabel', default='')
    return parser.parse_args()

def main():
    """
    Plot true verus predicted for selected columns.
    """
    args = parseArgs()
    d = np.loadtxt(args.file)
    if args.maxs is None:
        args.maxs = [1.] * d.shape[1]
    if args.mins is None:
        args.mins = [0.] * d.shape[1]

    pred = d[:,::2] 
    true = d[:,1::2]
    print(args.maxs, args.mins)
    for i in range(pred.shape[1]):
        diff = args.maxs[i] - args.mins[i]
        pred[:,i] *= diff
        pred[:,i] += args.mins[i]
        true[:,i] *= diff
        true[:,i] += args.mins[i]


    if args.latex:
        font = {'family' : 'CMU Serif',
        #         'weight' : 'light',
        'size'   : 18}

        plt.rc('font', **font)
        plt.rc('text', usetex=True)
        plt.rc('text.latex', preamble=r'\usepackage{bm}')

    print ("Median error            :", np.median(np.abs(pred - true)))
    print ("Median squared error    :", np.median((pred - true)**2))
    print ("Mean absolute error     :", np.mean(np.abs(pred - true)))
    print ("Root Mean Squared Error :", np.sqrt(np.mean((pred - true)**2)))
    print ("Max absolute error      :", np.max(np.abs(pred - true)))

    fig, ax  = plt.subplots(1, 1, figsize=(7, 5))
    colors = np.linspace(0, 1.0, pred.shape[1])
    #for i in range(pred.shape[1]):
    norm_true = np.linalg.norm(true, 2, axis=-1)
    norm_pred = np.linalg.norm(pred, 2, axis=-1)
    print(np.argmax(np.abs(norm_true - norm_pred)))
    x = np.linspace(np.min(norm_true), np.max(norm_true), 32)
    ax.plot(x,np.zeros_like(x),'--', c='gray')
    diffs = np.abs(norm_true - norm_pred)
    diffs /= (diffs.max() - diffs.min())
    im = ax.scatter(norm_true, norm_true - norm_pred, c=cm.haline_r(diffs))
    im.set_cmap(cm.haline_r)
    im.set_clim(np.abs(norm_true - norm_pred).min(), np.abs(norm_true - norm_pred).max())
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Absolute Difference ' + args.unit)
    ax.set_xlabel(args.xlabel + ' ' + args.unit)
    ax.set_ylabel(args.ylabel + ' ' + args.unit)
    ax.set_ylim([-np.std(diffs), 3 * np.std(diffs)])
    ax.grid(linestyle='-.')

    ax2 = inset_axes(ax, width="40%", height="40%")
    ax2.scatter(norm_true, norm_pred, c=cm.haline_r(diffs))
    ax2.grid(linestyle='-.')
    ax2.set_xlabel('True ' + args.unit)
    ax2.set_ylabel('Pred. ' + args.unit)
    ax2.plot(x,x, '--', c='gray')
    plt.tight_layout()
    # plt.show()

    # im = axes[0].scatter(true, pred, s=10, c=colors, cmap=cm.matter)
    # axes[0].plot(x,x, '--', color='k', markersize=20)
    # # plt.colorbar(im, ax=axes[0])
    # axes[0].grid(True)


    # # plt.title('True vs. predicted distances for $H_2$', {'family': 'serif','fontsize': 15})
    # if args.unit != '':
    #     axes[0].set_xlabel('True [{}]'.format(args.unit))
    #     axes[0].set_ylabel('Predicted [{}]'.format(args.unit))
    #     axes[1].set_xlabel('True [{}]'.format(args.unit))
    #     axes[1].set_ylabel('True - Predicted [{}]'.format(args.unit))
    # else:
    #     axes[0].set_xlabel('True')
    #     axes[0].set_ylabel('Predicted')
    #     axes[1].set_xlabel('True')
    #     axes[1].set_ylabel('True - Predicted')
    # diffs = true - pred
    # im = axes[1].scatter(true, true - pred, s=10, c=colors, cmap=cm.matter)
    # # axes[0].plot(x,x, '--', color='k', markersize=20)
    # cbar = plt.colorbar(im, ax=axes[1])
    # cbar.set_label('Distribution of Data')
    # axes[1].grid(True)


    # # plt.title('True vs. predicted distances for $H_2$', {'family': 'serif','fontsize': 15})
    # axes[0].set_aspect('equal', 'box')
    # axes[0].set_ylim([np.min(pred), np.max(pred)])
    # axes[0].set_xlim([np.min(true), np.max(true)])
    # axes[1].set_ylim([np.mean(diffs) - 10 * np.std(diffs), np.mean(diffs) + 10 * np.std(diffs)])
    # axes[1].set_xlim([np.min(true), np.max(true)])
    # plt.tight_layout()
    if args.save:
         plt.savefig('all_true_vs_pred.pdf')
    plt.show()

if __name__ == '__main__':
    main()
