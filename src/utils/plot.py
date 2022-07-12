import numpy as np
import matplotlib.pyplot as plt

from src.array import add_jitter
from src.base import stderr

def tuning()

def ori_tuning()

def plot_psth(ax, psth,
              norm=False, xtype=''
              tightx=False):
    if gratings is True:
        psth_bins = np.arange(-1500, 1501)

    if tightx is True:


def plot_STAs

def plot_STVs

def plot_cat_scatter(ax, data
                        labels=None, colors=None, use_median=False, markersize=2,
                        template=None):
    """ Categorical scatter plot
    With categories as columns

    data should be a list of arrays, where each list item is an array of values for that category
    not an array of arrays ebcause we want to be able to have different numbers of samples in each cat
    
    """

    labels = np.arange(0, len(data), 1)

    if template == 'eyemov':
        labels = ['early', 'late', 'biphasic', 'negative']

    for l, label in enumerate(labels):

        d = data[l]

        x_jitter = add_jitter(center=l, size=np.size(d,0), scale=0.2)

        ax.plot(x_jitter, d, '.', color=colors[c], markersize=markersize)

        # Add a horizontal line for the median
        if use_median:
            hline = np.nanmedian(cluster_data)
        elif not use_median:
            hline = np.nanmean(cluster_data)
        ax.hlines(hline, c-0.2, c+0.2, color='k', linewidth=2)

        # Add a vertical line for the std err
        err = stderr(cluster_data)
        ax.vlines(c, hline-err, hline+err, color='k', linewidth=2)

        ax.set_xticks(range(4), ['early','late','biphasic','negative'])