"""Recording split.

Electrophysiology data from all stimuli (e.g. head-fixed white noise, freely moving, etc.),
before spike sorting, was merged into a single binary file. Before preprocessing can be run
on the data for each recording, this script must be run to split the ephys data back into
separate files. The boundries for each recording were saved into the .mat file, which you
will select from a dialogue box or give as an argument when running this script.

Example
-------
This is run using
  $ python -m fmEphys.recsplit
Once this is run, a window will open in which you will select the merge .mat file. Alternatively,
you can run it with
  $ python -m fmEphys.recsplit -f /path/to/merge.mat
to avoid selecting the .mat file in a window.

Notes
-----
  * If the data has moved since the .mat file was written, the file paths saved in the .mat
    will be wrong and point the code to the wrong place. You can make a new .mat file with
    correct paths using the Matlab script /matlab/updateBasePath.m

"""
import os
import argparse

import scipy.io
import numpy as np
import pandas as pd
import PySimpleGUI as sg

import fmEphys

def split_recordings(mergemat_path):
    """Split spike data at time boundries.

    Parameters
    ----------
    mergemat_path : str
        The file path for a .mat file with fields for a list of binary files, a list of 
    """
    samprate = 30000 # kHz

    merge_info = scipy.io.loadmat(mergemat_path)
    files = merge_info['fileList']
    paths = merge_info['pathList']
    num_samples = merge_info['nSamps']

    # Load phy2 output data
    phy_path, _ = os.path.split(mergemat_path)
    all_spikeT = np.load(os.path.join(phy_path[0],'spike_times.npy'))
    clust = np.load(os.path.join(phy_path[0],'spike_clusters.npy'))
    templates = np.load(os.path.join(phy_path[0],'templates.npy'))

    # Ephys_data_master holds information that is same for all
    # recordings (i.e. cluster information + waveform)
    ephys_data_master = pd.read_csv(os.path.join(phy_path[0],'cluster_info.tsv'),sep = '\t',index_col=0)

    # Insert waveforms
    ephys_data_master['waveform'] = np.nan
    ephys_data_master['waveform'] = ephys_data_master['waveform'].astype(object)
    for _, ind in enumerate(ephys_data_master.index):
        ephys_data_master.at[ind,'waveform'] = templates[ind,21:,ephys_data_master.at[ind,'ch']]

    # Create boundaries between recordings (in terms of timesamples)
    boundaries = np.concatenate((np.array([0]),np.cumsum(num_samples)))

    # Loop over each recording and create/save ephys_data for each one
    for s in range(np.size(num_samples)):

        # Select spikes in this timerange
        use = (all_spikeT >= boundaries[s]) & (all_spikeT<boundaries[s+1])
        theseSpikes = all_spikeT[use]
        theseClust = clust[use[:,0]]

        # Place spikes into ephys data structure
        ephys_data = ephys_data_master.copy()
        ephys_data['spikeT'] = np.NaN
        ephys_data['spikeT'] = ephys_data['spikeT'].astype(object)
        for c in np.unique(clust):
            ephys_data.at[c,'spikeT'] =(theseSpikes[theseClust==c].flatten() - boundaries[s])/samprate
        
        # Get timestamp from .csv for this recording
        fname = files[0,s][0].copy()
        fname = fname[0:-4] + '_BonsaiBoardTS.csv'
        time_path = os.path.join(paths[0,s][0],fname)
        ephysT = fmEphys.utils.time.read_time(time_path)
        ephys_data['t0'] = ephysT[0]
        
        # Write ephys data into json file
        fname = files[0,s][0].copy()
        fname = fname[0:-10] + '_ephys_rec.h5'
        savepath = os.path.join(paths[0,s][0], fname)
        ephys_data.to_hdf(savepath, mode='w')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--matfile', type=str, default=None)
    args = parser.parse_args()

    # If there isn't a path as an argument, open a window so one can be selected
    if args.matfile is None:
        sg.theme('Default1')
        matfile = sg.popup_get_file('Choose merge .mat file')
    else:
        matfile = args.matfile

    split_recordings(matfile)
    