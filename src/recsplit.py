import argparse, os
import PySimpleGUI as sg
import scipy.io
import numpy as np
import pandas as pd

import src.utils as utils

def split_recordings(mergemat_path):
        samprate = 30000 # kHz

        merge_info = scipy.io.loadmat(mergemat_path)
        fileList = merge_info['fileList']
        pathList = merge_info['pathList']
        nSamps = merge_info['nSamps']

        # Load phy2 output data
        phy_path, _ = os.path.split(mergemat_path)
        allSpikeT = np.load(os.path.join(phy_path[0],'spike_times.npy'))
        clust = np.load(os.path.join(phy_path[0],'spike_clusters.npy'))
        templates = np.load(os.path.join(phy_path[0],'templates.npy'))

        # ephys_data_master holds information that is same for all recordings (i.e. cluster information + waveform)
        ephys_data_master = pd.read_csv(os.path.join(phy_path[0],'cluster_info.tsv'),sep = '\t',index_col=0)

        # insert waveforms
        ephys_data_master['waveform'] = np.nan
        ephys_data_master['waveform'] = ephys_data_master['waveform'].astype(object)
        for _, ind in enumerate(ephys_data_master.index):
            ephys_data_master.at[ind,'waveform'] = templates[ind,21:,ephys_data_master.at[ind,'ch']]

        # create boundaries between recordings (in terms of timesamples)
        boundaries = np.concatenate((np.array([0]),np.cumsum(nSamps)))

        # loop over each recording and create/save ephys_data for each one
        for s in range(np.size(nSamps)):

            # select spikes in this timerange
            use = (allSpikeT >= boundaries[s]) & (allSpikeT<boundaries[s+1])
            theseSpikes = allSpikeT[use]
            theseClust = clust[use[:,0]]

            # place spikes into ephys data structure
            ephys_data = ephys_data_master.copy()
            ephys_data['spikeT'] = np.NaN
            ephys_data['spikeT'] = ephys_data['spikeT'].astype(object)
            for c in np.unique(clust):
                ephys_data.at[c,'spikeT'] =(theseSpikes[theseClust==c].flatten() - boundaries[s])/samprate
            
            # get timestamp from csv for this recording
            fname = fileList[0,s][0].copy()
            fname = fname[0:-4] + '_BonsaiBoardTS.csv'
            timePath = os.path.join(pathList[0,s][0],fname)
            ephysT = utils.time.read_time(timePath)
            ephys_data['t0'] = ephysT[0]
            
            # write ephys data into json file
            fname = fileList[0,s][0].copy()
            fname = fname[0:-10] + '_ephys_rec.h5'
            savepath = os.path.join(pathList[0,s][0], fname)
            ephys_data.to_hdf(savepath, mode='w')

if __name__ == '__main__':

    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--matfile', type=str, default=None)
    args = parser.parse_args()

    # if no path was given as an argument, open a dialog box
    if args.matfile is None:
        sg.theme('Default1')
        matfile = sg.popup_get_file('Choose merge .mat file')
    else:
        matfile = args.matfile

    split_recordings(matfile)