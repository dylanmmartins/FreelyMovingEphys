import argparse
import PySimpleGUI as sg

from src.utils.auxiliary import str_to_bool
from src.prelim import RawEphys

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--matfile', type=str, default=None)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = get_args()

    if args.matfile is None:
        # if no path was given as an argument, open a dialog box
        matfile = sg.popup_get_file('Choose .mat file.')
    else:
        matfile = args.matfile

    rephys = RawEphys(matfile)
    rephys.format_spikes()


class RawEphys(BaseInput):
    def __init__(self, merge_file):
        self.merge_file = merge_file
        self.ephys_samprate = 30000

    def format_spikes(self):
        # open 
        merge_info = loadmat(self.merge_file)
        fileList = merge_info['fileList']
        pathList = merge_info['pathList']
        nSamps = merge_info['nSamps']

        # load phy2 output data
        phy_path = os.path.split(self.merge_file)
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
                ephys_data.at[c,'spikeT'] =(theseSpikes[theseClust==c].flatten() - boundaries[s])/self.ephys_samprate
            
            # get timestamp from csv for this recording
            fname = fileList[0,s][0].copy()
            fname = fname[0:-4] + '_BonsaiBoardTS.csv'
            self.timestamp_path = os.path.join(pathList[0,s][0],fname)
            ephysT = self.read_timestamp_file()
            ephys_data['t0'] = ephysT[0]
            
            # write ephys data into json file
            fname = fileList[0,s][0].copy()
            fname = fname[0:-10] + '_ephys_merge.json'
            ephys_json_path = os.path.join(pathList[0,s][0],fname)
            ephys_data.to_json(ephys_json_path)