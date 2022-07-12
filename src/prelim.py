"""
FreelyMovingEphys/src/prelim.py
"""
import os
import numpy as np
from glob import glob
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from matplotlib.backends.backend_pdf import PdfPages

from src.base import BaseInput
from src.worldcam import Worldcam
from src.ephys import Ephys
from src.utils.path import find, auto_recording_name

class PrelimRF(Ephys):
    def __init__(self, binary_path, probe):
        head, _ = os.path.split(binary_path)
        self.recording_path = head
        self.recording_name = auto_recording_name(head)
        self.probe = probe
        self.num_channels = next(int(num) for num in ['128','64','16'] if num in self.probe)
        self.n_cells = self.num_channels
        self.generic_config = {
            'animal_directory': self.recording_path
        }
        self.ephys_samprate = 30000
        self.ephys_offset = 0.1
        self.ephys_drift_rate = -0.1/1000
        self.model_dt = 0.025
        self.spike_thresh = -350
        # this will be applied to the worldcam twice
        # once when avi is packed into a np array, and again when it is put into new bins for spike times
        self.vid_dwnsmpl = 0.25

    def prelim_video_setup(self):
        cam_gamma = 2
        world_norm = (self.world_vid / 255) ** cam_gamma
        std_im = np.std(world_norm, 0)
        std_im[std_im<10/255] = 10 / 255
        self.img_norm = (world_norm - np.mean(world_norm, axis=0)) / std_im
        self.img_norm = self.img_norm * (std_im > 20 / 255)
        self.small_world_vid[self.img_norm < -2] = -2

    def minimal_process(self):
        self.detail_pdf = PdfPages(os.path.join(self.recording_name, 'prelim_raw_whitenoise.pdf'))

        wc = Worldcam(self.generic_config, self.recording_name, self.recording_path, 'WORLD')
        self.worldT = wc.read_timestamp_file()
        self.world_vid = wc.pack_video_frames(usexr=False, dwsmpl=0.25)
        
        self.ephys_bin_path = glob(os.path.join(self.recording_path, '*Ephys.bin'))[0]
        ephys_time_file = glob(os.path.join(self.recording_path, '*Ephys_BonsaiBoardTS.csv'))[0]

        lfp_ephys = self.read_binary_file(do_remap=True)
        ephys_center_sub = lfp_ephys - np.mean(lfp_ephys, 0)
        filt_ephys = self.butter_bandpass(ephys_center_sub, lowcut=800, highcut=8000, fs=30000, order=6)

        ephysT = self.read_timestamp_file()
        t0 = ephysT[0]

        self.worldT = self.worldT - t0

        num_samp = np.size(filt_ephys, 0)
        new_ephysT = np.array(t0 + np.linspace(0, num_samp-1, num_samp) / self.ephys_samprate) - t0
        
        self.model_t = np.arange(0, np.nanmax(self.worldT), self.model_dt)

        self.prelim_video_setup()
        self.worldcam_at_new_timebase(dwnsmpl=0.25)

        all_spikeT = []
        for ch in tqdm(range(np.size(filt_ephys,1))):
            spike_inds = list(np.where(filt_ephys[:,ch] < self.spike_thresh)[0])
            spikeT = new_ephysT[spike_inds]
            all_spikeT.append(spikeT - (self.ephys_offset + spikeT * self.ephys_drift_rate))

        self.model_nsp = np.zeros((self.n_cells, len(self.model_t)))
        bins = np.append(self.model_t, self.model_t[-1]+self.model_dt)
        for i in range(self.n_cells):
            self.model_nsp[i,:], _ = np.histogram(all_spikeT[i], bins)

        self.calc_sta(do_rotation=True, using_spike_sorted=False)

        self.detail_pdf.close()

    # def full_process(self):

# class PrelimDepth(Ephys):
#     def __init__(self, binary_path, probe):

"""
RAW RFs
"""
from modules.prelim_raw_rf.prelim_raw_rf import main as prelim_raw_rf
import PySimpleGUI as sg
import argparse

def make_window(theme):
    sg.theme(theme)
    options_layout =  [[sg.Text('Select the model of ephys probe used.')],
                       [sg.Combo(values=('default16', 'NN_H16', 'default64', 'NN_H64-LP', 'DB_P64-3', 'DB_P64-8', 'DB_P128-6','DB_128-D'), default_value='default16', readonly=True, k='-COMBO-', enable_events=True)],
                       [sg.Text('Select the whitenoise recording directory.')],
                       [sg.Button('Open hf1_wn directory')]]
    logging_layout = [[sg.Text('Run this module')],
                      [sg.Button('Run module')]]
    layout = [[sg.Text('Preliminary whitenoise receptive field mapping', size=(38, 1), justification='center', font=("Times", 16), relief=sg.RELIEF_RIDGE, k='-TEXT HEADING-', enable_events=True)]]
    layout +=[[sg.TabGroup([[sg.Tab('Options', options_layout),
               sg.Tab('Run', logging_layout)]], key='-TAB GROUP-')]]
    return sg.Window('Preliminary whitenoise receptive field mapping', layout)

def main():
    window = make_window(sg.theme())
    while True:
        event, values = window.read(timeout=100)
        if event == 'Open hf1_wn directory':
            wn_dir = sg.popup_get_folder('Choose hf1_wn directory')
            print('Whitenoise directory: ' + str(wn_dir))
        elif event in (None, 'Exit'):
            print('Exiting')
            break
        elif event == 'Run module':
            probe = values['-COMBO-']
            print('Probe: ' + str(probe))
            prelim_raw_rf(wn_dir, probe)
    window.close()
    exit(0)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wn_dir', type=str, default=None)
    parser.add_argument('--probe', type=str, default=None)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    if args.wn_dir is None or args.probe is None:
        main()
    else:
        prelim_raw_rf(args.wn_dir, args.probe)

from glob import glob
import os, cv2, json
from multiprocessing import freeze_support
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from datetime import datetime
import pandas as pd
import xarray as xr
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import interp1d
from scipy.signal import butter, sosfiltfilt

def read_ephys_bin(binary_path, probe_name, do_remap=True, mapping_json=None):
    """
    read in ephys binary file and apply remapping using the name of the probe,
    where the binary dimensions and remaping vector are read in from relative
    path within the FreelyMovingEphys directory (FreelyMovingEphys/matlab/channel_maps.json)
    INPUTS
        binary_path: path to binary file
        probe_name: name of probe, which should be a key in the dict stored in the .json of probe remapping vectors
        do_remap: bool, whether or not to remap the drive
        mapping_json: path to a .json in which each key is a probe name and each value is the 1-indexed sequence of channels
    OUTPUTS
        ephys: ephys DataFrame
    """
    # get channel number
    if '16' in probe_name:
        ch_num = 16
    elif '64' in probe_name:
        ch_num = 64
    elif '128' in probe_name:
        ch_num = 128
    if mapping_json is not None:
        # open file of default mappings
        with open(mapping_json, 'r') as fp:
            mappings = json.load(fp)
        # get the mapping for the probe name used in the current recording
        ch_remap = mappings[probe_name]
    # set up data types to read binary file into
    dtypes = np.dtype([("ch"+str(i),np.uint16) for i in range(0,ch_num)])
    # read in binary file
    ephys = pd.DataFrame(np.fromfile(binary_path, dtypes, -1, ''))
    # remap with known order of channels
    if do_remap is True:
        ephys = ephys.iloc[:,[i-1 for i in list(ch_remap)]]
    return ephys

def butter_bandpass(data, lowcut=1, highcut=300, fs=30000, order=5):
    """
    apply bandpass filter to ephys lfp applied along axis=0
    axis=0 should be the time dimension for any data passed in
    INPUTS
        data: 2d array of multiple channels of ephys data as a numpy array or pandas dataframe
        lowcut: low end of cut off for frequency
        highcut: high end of cut off for frequency
        fs: sample rate
        order: order of filter
    OUTPUTS
        filtered data in the same type as input data
    """
    nyq = 0.5 * fs # Nyquist frequency
    low = lowcut / nyq # low cutoff
    high = highcut / nyq # high cutoff
    sos = butter(order, [low, high], btype='bandpass', output='sos')
    return sosfiltfilt(sos, data, axis=0)

def open_time(path, dlc_len=None, force_shift=False):
    """ Read in the timestamps for a camera and adjust to deinterlaced video length if needed
    Parameters:
    path (str): path to a timestamp .csv file
    dlc_len (int): number of frames in the DLC data (used to decide if interpolation is needed, but this can be left as None to ignore)
    force_shift (bool): whether or not to interpolate timestamps without checking
    
    Returns:
    time_out (np.array): timestamps as numpy
    """
    # read in the timestamps if they've come directly from cameras
    read_time = pd.read_csv(path, encoding='utf-8', engine='c', header=None).squeeze()
    if read_time[0] == 0: # in case header == 0, which is true of some files, drop that header which will have been read in as the first entry  
        read_time = read_time[1:]
    time_in = []
    fmt = '%H:%M:%S.%f'
    if read_time.dtype!=np.float64:
        for current_time in read_time:
            currentT = str(current_time).strip()
            try:
                t = datetime.strptime(currentT,fmt)
            except ValueError as v:
                ulr = len(v.args[0].partition('unconverted data remains: ')[2])
                if ulr:
                    currentT = currentT[:-ulr]
            try:
                time_in.append((datetime.strptime(currentT, '%H:%M:%S.%f') - datetime.strptime('00:00:00.000000', '%H:%M:%S.%f')).total_seconds())
            except ValueError:
                time_in.append(np.nan)
        time_in = np.array(time_in)
    else:
        time_in = read_time.values

    # auto check if vids were deinterlaced
    if dlc_len is not None:
        # test length of the time just read in as it compares to the length of the data, correct for deinterlacing if needed
        timestep = np.nanmedian(np.diff(time_in, axis=0))
        if dlc_len > len(time_in):
            time_out = np.zeros(np.size(time_in, 0)*2)
            # shift each deinterlaced frame by 0.5 frame period forward/backwards relative to timestamp
            time_out[::2] = time_in - 0.25 * timestep
            time_out[1::2] = time_in + 0.25 * timestep
        elif dlc_len == len(time_in):
            time_out = time_in
        elif dlc_len < len(time_in):
            time_out = time_in
    elif dlc_len is None:
        time_out = time_in

    # force the times to be shifted if the user is sure it should be done
    if force_shift is True:
        # test length of the time just read in as it compares to the length of the data, correct for deinterlacing
        timestep = np.nanmedian(np.diff(time_in, axis=0))
        time_out = np.zeros(np.size(time_in, 0)*2)
        # shift each deinterlaced frame by 0.5 frame period forward/backwards relative to timestamp
        time_out[::2] = time_in - 0.25 * timestep
        time_out[1::2] = time_in + 0.25 * timestep

    return time_out

def plot_prelim_STA(spikeT, img_norm, worldT, movInterp, ch_count, lag=2):
    n_units = len(spikeT)
    # model setup
    model_dt = 0.025
    model_t = np.arange(0, np.nanmax(worldT), model_dt)
    model_nsp = np.zeros((n_units, len(model_t)))
    # get binned spike rate
    bins = np.append(model_t, model_t[-1]+model_dt)
    for i in range(n_units):
        model_nsp[i,:], bins = np.histogram(spikeT[i], bins)
    # settting up video
    nks = np.shape(img_norm[0,:,:])
    nk = nks[0]*nks[1]
    model_vid = np.zeros((len(model_t),nk))
    for i in range(len(model_t)):
        model_vid[i,:] = np.reshape(movInterp(model_t[i]+model_dt/2), nk)
    # spike-triggered average
    staAll = np.zeros((n_units, np.shape(img_norm)[1], np.shape(img_norm)[2]))
    model_vid[np.isnan(model_vid)] = 0
    fig = plt.figure(figsize=(20,np.ceil(n_units/2)))
    for c in range(n_units):
        sp = model_nsp[c,:].copy()
        sp = np.roll(sp, -lag)
        sta = model_vid.T @ sp
        sta = np.reshape(sta, nks)
        nsp = np.sum(sp)
        plt.subplot(int(np.ceil(n_units/10)),10,c+1)
        if nsp > 0:
            sta = sta/nsp
            # flip matrix so that physical top is at the top (worldcam comes in upsidedown)
            sta = np.fliplr(np.flipud(sta))
        else:
            sta = np.nan
        if pd.isna(sta) is True:
            plt.imshow(np.zeros([120,160]))
        else:
            starange = np.max(np.abs(sta))*1.1
            plt.imshow((sta-np.mean(sta)), vmin=-starange, vmax=starange, cmap='jet')
            staAll[c,:,:] = sta
    plt.tight_layout()
    return staAll, fig

def main(whitenoise_directory, probe):
    print('finding files')
    world_file = glob(os.path.join(whitenoise_directory, '*WORLD.avi'))[0]
    world_time_file = glob(os.path.join(whitenoise_directory, '*WORLD_BonsaiTS.csv'))[0]
    ephys_file = glob(os.path.join(whitenoise_directory, '*Ephys.bin'))[0]
    ephys_time_file = glob(os.path.join(whitenoise_directory, '*Ephys_BonsaiBoardTS.csv'))[0]
    print('loading and filtering ephys binary')
    pdf = PdfPages(os.path.join(whitenoise_directory, 'prelim_raw_whitenoise.pdf'))
    lfp_ephys = read_ephys_bin(ephys_file, probe, do_remap=False)
    ephys_center_sub = lfp_ephys - np.mean(lfp_ephys,0)
    filt_ephys = butter_bandpass(ephys_center_sub, lowcut=800, highcut=8000, fs=30000, order=6)
    t0 = open_time(ephys_time_file)[0]
    worldT = open_time(world_time_file)
    worldT = worldT - t0
    print('loading worldcam video')
    vidread = cv2.VideoCapture(world_file)
    world_vid = np.empty([int(vidread.get(cv2.CAP_PROP_FRAME_COUNT)),
                        int(vidread.get(cv2.CAP_PROP_FRAME_HEIGHT)*0.25),
                        int(vidread.get(cv2.CAP_PROP_FRAME_WIDTH)*0.25)], dtype=np.uint8)
    # iterate through each frame
    for frame_num in range(0,int(vidread.get(cv2.CAP_PROP_FRAME_COUNT))):
        # read the frame in and make sure it is read in correctly
        ret, frame = vidread.read()
        if not ret:
            break
        # convert to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # downsample the frame by an amount specified in the config file
        sframe = cv2.resize(frame, (0,0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
        # add the downsampled frame to all_frames as int8
        world_vid[frame_num,:,:] = sframe.astype(np.int8)
    print('setting up worldcam and ephys')
    offset0 = 0.1
    drift_rate = -0.1/1000
    num_samp = np.size(filt_ephys,0)
    samp_freq = 30000
    ephys_time = np.array(t0 + np.linspace(0, num_samp-1, num_samp) / samp_freq) - t0
    cam_gamma = 2
    world_norm = (world_vid/255)**cam_gamma
    std_im = np.std(world_norm,axis=0)
    std_im[std_im<10/255] = 10/255
    img_norm = (world_norm-np.mean(world_norm,axis=0))/std_im
    img_norm = img_norm * (std_im>20/255)
    img_norm[img_norm<-2] = -2
    movInterp = interp1d(worldT, img_norm, axis=0, bounds_error=False)
    plt.subplots(np.size(filt_ephys,1),1,figsize=(5,int(np.ceil(np.size(filt_ephys,1)/2))))
    print('getting receptive fields and plotting')
    all_spikeT = []
    for ch in tqdm(range(np.size(filt_ephys,1))):
        spike_thresh = -350
        spike_inds = list(np.where(filt_ephys[:,ch] < spike_thresh)[0])
        spikeT = ephys_time[spike_inds]
        all_spikeT.append(spikeT - (offset0 + spikeT * drift_rate))
    all_STA, fig = plot_prelim_STA(all_spikeT, img_norm, worldT, movInterp, np.size(filt_ephys,1))
    pdf.savefig(); plt.close()
    pdf.close()
    print('done')

"""
Spike sorted prelim RFs
"""
from modules.prelim_sorted_rf.prelim_sorted_rf import main as prelim_sorted_rf
import PySimpleGUI as sg

def make_window(theme):
    sg.theme(theme)
    options_layout =  [[sg.Text('Select the model of ephys probe used.')],
                       [sg.Combo(values=('default16', 'NN_H16', 'default64', 'NN_H64-LP', 'DB_P64-3', 'DB_P64-8', 'DB_P128-6', 'DB_P128-D' ), default_value='default16', readonly=True, k='-COMBO-', enable_events=True)],
                       [sg.Text('Select the whitenoise recording directory.')],
                       [sg.Button('Open hf1_wn directory')]]
    logging_layout = [[sg.Text('Run this module')],
                      [sg.Button('Run module')]]
    layout = [[sg.Text('Preliminary whitenoise receptive field mapping', size=(38, 1), justification='center', font=("Times", 16), relief=sg.RELIEF_RIDGE, k='-TEXT HEADING-', enable_events=True)]]
    layout +=[[sg.TabGroup([[sg.Tab('Options', options_layout),
               sg.Tab('Run', logging_layout)]], key='-TAB GROUP-')]]
    return sg.Window('Preliminary whitenoise receptive field mapping', layout)

def main():
    window = make_window(sg.theme())
    while True:
        event, values = window.read(timeout=100)
        if event == 'Open hf1_wn directory':
            binary_file = sg.popup_get_folder('Choose hf1_wn directory')
            print('Whitenoise directory: ' + str(binary_file))
        elif event in (None, 'Exit'):
            print('Exiting')
            break
        elif event == 'Run module':
            probe = values['-COMBO-']
            print('Probe: ' + str(probe))
            prelim_sorted_rf(binary_file, probe)
    window.close()
    exit(0)

if __name__ == '__main__':
    main()

from glob import glob
import os, cv2, subprocess
from tqdm import tqdm
from multiprocessing import freeze_support
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import interp1d
from datetime import datetime

from src.utils.path import find

def open_time(path, dlc_len=None, force_shift=False):
    """ Read in the timestamps for a camera and adjust to deinterlaced video length if needed
    Parameters:
    path (str): path to a timestamp .csv file
    dlc_len (int): number of frames in the DLC data (used to decide if interpolation is needed, but this can be left as None to ignore)
    force_shift (bool): whether or not to interpolate timestamps without checking
    
    Returns:
    time_out (np.array): timestamps as numpy
    """
    # read in the timestamps if they've come directly from cameras
    read_time = pd.read_csv(path, encoding='utf-8', engine='c', header=None).squeeze()
    if read_time[0] == 0: # in case header == 0, which is true of some files, drop that header which will have been read in as the first entry  
        read_time = read_time[1:]
    time_in = []
    fmt = '%H:%M:%S.%f'
    if read_time.dtype!=np.float64:
        for current_time in read_time:
            currentT = str(current_time).strip()
            try:
                t = datetime.strptime(currentT,fmt)
            except ValueError as v:
                ulr = len(v.args[0].partition('unconverted data remains: ')[2])
                if ulr:
                    currentT = currentT[:-ulr]
            try:
                time_in.append((datetime.strptime(currentT, '%H:%M:%S.%f') - datetime.strptime('00:00:00.000000', '%H:%M:%S.%f')).total_seconds())
            except ValueError:
                time_in.append(np.nan)
        time_in = np.array(time_in)
    else:
        time_in = read_time.values

    # auto check if vids were deinterlaced
    if dlc_len is not None:
        # test length of the time just read in as it compares to the length of the data, correct for deinterlacing if needed
        timestep = np.nanmedian(np.diff(time_in, axis=0))
        if dlc_len > len(time_in):
            time_out = np.zeros(np.size(time_in, 0)*2)
            # shift each deinterlaced frame by 0.5 frame period forward/backwards relative to timestamp
            time_out[::2] = time_in - 0.25 * timestep
            time_out[1::2] = time_in + 0.25 * timestep
        elif dlc_len == len(time_in):
            time_out = time_in
        elif dlc_len < len(time_in):
            time_out = time_in
    elif dlc_len is None:
        time_out = time_in

    # force the times to be shifted if the user is sure it should be done
    if force_shift is True:
        # test length of the time just read in as it compares to the length of the data, correct for deinterlacing
        timestep = np.nanmedian(np.diff(time_in, axis=0))
        time_out = np.zeros(np.size(time_in, 0)*2)
        # shift each deinterlaced frame by 0.5 frame period forward/backwards relative to timestamp
        time_out[::2] = time_in - 0.25 * timestep
        time_out[1::2] = time_in + 0.25 * timestep

    return time_out

def safe_xr_merge(obj_list, dim_name='frame'):
    """ Safely merge list of xarray dataarrays, even when their lengths do not match
    This is only a good idea if expected length differences will be minimal
    Parameters:
    obj_list (list of two or more xr.DataArray): DataArrays to merge as a list (objects should all have a shared dim)
    dim_name (str): name of xr dimension to merge along, default='frame'
    
    Returns:
    merge_objs (xr.DataArray): merged xarray of all objects in input list, even if lengths do not match
    """
    max_lens = []
    # iterate through objects
    for obj in obj_list:
        # get the sizes of the dim, dim_name
        max_lens.append(dict(obj.frame.sizes)[dim_name])
    # get the smallest of the object's length's
    set_len = np.min(max_lens)
    # shorten everything to the shortest length found
    out_objs = []
    for obj in obj_list:
        # get the length of the current object
        obj_len = dict(obj.frame.sizes)[dim_name]
        # if the size of dim is longer
        if obj_len > set_len:
            # how much does it need to be shortened by?
            diff = obj_len - set_len
            # what indeces should be kept?
            good_inds = range(0,obj_len-diff)
            # index to remove what would be jagged ends
            obj = obj.sel(frame=good_inds)
            # add to the list of objects to merge
            out_objs.append(obj)
        # if it is the smallest length or all objects have the same length
        else:
            # just append it to the list of objects to merge
            out_objs.append(obj)
    # do the merge with the lengths all matching along provided dimension
    merge_objs = xr.merge(out_objs)
    return merge_objs

def open_h5(path):
    """ Read in .h5 DLC files and manage column names
    Parameters:
    path (str): filepath to .h5 file outputs by DLC
    Returns:
    pts (pd.DataFrame): values for position
    pt_loc_names (list): column names
    """
    try:
        # read the .hf file when there is no key
        pts = pd.read_hdf(path)
    except ValueError:
        # read in .h5 file when there is a key set in corral_files.py
        pts = pd.read_hdf(path, key='data')
    # organize columns
    pts.columns = [' '.join(col[:][1:3]).strip() for col in pts.columns.values]
    pts = pts.rename(columns={pts.columns[n]: pts.columns[n].replace(' ', '_') for n in range(len(pts.columns))})
    pt_loc_names = pts.columns.values
    return pts, pt_loc_names

def open_ma_h5(path):
    """ Open .h5 file of a multianimal DLC project
    Parameters:
    path (str): filepath to .h5 file outputs by DLC
    Returns:
    pts (Pd.DataFrame): pandas dataframe of points
    """
    pts = pd.read_hdf(path)
    # flatten columns from MultiIndex 
    pts.columns = ['_'.join(col[:][1:]).strip() for col in pts.columns.values]
    return pts

def format_frames(vid_path, config):
    """ Add videos to xarray
    Parameters:
    vid_path (str): path to an avi
    config (dict): options
    Returns:
    formatted_frames (xr.DataArray): of video as b/w int8
    """
    # open the .avi file
    vidread = cv2.VideoCapture(vid_path)
    # empty array that is the target shape
    # should be number of frames x downsampled height x downsampled width
    all_frames = np.empty([int(vidread.get(cv2.CAP_PROP_FRAME_COUNT)),
                        int(vidread.get(cv2.CAP_PROP_FRAME_HEIGHT)*config['parameters']['outputs_and_visualization']['dwnsmpl']),
                        int(vidread.get(cv2.CAP_PROP_FRAME_WIDTH)*config['parameters']['outputs_and_visualization']['dwnsmpl'])], dtype=np.uint8)
    # iterate through each frame
    for frame_num in tqdm(range(0,int(vidread.get(cv2.CAP_PROP_FRAME_COUNT)))):
        # read the frame in and make sure it is read in correctly
        ret, frame = vidread.read()
        if not ret:
            break
        # convert to grayyscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # downsample the frame by an amount specified in the config file
        sframe = cv2.resize(frame, (0,0), fx=config['parameters']['outputs_and_visualization']['dwnsmpl'], fy=config['parameters']['outputs_and_visualization']['dwnsmpl'], interpolation=cv2.INTER_NEAREST)
        # add the downsampled frame to all_frames as int8
        all_frames[frame_num,:,:] = sframe.astype(np.int8)
    # store the combined video frames in an xarray
    formatted_frames = xr.DataArray(all_frames.astype(np.int8), dims=['frame', 'height', 'width'])
    # label frame numbers in the xarray
    formatted_frames.assign_coords({'frame':range(0,len(formatted_frames))})
    # delete all frames, since it's somewhat large in memory
    del all_frames
    return formatted_frames

def h5_to_xr(pt_path, time_path, view, config):
    """ Build an xarray DataArray of the a single camera's dlc point .h5 files and .csv timestamp
    Parameters:
    pt_path (str): filepath to the .h5
    time_path (str): filepath to a .csv
    view (str): camera name (i.e. REYE)
    
    Returns:
    xrpts (xr.DataArray): pose estimate
    """
    # check that pt_path exists
    if pt_path is not None and pt_path != [] and time_path is not None:
        # open multianimal project with a different function than single animal h5 files
        if 'TOP' in view and config['pose_estimation']['multianimal_top_project'] is True:
            # add a step to convert pickle files here?
            pts = open_ma_h5(pt_path)
        # otherwise, use regular h5 file read-in
        else:
            pts, names = open_h5(pt_path)
        # read time file, pass length of points so that it will know if that length matches the length of the timestamps
        # if they don't match because time was not interpolated to match deinterlacing, the timestamps will be interpolated
        time = open_time(time_path, len(pts))
        # label dimensions of the points dataarray
        xrpts = xr.DataArray(pts, dims=['frame', 'point_loc'])
        # label the camera view
        xrpts.name = view
        # assign timestamps as a coordinate to the 
        try:
            xrpts = xrpts.assign_coords(timestamps=('frame', time[1:])) # indexing [1:] into time because first row is the empty header, 0
        # correcting for issue caused by small differences in number of frames
        except ValueError:
            diff = len(time[1:]) - len(xrpts['frame'])
            if diff > 0: # time is longer
                diff = abs(diff)
                new_time = time.copy()
                new_time = new_time[0:-diff]
                xrpts = xrpts.assign_coords(timestamps=('frame', new_time[1:]))
            elif diff < 0: # frame is longer
                diff = abs(diff)
                timestep = time[1] - time[0]
                new_time = time.copy()
                for i in range(1,diff+1):
                    last_value = new_time[-1] + timestep
                    new_time = np.append(new_time, pd.Series(last_value))
                xrpts = xrpts.assign_coords(timestamps=('frame', new_time[1:]))
            else: # equal (probably won't happen because ValueError should have been caused by unequal lengths)
                xrpts = xrpts.assign_coords(timestamps=('frame', time[1:]))
    # pt_path will have no data in it for world cam data, so it will make an xarray with just timestamps
    elif pt_path is None or pt_path == [] and time_path is not None:
        if time_path is not None and time_path != []:
            # read in the time
            time = open_time(time_path)
            # setup frame indices
            xrpts = xr.DataArray(np.zeros([len(time)-1]), dims=['frame'])
            # assign frame coordinates, then timestamps
            xrpts = xrpts.assign_coords({'frame':range(0,len(xrpts))})
            xrpts = xrpts.assign_coords(timestamps=('frame', time[1:]))
            names = None
        elif time_path is None or time_path == []:
            xrpts = None; names = None
    # if timestamps are missing, still read in and format as xarray
    elif pt_path is not None and pt_path != [] and time_path is None:
        # open multianimal project with a different function than single animal h5 files
        if 'TOP' in view and config['pose_estimation']['multianimal_top_project'] is True:
            # add a step to convert pickle files here?
            pts = open_ma_h5(pt_path)
        # otherwise, use regular h5 file read-in
        else:
            pts, names = open_h5(pt_path)
        # label dimensions of the points dataarray
        xrpts = xr.DataArray(pts, dims=['frame', 'point_loc'])
        # label the camera view
        xrpts.name = view
    return xrpts

def deinterlace_data(config, vid_list=None, time_list=None):
    """ Deinterlace videos, shift times to match the new video frame count.
    Searches subdirectories if vid_list and time_list are both None.
    If lists of files are provided, it will not search subdirectories and instead analyze items in those lists.
    Parameters:
    config (dict): options dict
    vid_list (list): .avi file paths for videos to deinterlace (optional)
    time_list (list): .csv file paths of timestamps matching videos to deinterlace (optional)
    """
    # get paths out of the config dictionary
    data_path = config['animal_dir']
    # find all the files assuming no specific files are listed
    if vid_list is None:
        avi_list = find('*.avi', data_path)
        csv_list = find('*.csv', data_path)
    # if a specific list of videos is provided, ignore the config file's data path
    elif vid_list is not None:
        avi_list = vid_list.copy()
        csv_list = time_list.copy()
    # iterate through each video
    for this_avi in avi_list:
        current_path = os.path.split(this_avi)[0]
        # make a save path that keeps the subdirectories
        # get out an key from the name of the video that will be shared with all other data of this trial
        vid_name = os.path.split(this_avi)[1]
        key_pieces = vid_name.split('.')[:-1]
        key = '.'.join(key_pieces)
        # then, find those other pieces of the trial using the key
        try:
            this_csv = [i for i in csv_list if key in i][0]
            csv_present = True
        except IndexError:
            csv_present = False
        # open the video
        cap = cv2.VideoCapture(this_avi)
        # get some info about the video
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) # number of total frames
        fps = cap.get(cv2.CAP_PROP_FPS) # frame rate
        # make sure the save directory exists
        if not os.path.exists(current_path):
            os.makedirs(current_path)
        # files that will need to be deinterlaced will be read in with a frame rate of 30 frames/sec
        elif fps == 30:
            print('starting to deinterlace and interpolate on ' + key)
            # create save path
            avi_out_path = os.path.join(current_path, (key + 'deinter.avi'))
            # flip the eye video horizonally and vertically and deinterlace, if this is specified in the config
            if config['deinterlace']['flip_eye_during_deinter'] is True and ('EYE' in this_avi or 'WORLD' in this_avi):
                subprocess.call(['ffmpeg', '-i', this_avi, '-vf', 'yadif=1:-1:0, vflip, hflip, scale=640:480', '-c:v', 'libx264', '-preset', 'slow', '-crf', '19', '-c:a', 'aac', '-b:a', '256k', '-y', avi_out_path])
            # or, deinterlace without flipping
            elif config['deinterlace']['flip_eye_during_deinter'] is False and ('EYE' in this_avi or 'WORLD' in this_avi):
                subprocess.call(['ffmpeg', '-i', this_avi, '-vf', 'yadif=1:-1:0, scale=640:480', '-c:v', 'libx264', '-preset', 'slow', '-crf', '19', '-c:a', 'aac', '-b:a', '256k', '-y', avi_out_path])
            # correct the frame count of the video
            # now that it's deinterlaced, the video has 2x the number of frames as before
            # this will be used to correct the timestamps associated with this video
            frame_count_deinter = frame_count * 2
            if csv_present is True:
                # get the save path for new timestamps
                csv_out_path = os.path.join(current_path, (key + '_BonsaiTSformatted.csv'))
                # read in the exiting timestamps, interpolate to match the new number of steps, and format as dataframe
                csv_out = pd.DataFrame(open_time(this_csv, int(frame_count_deinter)))
                # save new timestamps
                csv_out.to_csv(csv_out_path, index=False)
        else:
            print('frame rate not 30 or 60 for ' + key)

def undistort_vid(vidpath, savepath, mtx, dist, rvecs, tvecs):
    """
    undistort novel videos using provided camera calibration properties
    INPUTS
        vidpath: path to the video file
        savepath: file path (not a directory) into which the undistorted video will be saved
        mtx: camera matrix
        dist: distortion coefficients
        rvecs: rotation vectors
        tvecs: translation vectors
    OUTPUTS
        None
    if vidpath and savepath are the same filename, the file will be overwritten
    saves a new copy of the video, after it has been undistorted
    """
    # open the video
    cap = cv2.VideoCapture(vidpath)
    # setup the file writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_vid = cv2.VideoWriter(savepath, fourcc, 60.0, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    # iterate through all frames
    print('undistorting video')
    for step in tqdm(range(0,int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
        # open frame and check that it opens correctly
        ret, frame = cap.read()
        if not ret:
            break
        # run opencv undistortion function
        undist_frame = cv2.undistort(frame, mtx, dist, None, mtx)
        # write the frame to the video
        out_vid.write(undist_frame)
    out_vid.release()

def calibrate_new_world_vids(config):
    """
    calibrate novel world videos using previously genreated .npy of parameters
    INPUTS
        config: options dictionary
    OUTPUTS
        None
    """
    # load the parameters
    checker_in = np.load(config['calibration']['world_checker_npz'])
    # unpack camera properties
    mtx = checker_in['mtx']; dist = checker_in['dist']; rvecs = checker_in['rvecs']; tvecs = checker_in['tvecs']
    # iterate through eye videos and save out a copy which has had distortions removed
    world_list = find('*WORLDdeinter*.avi', config['animal_dir'])
    for world_vid in world_list:
        if 'plot' not in world_vid:
            savepath = '_'.join(world_vid.split('_')[:-1])+'_WORLDcalib.avi'
            undistort_vid(world_vid, savepath, mtx, dist, rvecs, tvecs)

def plot_spike_raster(goodcells):
    """
    plot spike raster so that superficial channels are at the top of the panel
    INPUTS
        goodcells: ephys dataframe
    OUTPUTS
        fig: figure
    """
    fig, ax = plt.subplots()
    ax.fontsize = 20
    n_units = len(goodcells)
    # iterate through units
    for i, ind in enumerate(goodcells.index):
        # array of spike times
        sp = np.array(goodcells.at[ind,'spikeT'])
        # make vertical line for each time the unit fires
        plt.vlines(sp[sp<10],i-0.25,i+0.25)
        # plot only ten seconds
        plt.xlim(0, 10)
        # turn off ticks
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    plt.xlabel('secs',fontsize = 20)
    plt.ylabel('unit number',fontsize=20)
    plt.ylim([n_units,0])
    return fig

def plot_spike_rate_vs_var(use, var_range, goodcells, useT, t, var_label):
    """
    plot spike rate vs a given variable (e.g. pupil radius, worldcam contrast, etc.)
    INPUTS
        use: varaible to plot (can be filtered e.g. only active times)
        var_range: range of bins to calculate response over
        goodcells: ephys dataframe
        useT: timestamps that match the vairable (use)
        t: timebase
        var_label: label for last panels xlabel
    OUTPUTS
        var_cent: x axis bins
        tuning: tuning curve across bins
        tuning_err: stderror of variable at each bin
        fig: figure
    """
    n_units = len(goodcells)
    scatter = np.zeros((n_units,len(use)))
    tuning = np.zeros((n_units,len(var_range)-1))
    tuning_err = tuning.copy()
    var_cent = np.zeros(len(var_range)-1)
    for j in range(len(var_range)-1):
        var_cent[j] = 0.5*(var_range[j] + var_range[j+1])
    for i, ind in enumerate(goodcells.index):
        rateInterp = interp1d(t[0:-1], goodcells.at[ind,'rate'], bounds_error=False)
        scatter[i,:] = rateInterp(useT)
        for j in range(len(var_range)-1):
            usePts = (use>=var_range[j]) & (use<var_range[j+1])
            tuning[i,j] = np.nanmean(scatter[i, usePts])
            tuning_err[i,j] = np.nanstd(scatter[i, usePts]) / np.sqrt(np.count_nonzero(usePts))
    fig = plt.subplots(int(np.ceil(n_units/7)),7,figsize=(35,np.int(np.ceil(n_units/3))),dpi=50)
    for i, ind in enumerate(goodcells.index):
        plt.subplot(int(np.ceil(n_units/7)),7,i+1)
        plt.errorbar(var_cent,tuning[i,:],yerr=tuning_err[i,:])
        try:
            plt.ylim(0,np.nanmax(tuning[i,:]*1.2))
        except ValueError:
            plt.ylim(0,1)
        plt.xlim([var_range[0], var_range[-1]]); plt.title(ind,fontsize=5)
        plt.xlabel(var_label,fontsize=5); plt.ylabel('sp/sec',fontsize=5)
        plt.xticks(fontsize=5); plt.yticks(fontsize=5)
    plt.tight_layout()
    return var_cent, tuning, tuning_err, fig

def plot_STA(goodcells, img_norm, worldT, movInterp, ch_count, lag=2, show_title=True):
    """
    plot spike-triggered average for either a single lag or a range of lags
    INPUTS
        goodcells: dataframe of ephys data
        img_norm: normalized worldcam video
        worldT: worldcam timestamps
        movInterp: interpolator for worldcam movie
        ch_count: number of probe channels
        lag: time lag, should be np.arange(-2,8,2) for range of lags, or 2 for single lag
        show_title: bool, whether or not to show title above each panel
    OUTPUTS
        staAll: STA receptive field of each unit
        fig: figure
    """
    n_units = len(goodcells)
    # model setup
    model_dt = 0.025
    model_t = np.arange(0, np.max(worldT), model_dt)
    model_nsp = np.zeros((n_units, len(model_t)))
    # get binned spike rate
    bins = np.append(model_t, model_t[-1]+model_dt)
    for i, ind in enumerate(goodcells.index):
        model_nsp[i,:], bins = np.histogram(goodcells.at[ind,'spikeT'], bins)
    # settting up video
    nks = np.shape(img_norm[0,:,:])
    nk = nks[0]*nks[1]
    model_vid = np.zeros((len(model_t),nk))
    for i in range(len(model_t)):
        model_vid[i,:] = np.reshape(movInterp(model_t[i]+model_dt/2), nk)
    # spike-triggered average
    staAll = np.zeros((n_units, np.shape(img_norm)[1], np.shape(img_norm)[2]))
    model_vid[np.isnan(model_vid)] = 0
    if type(lag) == int:
        fig = plt.subplots(int(np.ceil(n_units/10)),10,figsize=(20,np.int(np.ceil(n_units/3))),dpi=50)
        for c, ind in enumerate(goodcells.index):
            sp = model_nsp[c,:].copy()
            sp = np.roll(sp, -lag)
            sta = model_vid.T @ sp
            sta = np.reshape(sta, nks)
            nsp = np.sum(sp)
            plt.subplot(int(np.ceil(n_units/10)),10,c+1)
            ch = int(goodcells.at[ind,'ch'])
            if ch_count == 64 or ch_count == 128:
                shank = np.floor(ch/32); site = np.mod(ch,32)
            else:
                shank = 0; site = ch
            if show_title:
                plt.title(f'ind={ind!s} nsp={nsp!s}\n ch={ch!s} shank={shank!s}\n site={site!s}',fontsize=5)
            plt.axis('off')
            if nsp > 0:
                sta = sta/nsp
            else:
                sta = np.nan
            if pd.isna(sta) is True:
                plt.imshow(np.zeros([120,160]))
            else:
                plt.imshow((sta-np.mean(sta) ),vmin=-0.3,vmax=0.3,cmap = 'jet')
                staAll[c,:,:] = sta
        plt.tight_layout()
        return staAll, fig
    else:
        lagRange = lag
        fig = plt.subplots(n_units,5,figsize=(6, np.int(np.ceil(n_units/2))),dpi=300)
        for c, ind in enumerate(goodcells.index):
            for lagInd, lag in enumerate(lagRange):
                sp = model_nsp[c,:].copy()
                sp = np.roll(sp,-lag)
                sta = model_vid.T@sp
                sta = np.reshape(sta,nks)
                nsp = np.sum(sp)
                plt.subplot(n_units,5,(c*5)+lagInd + 1)
                if nsp > 0:
                    sta = sta/nsp
                else:
                    sta = np.nan
                if pd.isna(sta) is True:
                    plt.imshow(np.zeros([120,160]))
                else:
                    plt.imshow((sta-np.mean(sta)),vmin=-0.3,vmax=0.3,cmap = 'jet')
                if c == 0:
                    plt.title(str(np.round(lag*model_dt*1000)) + 'msec',fontsize=5)
                plt.axis('off')
            plt.tight_layout()
        return fig

def plot_STV(goodcells, movInterp, img_norm, worldT):
    """
    plot spike-triggererd varaince
    INPUTS
        goodcells: ephys dataframe
        movInterp: interpolator for worldcam movie
        img_norm: normalized worldcam video
        worldT: world timestamps
    OUTPUTS
        stvAll: spike triggered variance for all units
        fig: figure
    """
    n_units = len(goodcells)
    # model setup
    model_dt = 0.025
    model_t = np.arange(0, np.max(worldT), model_dt)
    model_nsp = np.zeros((n_units, len(model_t)))
    # get binned spike rate
    bins = np.append(model_t, model_t[-1]+model_dt)
    for i, ind in enumerate(goodcells.index):
        model_nsp[i,:], bins = np.histogram(goodcells.at[ind,'spikeT'], bins)
    # settting up video
    nks = np.shape(img_norm[0,:,:])
    nk = nks[0]*nks[1]
    model_vid = np.zeros((len(model_t),nk))
    for i in range(len(model_t)):
        model_vid[i,:] = np.reshape(movInterp(model_t[i]+model_dt/2), nk)
    model_vid = model_vid**2
    lag = 2
    stvAll = np.zeros((n_units, np.shape(img_norm)[1], np.shape(img_norm)[2]))
    fig = plt.subplots(int(np.ceil(n_units/10)),10,figsize=(20,np.int(np.ceil(n_units/3))),dpi=50)
    for c, ind in enumerate(goodcells.index):
        sp = model_nsp[c,:].copy()
        sp = np.roll(sp, -lag)
        sta = np.nan_to_num(model_vid,0).T @ sp
        sta = np.reshape(sta, nks)
        nsp = np.sum(sp)
        plt.subplot(int(np.ceil(n_units/10)), 10, c+1)
        if nsp > 0:
            sta = sta / nsp
        else:
            sta = np.nan
        if pd.isna(sta) is True:
            plt.imshow(np.zeros([120,160]))
        else:
            plt.imshow(sta - np.mean(img_norm**2,axis=0), vmin=-1, vmax=1)
        stvAll[c,:,:] = sta - np.mean(img_norm**2, axis=0)
        plt.axis('off')
    plt.tight_layout()
    return stvAll, fig

def main(whitenoise_directory, probe):
    temp_config = {
        'animal_dir': whitenoise_directory,
        'deinterlace':{
            'flip_eye_during_deinter': True,
            'flip_world_during_deinter': True
        },
        'calibration': {
            'world_checker_npz': 'E:/freely_moving_ephys/camera_calibration_params/world_checkerboard_calib.npz'
        },
        'parameters':{
            'follow_strict_naming': True,
            'outputs_and_visualization':{
                'save_nc_vids': True,
                'dwnsmpl': 0.25
            },
            'ephys':{
                'ephys_sample_rate': 30000
            }
        }
    }
    # find world files
    world_vids = glob(os.path.join(whitenoise_directory, '*WORLD.avi'))
    world_times = glob(os.path.join(whitenoise_directory, '*WORLD_BonsaiTS.csv'))
    # deinterlace world video
    deinterlace_data(temp_config, world_vids, world_times)
    # apply calibration parameters to world video
    calibrate_new_world_vids(temp_config)
    # organize nomenclature
    trial_units = []; name_check = []; path_check = []
    for avi in find('*.avi', temp_config['animal_dir']):
        bad_list = ['plot','IR','rep11','betafpv','side_gaze'] # don't use trials that have these strings in their path
        if temp_config['parameters']['follow_strict_naming'] is True:
            if all(bad not in avi for bad in bad_list):
                split_name = avi.split('_')[:-1]
                trial = '_'.join(split_name)
                path_to_trial = os.path.join(os.path.split(trial)[0])
                trial_name = os.path.split(trial)[1]
        elif temp_config['parameters']['follow_strict_naming'] is False:
            if all(bad not in avi for bad in bad_list):
                trial_path_noext = os.path.splitext(avi)[0]
                path_to_trial, trial_name_long = os.path.split(trial_path_noext)
                trial_name = '_'.join(trial_name_long.split('_')[:3])
        try:
            if trial_name not in name_check:
                trial_units.append([path_to_trial, trial_name])
                path_check.append(path_to_trial); name_check.append(trial_name)
        except UnboundLocalError:
            pass
    # there should only be one item in trial_units in this case
    # iterate into that
    for trial_unit in trial_units:
        temp_config['trial_path'] = trial_unit[0]
        t_name = trial_unit[1]
        # find the timestamps and video for all camera inputs
        trial_cam_csv = find(('*BonsaiTS*.csv'), temp_config['trial_path'])
        trial_cam_avi = find(('*.avi'), temp_config['trial_path'])
        trial_cam_csv = [x for x in trial_cam_csv if x != []]
        trial_cam_avi = [x for x in trial_cam_avi if x != []]
        # filter the list of files for the current trial to get the world view of this side
        world_csv = [i for i in trial_cam_csv if 'WORLD' in i and 'formatted' in i][0]
        world_avi = [i for i in trial_cam_avi if 'WORLD' in i and 'calib' in i][0]
        # make an xarray of timestamps without dlc points, since there aren't any for world camera
        worlddlc = h5_to_xr(pt_path=None, time_path=world_csv, view=('WORLD'), config=temp_config)
        worlddlc.name = 'WORLD_times'
        # make xarray of video frames
        if temp_config['parameters']['outputs_and_visualization']['save_nc_vids'] is True:
            xr_world_frames = format_frames(world_avi, temp_config); xr_world_frames.name = 'WORLD_video'
        # merge but make sure they're not off in lenght by one value, which happens occasionally
        print('saving nc file of world view...')
        if temp_config['parameters']['outputs_and_visualization']['save_nc_vids'] is True:
            trial_world_data = safe_xr_merge([worlddlc, xr_world_frames])
            trial_world_data.to_netcdf(os.path.join(temp_config['trial_path'], str(t_name+'_world.nc')), engine='netcdf4', encoding={'WORLD_video':{"zlib": True, "complevel": 4}})
        elif temp_config['parameters']['outputs_and_visualization']['save_nc_vids'] is False:
            worlddlc.to_netcdf(os.path.join(temp_config['trial_path'], str(t_name+'_world.nc')))
        # now start minimal ephys analysis
        print('generating ephys plots')
        pdf = PdfPages(os.path.join(whitenoise_directory, (t_name + '_prelim_wn_figures.pdf')))
        ephys_file_path = glob(os.path.join(whitenoise_directory, '*_ephys_merge.json'))[0]
        world_file_path = glob(os.path.join(whitenoise_directory, '*_world.nc'))[0]
        world_data = xr.open_dataset(world_file_path)
        world_vid_raw = np.uint8(world_data['WORLD_video'])
        # ephys data
        if '16' in probe:
            ch_count = 16
        elif '64' in probe:
            ch_count = 64
        elif '128' in probe:
            ch_count = 128
        ephys_data = pd.read_json(ephys_file_path)
        ephysT0 = ephys_data.iloc[0,12]
        worldT = world_data.timestamps - ephysT0
        ephys_data['spikeTraw'] = ephys_data['spikeT'].copy()
        # sort ephys units by channel
        ephys_data = ephys_data.sort_values(by='ch', axis=0, ascending=True)
        ephys_data = ephys_data.reset_index()
        ephys_data = ephys_data.drop('index', axis=1)
        # correct offset between ephys and other data inputs
        offset0 = 0.1
        drift_rate = -0.1/1000
        for i in ephys_data.index:
            ephys_data.at[i,'spikeT'] = np.array(ephys_data.at[i,'spikeTraw']) - (offset0 + np.array(ephys_data.at[i,'spikeTraw']) *drift_rate)
        # get cells labeled as good
        goodcells = ephys_data.loc[ephys_data['group']=='good']
        # occasional problem with worldcam timestamps
        if worldT[0]<-600:
            worldT = worldT + 8*60*60
        # resize worldcam to make more manageable
        world_vid = world_vid_raw.copy()
        # img correction applied to worldcam
        cam_gamma = 2
        world_norm = (world_vid/255)**cam_gamma
        std_im = np.std(world_norm,axis=0)
        std_im[std_im<10/255] = 10/255
        img_norm = (world_norm-np.mean(world_norm,axis=0))/std_im
        img_norm = img_norm * (std_im>20/255)
        contrast = np.empty(worldT.size)
        for i in range(worldT.size):
            contrast[i] = np.std(img_norm[i,:,:])
        newc = interp1d(worldT,contrast,fill_value="extrapolate")
        # bin ephys spike times as spike rate / s
        dt = 0.025
        t = np.arange(0, np.max(worldT),dt)
        ephys_data['rate'] = np.nan
        ephys_data['rate'] = ephys_data['rate'].astype(object)
        for i,ind in enumerate(ephys_data.index):
            ephys_data.at[ind,'rate'], bins = np.histogram(ephys_data.at[ind,'spikeT'],t)
        ephys_data['rate']= ephys_data['rate']/dt
        goodcells = ephys_data.loc[ephys_data['group']=='good']
        n_units = len(goodcells)
        contrast_interp = newc(t[0:-1])
        # worldcam interp and set floor to values
        img_norm[img_norm<-2] = -2
        movInterp = interp1d(worldT,img_norm,axis=0, bounds_error=False) # added extrapolate for cases where x_new is below interpolation range
        # raster
        raster_fig = plot_spike_raster(goodcells)
        pdf.savefig()
        plt.close()
        print('making diagnostic plots')
        # plot contrast over entire video
        plt.figure()
        plt.plot(worldT[0:12000],contrast[0:12000])
        plt.xlabel('time')
        plt.ylabel('contrast')
        pdf.savefig()
        plt.close()
        # plot contrast over ~2min
        plt.figure()
        plt.plot(t[0:600],contrast_interp[0:600])
        plt.xlabel('secs'); plt.ylabel('contrast')
        pdf.savefig()
        plt.close()
        # worldcam timing diff
        plt.figure()
        plt.plot(np.diff(worldT)); plt.xlabel('frame'); plt.ylabel('deltaT'); plt.title('world cam')
        pdf.savefig()
        plt.close()
        print('getting contrast response function')
        crange = np.arange(0,1.2,0.1)
        crf_cent, crf_tuning, crf_err, crf_fig = plot_spike_rate_vs_var(contrast, crange, goodcells, worldT, t, 'contrast')
        pdf.savefig()
        plt.close()
        print('getting spike-triggered average')
        _, STA_singlelag_fig = plot_STA(goodcells, img_norm, worldT, movInterp, ch_count, lag=2, show_title=True)
        pdf.savefig()
        plt.close()
        print('getting spike-triggered average with range in lags')
        _, STA_multilag_fig = plot_STA(goodcells, img_norm, worldT, movInterp, ch_count, lag=np.arange(-2,8,2), show_title=False)
        pdf.savefig()
        plt.close()
        # print('getting spike-triggered variance')
        # _, STV_fig = plot_STV(goodcells, movInterp, img_norm, worldT)
        # pdf.savefig()
        plt.close()
        print('closing pdf')
        pdf.close()
        print('done')