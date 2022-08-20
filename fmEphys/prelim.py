import argparse, os, json
import PySimpleGUI as sg
import numpy as np
import scipy.interpolate

import fmEphys.utils as utils

def prelimRF_raw(wn_dir, probe):

    # Get files
    world_avi = utils.path.find('*WORLD.avi', wn_dir)
    world_csv = utils.path.find('*WORLD_BonsaiTS.csv', wn_dir)
    ephys_bin = utils.path.find('*Ephys.bin', wn_dir)
    ephys_csv = utils.path.find('*Ephys_BonsaiBoardTS.csv', wn_dir)

    # Worldcam setup
    worldT = utils.time.read_time(world_csv)
    worldT = worldT - ephysT0

    stim_arr = utils.video.avi_to_arr(world_avi, ds=0.25)

    # Ephys setup
    ephysT = utils.time.read_time(ephys_csv)
    ephysT0 = ephysT[0]

    n_ch, _ = utils.base.probe_to_nCh(probe)
    ephys = utils.ephys.read_ephysbin(ephys_bin, n_ch=n_ch)
    spikeT = utils.ephys.volt_to_spikes(ephys, ephysT0, fixT=True) # values are corrected for drift/offset, too

    # get stimulus
    cam_gamma = 2
    norm_stim = (stim_arr / 255)**cam_gamma

    std_im = np.std(norm_stim, axis=0)
    std_im[std_im < 10/255] = 10/255

    img_norm = (norm_stim - np.mean(norm_stim, axis=0)) / std_im
    img_norm = img_norm * (std_im > 20/255)
    img_norm[img_norm < -2] = -2

    movInterp = scipy.interpolate.interp1d(worldT, img_norm, axis=0, bounds_error=False)

    # calculate STAs















def prelimRF_sorted():

def window(probe_opts):
    sg.theme('Default1')
    opt_layout =  [[sg.Text('Probe layout')],
                   [sg.Combo(values=(probe_opts), default_value=probe_opts[0],
                      readonly=True, k='k_probe', enable_events=True)],

                   [sg.Text('White noise directory')],
                   [sg.Button('Open directory', k='k_dir')],

                   [sg.Radio('Raw', group_id='code_type', k='k_raw', default=True)],
                   [sg.Radio('Spike-sorted', group_id='code_type', k='k_sorted')],
                
                   [sg.Button('Start', k='k_start')]]

    return sg.Window('FreelyMovingEphys: Preliminary Modules', opt_layout)


def make_window(chmaps_path):

    with open(chmaps_path, 'r') as fp:
            mappings = json.load(fp)
    probe_opts = mappings.keys()

    sg.theme('Default1')
    ready = False
    w = window(probe_opts)
    while True:
        event, values = w.read(timeout=100)

        if event == 'k_dir':
            wn_dir = sg.popup_get_folder('Open directory')
            print('Set {} as white noise directory'.format(wn_dir))

        elif event in (None, 'Exit'):
            break

        elif event == 'k_start':
            use_probe = values['k_probe']

            if values['k_raw'] is True:
                spike_sorted = False
            elif values['k_sorted'] is True:
                spike_sorted = True

            ready = True
            break

    w.close()
    if ready:
        return wn_dir, use_probe, spike_sorted
    else:
        return None, None, None

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', type=str, default=None)
    parser.add_argument('-p', '--probe', type=str, default=None)
    parser.add_argument('-s', '--sorted', type=utils.base.str_to_bool, nargs='?', const=True, default=False)
    args = parser.parse_args()

    if args.dir is None or args.probe is None:
        src_dir, _ = os.path.split(__file__)
        repo_dir, _ = os.path.split(src_dir)
        chmaps_path = os.path.join(repo_dir, 'config/channel_maps.json')
        
        wn_dir, probe, ssorted = make_window(chmaps_path)
    else:
        wn_dir = args.dir; probe = args.probe; ssorted = args.sorted

    if all(x is not None for x in [wn_dir, probe, ssorted]):
        print('White noise path: {}\n Probe map: {}\n Spike sorted: {}'.format(
            wn_dir, probe, ssorted))

        if ssorted is False:
            prelimRF_raw(wn_dir, probe)

        elif ssorted is True:
            prelimRF_sorted(wn_dir, probe)





"""
RAW RFs
"""



"""
Spike sorted prelim RFs
"""


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

from fmEphys.utils.path import find

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