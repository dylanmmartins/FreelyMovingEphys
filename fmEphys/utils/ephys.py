"""Utilities for analyzing ephys and stimulus data.
"""
import os
import json

import cv2
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.linalg
import scipy.sparse
import scipy.ndimage
import sklearn.linear_model
from sklearn.neighbors import KernelDensity
from tqdm import tqdm

import fmEphys.utils as fmUtil

def read_ephysbin(path, n_ch, probe_name=None, chmap_path=None):
    """ Read in ephys binary file.

    If `probe_name` and `chmap_path` are both left as None, the binary
    file will be read in and remain in the current non-physical order.
    If either is given, the ephys data will be remapped.

    Parameters
    ----------
    path : str
        File path to a binary file of ephys data. Data will be read
        in as type uint16.
    n_ch : int
        Number of channels in the probe's data.
    probe_name : str
        The channels in the binary file are shuffled relative to physical
        position. If `probe_name` is given, and it exists as a key in the
        file /fmEphys/internals/probe_maps.json, then the remapped sequence
        of channels will be read in and data in the ephys binary file will
        be rearrenged in this order.
    chmap_path : str
        If there is another .json file with remapping orders, this will be
        used instead of the repository's default .json.
        
    Returns:
    ephys_arr : np.array
        Ephys data with shape (time, channel)
    """
    # Set up data types
    dtypes = np.dtype([('ch'+str(i), np.uint16) for i in range(0, n_ch)])
    
    # Read in binary file
    ephys_arr = pd.DataFrame(np.fromfile(path, dtypes, -1, ''))

    # Probe name is provided, channel map json was not.
    if (probe_name is not None) and (chmap_path is None):
        # Open channel map file
        utils_dir, _ = os.path.split(__file__)
        src_dir, _ = os.path.split(utils_dir)
        repo_dir, _ = os.path.split(src_dir)
        chmap_path = os.path.join(repo_dir, 'config/channel_maps.json')
    
    # Read in the probe map, whether it was provided as a path or if
    # we will use the default json.
    if chmap_path is not None:
        # Read in the channel order
        with open(chmap_path, 'r') as fp:
            all_maps = json.load(fp)

        # Get channel map for the current probe
        ch_map = all_maps[probe_name]

        # Remap with known order of channels
        ephys_arr = ephys_arr.iloc[:,[i-1 for i in list(ch_map)]]
    
    ephys_arr = ephys_arr.to_numpy()

    return ephys_arr

def calc_approx_sp(ephys, t0, spike_thresh=-350, fixT=True):
    """Calculate spike times from ephys binary.

    This isn't a replacement for spike sorting, but can be a useful
    method to get approximate spike times, either to debug or when doing
    preliminary analysis before spike sorting has been run.

    Parameters
    ----------
    ephys : np.array
        Array of spike data with shape (time, channels) i.e. having been
        read in using `read_ephysbin()` function.
    t0 : float
        Timestamp (in seconds) for the start of aquisition.
    spike_thresh : int
        Threshold for deflection used to get the index of spikes. Should be a
        negative value.
    fixT : bool
        If True, the spike times for each channel will be corrected
        for offset and drift in the aquisition rate relative to camera
        and IMU data.

    Returns
    -------
    spikeT_arr : np.array
        Array of approximate spike times (in seconds).

    """

    ephys_offset_val = 0.1
    ephys_drift_rate = -0.000114
    samp_freq = 30000

    # Center values on the mean
    ephys = ephys - np.mean(ephys, 0)

    # Highpass filter
    filt_ephys = fmUtil.filter.butter_filt(ephys, lowcut=800, highcut=8000, order=5)

    # Samples start at t0, and are acquired at rate of n_samples / freq
    num_samp = np.size(filt_ephys,0)
    ephysT = np.array(t0 + np.linspace(0, num_samp-1, num_samp) / samp_freq)

    n_ch = np.size(filt_ephys,1)
    all_spikeT = []
    for ch in tqdm(range(n_ch)):
        # Get spike times
        spike_inds = list(np.where(filt_ephys[:,ch] < spike_thresh)[0])
        spikeT = ephysT[spike_inds]
        if fixT:
            # Correct the spike times
            spikeT = spikeT - (ephys_offset_val + spikeT * ephys_drift_rate)
        all_spikeT.append(spikeT)
    spikeT_arr = np.array(all_spikeT)

    return spikeT_arr

def calc_PSTH(spikeT, eventT, bandwidth=10, resample_size=1, edgedrop=15, win=1000):
    """Calculate PSTH for a single unit.

    The Peri-Stimulus Time Histogram (PSTH) will be calculated using Kernel
    Density Estimation by sliding a gaussian along the spike times centered
    on the event time.

    Because the gaussian filter will create artifacts at the edges (i.e. the
    start and end of the time window), it's best to add extra time to the start
    and end and then drop that time from the PSTH, leaving the final PSTH with no
    artifacts at the start and end. The time (in msec) set with `edgedrop` pads
    the start and end with some time which is dropped from the final PSTH before
    the PSTH is returned.

    Parameters
    ----------
    spikeT : np.array
        Array of spike times in seconds and with the type float. Should be 1D and be
        the spike times for a single ephys unit.
    eventT : np.array
        Array of event times (e.g. presentation of stimulus or the time of a saccade)
        in seconds and with the type float.
    bandwidth : int
        Bandwidth of KDE filter in units of milliseconds.
    resample_size : int
        Size of binning when resampling spike rate, in units of milliseconds.
    edgedrop : int
        Time to pad at the start and end, and then dropped, to eliminate edge artifacts.
    win : int
        Window in time to use in positive and negative directions. For win=1000, the
        PSTH will start -1000 ms before the event and end +1000 ms after the event.

    Returns
    -------
    psth : np.array
        Peri-Stimulus Time Histogram

    """
    # Unit conversions
    bandwidth = bandwidth / 1000
    resample_size = resample_size / 1000
    win = win / 1000
    edgedrop = edgedrop / 1000
    edgedrop_ind = int(edgedrop / resample_size)

    bins = np.arange(-win-edgedrop, win+edgedrop+resample_size, resample_size)

    # Timestamps of spikes (`sps`) relative to `eventT`
    sps = []
    for i, t in enumerate(eventT):
        sp = spikeT-t
        # Only keep spikes in this window
        sp = sp[(sp <= (win+edgedrop)) & (sp >= (-win-edgedrop))] 
        sps.extend(sp)
    sps = np.array(sps)

    kernel = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(sps[:, np.newaxis])
    density = kernel.score_samples(bins[:, np.newaxis])

    # Multiply by the # spikes to get spike count per point. Divide
    # by # events for rate/event.
    psth = np.exp(density) * (np.size(sps ) / np.size(eventT))

    # Drop padding at start & end to eliminate edge effects.
    psth = psth[edgedrop_ind:-edgedrop_ind]

    return psth

def drop_nearby_events(thin, avoid, win=0.25):
    """Drop events that fall near others.

    When eliminating compensatory eye/head movements which fall right after
    gaze-shifting eye/head movements, `thin` should be the compensatory event
    times.

    Parameters
    ----------
    thin : np.array
        Array of timestamps (as float in units of seconds) that
        should be thinned out, removing any timestamps that fall
        within `win` seconds of timestamps in `avoid`.
    avoid : np.array
        Timestamps to avoid being near.
    win : np.array
        Time (in seconds) that times in `thin` must fall before or
        after items in `avoid` by.
    
    """
    to_drop = np.array([c for c in thin for g in avoid if ((g>(c-win)) & (g<(c+win)))])
    thinned = np.delete(thin, np.isin(thin, to_drop))
    return thinned

def drop_repeat_events(eventT, onset=True, win=0.020):
    """Eliminate saccades repeated over sequential camera frames.

    Saccades sometimes span sequential camera frames, so that two or
    three sequential camera frames are labaled as saccade events, despite
    only being a single eye/head movement. This function keeps only a
    single frame from the sequence, either the first or last in the
    sequence.

    Parameters
    ----------
    eventT : np.array
        Array of saccade times (in seconds as float).
    onset : bool
        If True, a sequence of back-to-back frames labeled as a saccade will
        be reduced to only the first/onset frame in the sequence. If false, the
        last in the sequence will be used.
    win : float
        Distance in time (in seconds) that frames must follow each other to be
        considered repeating. Frames are 0.016 ms, so the default value, 0.020
        requires that frames directly follow one another.

    Returns
    -------
    thinned : np.array
        Array of saccade times, with repeated labels for single events removed.

    """
    duplicates = set([])
    for t in eventT:
        if onset:
            # keep first
            new = eventT[((eventT-t)<win) & ((eventT-t)>0)]
        else:
            # keep last
            new = eventT[((t-eventT)<win) & ((t-eventT)>0)]
        duplicates.update(list(new))

    thinned = np.sort(np.setdiff1d(eventT, np.array(list(duplicates)), assume_unique=True))
    
    return thinned

def calc_sp_rate(spikeT, maxT, dT=0.025):
    """Get binned spike rate from spike times.

    array of arrays where spikes are indicated by a
    timestamp to 2D array at binned intervals of a
    spike rate
    """
    n_units = len(spikeT)
    time = np.arange(0, maxT, dT)
    n_sp = np.zeros([n_units, len(time)])
    bins = np.append(time, time[-1]+dT)

    for i in range(n_units):
        n_sp[i,:], _ = np.histogram(spikeT[i], bins)

    return n_sp

def calc_STA(spikes, stim, lags=None):
    """ Spike-triggered average for multiple cells at once
    
    default is to use lags [-2,0,2,4,6]
    if you want to do a single lag, use parameter lags=2 or lags=[2]

    """

    if lags is None:
        lags = np.arange(-2,8,2)
    if type(lags)==int:
        lags = list(lags)

    nks = np.shape(stim[0,:,:])

    # shape: [cell, lag, x, y]
    sta_out = np.zeros([np.size(spikes,0),
                        np.size(lags),
                        np.size(stim,1),
                        np.size(stim,2)])

    for c in range(np.size(spikes,0)): # cells
        for l_i, l in enumerate(lags): # stimulus lags

            sp = spikes[c,:].copy()
            sp = np.roll(sp, -l)

            sta = stim.T @ sp
            sta = np.reshape(sta, nks)
            nsp = np.sum(sp)

            if nsp < 1:
                sta = np.nan
            else:
                sta = sta / nsp
                sta = sta - np.mean(sta)

            sta_out[c,l_i,:,:] = sta.copy()

    return sta_out
            
def calc_STA_prelim(spikes, stim, lag=2):
    """ STA from preliminary data that hasn't been spike sorted
    
    """
    nks = np.shape(stim[0,:,:])
    all_sta = np.zeros([np.size(spikes,0),
                        np.size(stim,1),
                        np.size(stim,2)])

    # shape: [cell, x, y]
    sta_out = np.zeros([np.size(spikes,0),
                        np.size(stim,1),
                        np.size(stim,2)])

    for c in range(np.size(spikes,0)):

        sp = spikes[c,:].copy()
        sp = np.roll(sp, -lag)
        
        sta = stim.T @ sp
        sta = np.reshape(sta, nks)
        nsp = np.sum(sp)

        if nsp < 1:
            sta = np.nan
        else:
            sta = sta / nsp
            sta = sta - np.mean(sta)
            # rotate stimulus
            sta = np.fliplr(np.flipud(sta))

        sta_out[c,:,:] = sta.copy()

    return sta_out

def calc_worldshift(stim, theta, phi, eyeT, worldT,
               max_frames=3600):
    """ Approx shift of worldcam
    """

    dTheta = np.diff(scipy.interpolate.interp1d(eyeT, theta, bounds_error=False)(worldT))
    dPhi = np.diff(scipy.interpolate.interp1d(eyeT, phi, bounds_error=False)(worldT))

    num_iter = 5000
    term_eps = 1e-4
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, num_iter, term_eps)
    warp_mode = cv2.MOTION_TRANSLATION

    cc = np.zeros(max_frames)
    xshift = np.zeros(max_frames)
    yshift = np.zeros(max_frames)
    warp_all = np.zeros([6, max_frames])

    print('Get shift between adjacent frames')
    for f in tqdm(range(max_frames)):
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        try: 
            (cc[f], warp_matrix) = cv2.findTransformECC(stim[f, :, :],
                                        stim[f+1, :, :],
                                        warp_matrix,
                                        warp_mode,
                                        criteria,
                                        inputMask=None,
                                        gaussFiltSize=1)
            xshift[f] = warp_matrix[0,2]
            yshift[f] = warp_matrix[1,2]
        except:
            cc[f] = np.nan
            xshift[f] = np.nan
            yshift[f] = np.nan
    
    # Regression to predict frame shift based on eye shifts
    reg_x = sklearn.linear_model.LinearRegression()
    reg_y = sklearn.linear_model.LinearRegression()
    
    # Eye data as predictors
    eye_vals = np.zeros([max_frames, 2])
    eye_vals[:,0] = dTheta[:max_frames]
    eye_vals[:,1] = dPhi[:max_frames]

    # Shift in x and y as outputs
    xshiftdata = xshift[:max_frames]
    yshiftdata = yshift[:max_frames]

    # Only use data that is not a NaN, has high
    # correlation between frames, and has small eye
    # movements (i.e. no saccades, only compensatory
    # movements).
    use = ~np.isnan(eye_vals[:,0])       \
            & ~np.isnan(eye_vals[:,1])   \
            & (cc>0.95)                 \
            & (np.abs(eye_vals[:,0])<2)  \
            & (np.abs(eye_vals[:,1])<2)  \
            & (np.abs(xshiftdata)<5)    \
            & (np.abs(yshiftdata)<5)

    # Fit x/y shifts
    reg_x.fit(eye_vals[use,:], xshiftdata[use])
    xmap = reg_x.coef_
    xr_score = reg_x.score(eye_vals[use,:], xshiftdata[use])

    reg_y.fit(eye_vals[use,:], yshiftdata[use])
    ymap = reg_y.coef_
    yrscore = reg_y.score(eye_vals[use,:], yshiftdata[use])

    # Diagnostic plots
    fig, [[ax0,ax1],[ax2,ax3]] = plt.subplots(2,2, figsize=(11,8.5), dpi=300)

    ax0.plot(dTheta[:max_frames], xshift[:max_frames], 'k.')
    ax0.plot([-5,5], [5,-5], 'r')
    ax0.set_xlim(-12,12)
    ax0.set_ylim(-12,12)
    ax0.set_xlabel('dTheta')
    ax0.set_ylabel('x shift')
    ax0.set_title('xmap={}'.format(xmap))

    ax1.plot(dTheta[:max_frames], yshift[:max_frames], 'k.')
    ax1.plot([-5,5], [5,-5], 'r')
    ax1.set_xlim(-12,12)
    ax1.set_ylim(-12,12)
    ax1.set_xlabel('dTheta')
    ax1.set_ylabel('y shift')
    ax1.set_title('ymap={}'.format(ymap))
    
    ax2.plot(dPhi[:max_frames], xshift[:max_frames], 'k.')
    ax2.plot([-5,5], [5,-5], 'r')
    ax2.set_xlim(-12,12)
    ax2.set_ylim(-12,12)
    ax2.set_xlabel('dPhi')
    ax2.set_ylabel('x shift')
    
    ax3.plot(dPhi[:max_frames], yshift[:max_frames], 'k.')
    ax3.plot([-5,5], [5,-5], 'r')
    ax3.set_xlim(-12,12)
    ax3.set_ylim(-12,12)
    ax3.set_xlabel('dPhi')
    ax3.set_ylabel('y shift')

    fig.tight_layout()
    pdf.savefig()
    plt.close()

    return xmap, ymap

def calc_STV(spikes, stim):
    """ Spike-triggered average for multiple cells at once
    
    default is to use lags [-2,0,2,4,6]
    if you want to do a single lag, use parameter lags=2 or lags=[2]

    """
    lag = 2

    nks = np.shape(stim[0,:,:])
    stim = stim.copy()**2 # square image
    mean_img = np.mean(stim, axis=0)

    # shape: [cell, lag, x, y]
    stv_out = np.zeros([np.size(spikes,0),
                        np.size(stim,1),
                        np.size(stim,2)])

    for c in range(np.size(spikes,0)):

        sp = spikes[c,:].copy()
        sp = np.roll(sp, -lag)

        stv = np.nan_to_num(stim, 0).T @ sp
        stv = np.reshape(stv, nks)
        nsp = np.sum(sp)

        if nsp < 1:
            stv = np.nan
        else:
            stv = stv / nsp
            stv = stv - mean_img
            stv = stv - np.mean(stv)

        stv[c,:,:] = stv.copy()

    return stv_out

def calc_tuning(spikes, vals, valsT, binT, range):
    """Tuning curve

    spikes: dict where each key is a cell and value is a list of
        spike times for that cell
    vals: values for 
    valsT: timestamps matching vals
    binT: timebase to interpolate to
    """
    n_cells = len(spikes.keys())
    scatter = np.zeros((n_cells, len(vals)))
    tuning = np.zeros((n_cells, len(range)-1))
    err = tuning.copy()
    bins = np.zeros(len(range)-1)

    for x in range(len(range)-1):
        bins[x] = 0.5 * (range[x] + range[x+1])

    for i, c in enumerate(spikes.keys()):

        scatter[i,:] = scipy.interpolate.interp1d(valsT[:-1], spikes[c],  \
                                    bounds_error=False)(binT)
        
        for x in range(len(range)-1):

            use = (vals >= range[x]) & (vals < range[x+1])
            tuning[i,x] = np.nanmean(scatter[i, use])
            err[i,x] = np.nanstd(scatter[i, use]) / np.sqrt(np.count_nonzero(use))

    return tuning, err, bins

def calc_PSTH_hist(spikes, eventT):
    n_cells = len(spikes.keys())
    bins = np.arange(-1, 1.1, 0.025)

    psth_out = np.zeros((n_cells, bins.size-1))

    for i, c in enumerate(spikes.keys()): # cells

        sps = spikes[c].copy()

        for s in np.array(eventT): # event onset timestamps
            hist, _ = np.histogram(sps-s, bins)
            psth_out[i,:] = psth_out[i,:] + hist / (eventT.size * np.diff(bins))

    return psth_out

def quick_GLM_setup(eyeT, theta, phi, modelT, dT, img_norm, worldT,
                    imuT, gyro_z_raw, gyro_z, roll, pitch):

    dT = 0.025 # sec
    usethresh_eyes = 10
    usethresh_active = 40

    th = scipy.interpolate.interp1d(eyeT, theta, bounds_error=False)(modelT + dT/2)
    ph = scipy.interpolate.interp1d(eyeT, phi, bounds_error=False)(modelT + dT/2)

    ### Get active times
    if fm:
        gz_r = scipy.interpolate.interp1d(imuT,     \
                (gyro_z_raw - np.nanmean(gyro_z_raw)*7.5), bounds_error=False)(modelT)
        gz = scipy.interpolate.interp1d(imuT, gyro_z, bounds_error=False)(modelT)
        rl = scipy.interpolate.interp1d(imuT, roll, bounds_error=False)(modelT)
        pt = scipy.interpolate.interp1d(imuT, pitch, bounds_error=False)(modelT)
        
        model_active = np.convolve(np.abs(gz_r), np.ones(np.int(1/dT)), 'same')

        model_use = np.where((np.abs(th) < usethresh_eyes)            \
                             & (np.abs(ph) < usethresh_eyes)          \
                             & (model_active > usethresh_active))[0]

    else:
        model_use = np.array([True for i in range(len(th))])

    ### Set up video
    ds = 0.25 # stimulus downsample

    # Test shape with one frame
    testimg = img_norm[0,:,:]
    testimg = cv2.resize(testimg,   \
                    (int(np.shape(testimg)[1]*ds), int(np.shape(testimg)[0]*ds)))
    testimg = testimg[5:-5, 5:-5] # remove area affected by eye movement correction

    world_ds = np.zeros([np.size(img_norm,0),       \
                    np.int(np.size(testimg,0)*np.size(testimg,1))])

    for f in tqdm(range(np.size(img_norm,0))):

        s_img = cv2.resize(img_norm[f,:,:],                 \
                        (np.int(np.size(img_norm,2)*ds),    \
                         np.int(np.size(img_norm,1)*ds)),   \
                        interpolation=cv2.INTER_LINEAR_EXACT)

        s_img = s_img[5:-5, 5:-5]

        world_ds[f,:] = np.reshape(s_img, np.size(s_img,0)*np.shape(s_img,1))

    glm_input = scipy.interpolate.interp1d(worldT, world_ds,            \
                        'nearest', axis=0, bounds_error=False)(modelT)
    
    nks = np.shape(s_img)
    nk = nks[0]*nks[1]

    glm_input[np.isnan(glm_input)] = 0

    return glm_input, model_active, model_use

def quick_GLM_RFs(x, model_nsp, model_use, nks):
    """
    
    x: the model video
    """


    nT = np.size(model_nsp, 1)

    # Image dimensions
    nk  = nks[0] * nks[1]
    n_cells = np.shape(model_nsp)[0]

    # Subtract mean and renormalize -- necessary?
    mn_img = np.mean(x[model_use,:], axis=0)
    x = x - mn_img

    img_std = np.std(x[model_use,:], axis=0)
    x[:,img_std==0] = 0
    
    x = np.nan_to_num(x / img_std, 0)
    x = np.append(x, np.ones((nT, 1)), axis=1) # append column of ones
    x = x[model_use,:]

    # Set up prior matrix (regularizer)
    
    # L2 prior
    Imat = np.eye(nk)
    Imat = scipy.linalg.block_diag(Imat, np.zeros((1,1)))
    
    # Smoothness prior
    consecutive = np.ones((nk, 1))
    consecutive[nks[1]-1::nks[1]] = 0

    diff = np.zeros((1,2))
    diff[0,0] = -1
    diff[0,1] = 1

    Dxx = scipy.sparse.diags((consecutive @ diff).T,
                              np.array([0, 1]),
                              (nk-1, nk))

    Dxy = scipy.sparse.diags((np.ones((nk,1))@ diff).T,
                              np.array([0, nks[1]]),
                              (nk - nks[1], nk))

    Dx = Dxx.T @ Dxx  \
         + Dxy.T @ Dxy

    bD  = scipy.linalg.block_diag(Dx.toarray(), np.zeros((1,1)))

    # Summed prior matrix
    Cinv = bD + Imat
    lag_list = [-4, -2, 0, 2, 4]
    lambdas = 1024 * (2**np.arange(0,16))
    n_lam = len(lambdas)

    # Set up empty arrays for receptive field and cross correlation
    rf_out = np.zeros((n_cells,
                       len(lag_list),
                       nks[0],
                       nks[1]))

    cc_out = np.zeros((n_cells,
                       len(lag_list)))

    # Iterate through cells
    for c in tqdm(range(n_cells)):

        # Iterate through timing lags
        for lag_ind, lag in enumerate(lag_list):
            
            sps = np.roll(model_nsp[c,:], -lag)
            sps = sps[model_use]
            nT = len(sps)

            # Split training and test data
            test_frac = 0.3
            ntest = int(nT * test_frac)

            x_train = x[ntest:,:]
            sps_train = sps[ntest:]

            x_test = x[:ntest,:]
            sps_test = sps[:ntest]

            # Calculate a few terms
            rf = x_train.T @ sps_train / np.sum(sps_train)

            XXtr = x_train.T @ x_train
            XYtr = x_train.T @ sps_train

            msetrain = np.zeros((n_lam, 1))
            msetest = np.zeros((n_lam, 1))
            w_ridge = np.zeros((nk+1, n_lam))
            
            # Initial guess
            w = rf

            # Loop over regularization strength
            for l in range(len(lambdas)): 

                # Calculate MAP estimate         

                # equivalent of \ (left divide) in matlab      
                w = np.linalg.solve(XXtr + lambdas[l]*Cinv, XYtr) 
                w_ridge[:,l] = w
                
                # Calculate test and training rms error
                msetrain[l] = np.mean((sps_train - x_train @ w)**2)
                msetest[l] = np.mean((sps_test - x_test @ w)**2)

            # Select best cross-validated lambda for RF
            best_lambda = np.argmin(msetest)
            w = w_ridge[:, best_lambda]
            ridge_rf = w_ridge[:, best_lambda]

            rf_out[c,lag_ind,:,:] = np.reshape(w[:-1], nks)

            # Predicted firing rate
            sp_pred = x_test @ ridge_rf

            # Bin the firing rate to get smooth rate vs time
            bin_len = 80
            model_dt = 0.025
            sp_smooth = (np.convolve(sps_test, np.ones(bin_len), 'same')) / (bin_len * model_dt)
            pred_smooth = (np.convolve(sp_pred, np.ones(bin_len), 'same')) / (bin_len * model_dt)
            
            # Diagnostics
            err = np.mean((sp_smooth - pred_smooth)**2)
            cc = np.corrcoef(sp_smooth, pred_smooth)

            cc_out[c,lag_ind] = cc[0,1]

    return rf_out, cc_out

def get_delayed_frames(time, thresh=0.03, win=3):
    """
    thresh: (seconds) frames that arrive slower than this will be dropped
    """
    fast_inds = np.diff(time) <= thresh
    slow_inds = sorted(list(set(itertools.chain.from_iterable(  \
                [list(range(int(i) - win, int(i) + (win+1))) for i in np.where(fast_inds==False)[0]]))))

    return slow_inds

def apply_worldshift(worldVid, worldT, eyeT, theta, phi, x_c, y_c):
    """
    x_c: x correction
    y_c: y correction
    """

    th_interp = scipy.interpolate.interp1d(eyeT, theta, bounds_error=False)
    ph_interp = scipy.interpolate.interp1d(eyeT, phi, bounds_error=False)

    worldVid_shift = worldVid.copy()

    print('Applying shift to worldcam frames')
    for f in tqdm(range(np.size(worldVid, 0))):

        th = th_interp(worldT[f])
        ph = ph_interp(worldT[f])

        worldVid_shift[f,:,:] = scipy.ndimage.shift(worldVid[f,:,:],            \
                                       (-np.int8(th * y_c[0] + ph * y_c[1]),    \
                                        -np.int8(th * x_c[0] + ph * x_c[1])))

    def pupil_tuning(self):
        # pupil radius
        self.longaxis = self.eye_params.sel(ellipse_params='longaxis').copy()
        self.norm_longaxis = (self.longaxis - np.mean(self.longaxis)) / np.std(self.longaxis)
        
        # pupil radius over time
        plt.figure()
        plt.plot(self.eyeT, self.norm_longaxis, 'k')
        plt.xlabel('sec')
        plt.ylabel('normalized pupil radius')
        if self.figs_in_pdf:
            self.detail_pdf.savefig(); plt.close()
        elif not self.figs_in_pdf:
            plt.show()

        # rate vs pupil radius
        radius_range = np.linspace(10,50,10)
        self.pupilradius_tuning_bins, self.pupilradius_tuning, self.pupilradius_tuning_err = self.calc_tuning(self.longaxis, radius_range, self.eyeT, 'pupil radius')

        # normalize eye position
        self.norm_theta = (self.theta - np.nanmean(self.theta)) / np.nanstd(self.theta)
        self.norm_phi = (self.phi - np.nanmean(self.phi)) / np.nanstd(self.phi)

        plt.figure()
        plt.plot(self.eyeT[:3600], self.norm_theta[:3600], 'k')
        plt.xlabel('sec'); plt.ylabel('norm theta')
        if self.figs_in_pdf:
            self.diagnostic_pdf.savefig(); plt.close()
        elif not self.figs_in_pdf:
            plt.show()

        # theta tuning
        theta_range = np.linspace(-30,30,10)
        self.theta_tuning_bins, self.theta_tuning, self.theta_tuning_err = self.calc_tuning(self.theta, theta_range, self.eyeT, 'theta')

        # phi tuning
        phi_range = np.linspace(-30,30,10)
        self.phi_tuning_bins, self.phi_tuning, self.phi_tuning_err = self.calc_tuning(self.phi, phi_range, self.eyeT, 'phi')

    def mua_power_laminar_depth(self):
        # don't run for freely moving, at least for now, because recordings can be too long to fit ephys binary into memory
        # was only a problem for a 128ch recording
        # but hf recordings should be sufficient length to get good estimate
        # read in ephys binary
        lfp_ephys = self.read_binary_file()
        # subtract mean in time dim and apply bandpass filter
        ephys_center_sub = lfp_ephys - np.mean(lfp_ephys,0)
        filt_ephys = self.butter_bandpass(ephys_center_sub, order=6)
        # get lfp power profile for each channel
        lfp_power_profiles = np.zeros([self.num_channels])
        for ch in range(self.num_channels):
            lfp_power_profiles[ch] = np.sqrt(np.mean(filt_ephys[:,ch]**2)) # multiunit LFP power profile
        # median filter
        lfp_power_profiles_filt = signal.medfilt(lfp_power_profiles)
        if self.probe=='DB_P64-8':
            ch_spacing = 25/2
        else:
            ch_spacing = 25
        if self.num_channels==64:
            norm_profile_sh0 = lfp_power_profiles_filt[:32]/np.max(lfp_power_profiles_filt[:32])
            layer5_cent_sh0 = np.argmax(norm_profile_sh0)
            norm_profile_sh1 = lfp_power_profiles_filt[32:64]/np.max(lfp_power_profiles_filt[32:64])
            layer5_cent_sh1 = np.argmax(norm_profile_sh1)
            self.lfp_power_profiles = [norm_profile_sh0, norm_profile_sh1]
            self.lfp_layer5_centers = [layer5_cent_sh0, layer5_cent_sh1]
            plt.subplots(1,2)
            plt.subplot(1,2,1)
            plt.plot(norm_profile_sh0,range(0,32))
            plt.plot(norm_profile_sh0[layer5_cent_sh0]+0.01,layer5_cent_sh0,'r*',markersize=12)
            plt.ylim([33,-1]); plt.yticks(ticks=list(range(-1,33)),labels=(ch_spacing*np.arange(34)-(layer5_cent_sh0*ch_spacing)))
            plt.title('shank0')
            plt.subplot(1,2,2)
            plt.plot(norm_profile_sh1,range(0,32))
            plt.plot(norm_profile_sh1[layer5_cent_sh1]+0.01,layer5_cent_sh1,'r*',markersize=12)
            plt.ylim([33,-1]); plt.yticks(ticks=list(range(-1,33)),labels=(ch_spacing*np.arange(34)-(layer5_cent_sh1*ch_spacing)))
            plt.title('shank1')
            plt.tight_layout()
            if self.figs_in_pdf:
                self.detail_pdf.savefig(); plt.close()
            elif not self.figs_in_pdf:
                plt.show()
        elif self.num_channels==16:
            norm_profile_sh0 = lfp_power_profiles_filt[:16]/np.max(lfp_power_profiles_filt[:16])
            layer5_cent_sh0 = np.argmax(norm_profile_sh0)
            self.lfp_power_profiles = [norm_profile_sh0]
            self.lfp_layer5_centers = [layer5_cent_sh0]
            plt.figure()
            plt.tight_layout()
            plt.plot(norm_profile_sh0,range(0,16))
            plt.plot(norm_profile_sh0[layer5_cent_sh0]+0.01,layer5_cent_sh0,'r*',markersize=12)
            plt.ylim([17,-1]); plt.yticks(ticks=list(range(-1,17)),labels=(ch_spacing*np.arange(18)-(layer5_cent_sh0*ch_spacing)))
            plt.title('shank0')
            if self.figs_in_pdf:
                self.detail_pdf.savefig(); plt.close()
            elif not self.figs_in_pdf:
                plt.show()
        elif self.num_channels==128:
            norm_profile_sh0 = lfp_power_profiles_filt[:32]/np.max(lfp_power_profiles_filt[:32])
            layer5_cent_sh0 = np.argmax(norm_profile_sh0)
            norm_profile_sh1 = lfp_power_profiles_filt[32:64]/np.max(lfp_power_profiles_filt[32:64])
            layer5_cent_sh1 = np.argmax(norm_profile_sh1)
            norm_profile_sh2 = lfp_power_profiles_filt[64:96]/np.max(lfp_power_profiles_filt[64:96])
            layer5_cent_sh2 = np.argmax(norm_profile_sh2)
            norm_profile_sh3 = lfp_power_profiles_filt[96:128]/np.max(lfp_power_profiles_filt[96:128])
            layer5_cent_sh3 = np.argmax(norm_profile_sh3)
            self.lfp_power_profiles = [norm_profile_sh0, norm_profile_sh1, norm_profile_sh2, norm_profile_sh3]
            self.lfp_layer5_centers = [layer5_cent_sh0, layer5_cent_sh1, layer5_cent_sh2, layer5_cent_sh3]
            plt.subplots(1,4)
            plt.subplot(1,4,1)
            plt.plot(norm_profile_sh0,range(0,32))
            plt.plot(norm_profile_sh0[layer5_cent_sh0]+0.01,layer5_cent_sh0,'r*',markersize=12)
            plt.ylim([33,-1]); plt.yticks(ticks=list(range(-1,33)),labels=(ch_spacing*np.arange(34)-(layer5_cent_sh0*ch_spacing)))
            plt.title('shank0')
            plt.subplot(1,4,2)
            plt.plot(norm_profile_sh1,range(0,32))
            plt.plot(norm_profile_sh1[layer5_cent_sh1]+0.01,layer5_cent_sh1,'r*',markersize=12)
            plt.ylim([33,-1]); plt.yticks(ticks=list(range(-1,33)),labels=(ch_spacing*np.arange(34)-(layer5_cent_sh1*ch_spacing)))
            plt.title('shank1')
            plt.subplot(1,4,3)
            plt.plot(norm_profile_sh2,range(0,32))
            plt.plot(norm_profile_sh2[layer5_cent_sh2]+0.01,layer5_cent_sh2,'r*',markersize=12)
            plt.ylim([33,-1]); plt.yticks(ticks=list(range(-1,33)),labels=(ch_spacing*np.arange(34)-(layer5_cent_sh2*ch_spacing)))
            plt.title('shank2')
            plt.subplot(1,4,4)
            plt.plot(norm_profile_sh3,range(0,32))
            plt.plot(norm_profile_sh3[layer5_cent_sh3]+0.01,layer5_cent_sh3,'r*',markersize=12)
            plt.ylim([33,-1]); plt.yticks(ticks=list(range(-1,33)),labels=(ch_spacing*np.arange(34)-(layer5_cent_sh3*ch_spacing)))
            plt.title('shank3')
            plt.tight_layout()
            if self.figs_in_pdf:
                self.detail_pdf.savefig(); plt.close()
            elif not self.figs_in_pdf:
                plt.show()

    def base_ephys_analysis(self):
        print('gathering files')
        if not self.fm:
            self.gather_hf_files()
        elif self.fm:
            self.gather_fm_files()
        print('opening worldcam')
        self.open_worldcam()
        if self.fm:
            if self.stim == 'lt':
                print('opening topcam')
                self.open_topcam()
            print('opening imu')
            self.open_imu()
        if not self.fm:
            print('opening running ball')
            self.open_running_ball()
        print('opening ephys')
        self.open_cells()
        print('opening eyecam')
        self.open_eyecam()
        print('aligning timestamps to ephys')
        self.align_time()
        if self.fm and self.stim != 'dk' and self.do_rough_glm_fit:
            print('shifting worldcam for eye movements')
            self.estimate_visual_scene()
        print('dropping static worldcam pixels')
        self.drop_static_worldcam_pxls()
        if self.save_diagnostic_video:
            print('writing diagnostic video')
            self.diagnostic_video()
            self.diagnostic_audio()
            self.merge_video_with_audio()
        if self.fm:
            print('a few more diagnostic figures')
            self.head_and_eye_diagnostics()
        print('firing rates at new timebase')
        self.firing_rate_at_new_timebase()
        print('contrast response functions')
        self.contrast_tuning_bins, self.contrast_tuning, self.contrast_tuning_err = self.calc_tuning(self.contrast, self.contrast_range, self.worldT, 'contrast')
        print('mua power profile laminar depth')
        if self.stim == 'wn':
            self.mua_power_laminar_depth()
        print('interpolating worldcam data to match model timebase')
        self.worldcam_at_new_timebase()
        if self.fm and self.stim=='lt':
            print('interpolating topcam data to match model timebase')
            self.topcam_props_at_new_timebase()
        self.setup_model_spikes()
        print('calculating stas')
        self.calc_sta()
        print('calculating multilag stas')
        self.calc_multilag_sta()
        print('calculating stvs')
        self.calc_stv()
        if self.do_rough_glm_fit and ((self.fm and self.stim == 'lt') or self.stim == 'wn'):
            print('using glm to get receptive fields')
            self.rough_glm_setup()
            self.fit_glm_rfs()
        elif self.fm and (self.stim == 'dk' or not self.do_rough_glm_fit):
            print('getting active times without glm')
            self.get_active_times_without_glm()
        if not self.do_rough_glm_fit and self.do_glm_model_preprocessing and ((self.fm and self.stim == 'lt') or self.stim == 'wn'):
            print('preparing inputs for full glm model')
            self.rough_glm_setup()
        print('saccade psths')
        self.head_and_eye_movements()
        print('tuning to pupil properties')
        self.pupil_tuning()
        print('tuning to movement signals')
        self.movement_tuning()



def main():

    ### Gather files

    ### Read in data

    # Basic behavior-related calculations
    self.dEye = np.diff(self.theta) # deg/frame
    self.dEye_dps = self.dEye / np.diff(self.eyeT) # deg/sec

    ### Align timing

    eyeT = eyeT - ephysT0
    if eyeT[0] < -600:
        eyeT = eyeT + 8*60*60 # 8hr offset for some data
    worldT = worldT - ephysT0
    if worldT[0] < -600:
        worldT = worldT + 8*60*60

    if fm:
        imuT_raw = imuT_raw - ephysT0
        if stim=='lt':
            topT = topT - ephysT0
    elif not fm:
        ballT = ballT - ephysT0

    if fm is False:
        # Default values for offset and drift between
        # ephys aquisition and other data
        ephys_offset = 0.1
        ephys_drift = -0.000114

    # Plot eye velocity against head movements
    plt.figure
    plt.plot(self.eyeT[0:-1], -self.dEye, label='-dEye')
    plt.plot(self.imuT_raw, self.gyro_z, label='gyro z')
    plt.legend()
    plt.xlim(0,10); plt.xlabel('secs'); plt.ylabel('gyro (deg/s)')

    ### Apply timing correction to IMU and Ephys
    lag_range = np.arange(-0.2, 0.2, 0.002)
    cc = np.zeros(np.shape(lag_range))

    t1 = np.arange(5, len(dEye)/60 - 120, 20).astype(int)
    t2 = t1 + 60

    offset = np.zeros(np.shape(t1))
    ccmax = np.zeros(np.shape(t1))

    imu_interp = scipy.interpolate.interp1d(imuT, gyro_z)

    for i in tqdm(range(len(t1))):

        t1_i = t1[i] * 60
        t2_i = t2[i] * 60

        for l_i, lag in enumerate(lag_range):
            try:
                c, _ = utils.correlation.nanxcorr( -dEye[t1_i:t2_i],            \
                                         imu_interp(eyeT[t1_i:t2_i] + lag), 1)
                cc[l_i] = c[1]
            except:
                cc[l_i] = np.nan

        offset[i] = lag_range[np.argmax(cc)]    
        ccmax[i] = np.max(cc)

    offset[ccmax < 0.2] = np.nan

    # Plot IMU/eye alignment


    # Fit regression to timing drift
    reg = sklearn.linear_model.LinearRegression()
    time = np.array(eyeT[t1*60 + 30*60])

    reg.fit(time[~np.isnan(offset)].reshape(-1,1),  \
            offset[~np.isnan(offset)])

    ephys_offset = reg.intercept_
    ephys_drift = reg.coef_

    print('Ephys timing correction: offset={} drift={}'.format(ephys_offset, ephys_drift))

    # Plot regression timing fit

    imuT = imuT_raw - (ephys_offset + imuT_raw * ephys_drift)

    for c, sps in spikes.items():
        sps = np.array(sps)
        new_sps = sps - (ephys_offset + sps * ephys_drift)
        spikes[c] = new_sps

    ### Worldcam

    # Normalize
    std_img = np.std(stim, axis=0)
    img_norm = (stim - np.mean(stim, axis=0)) / std_img

    # Drop static pixels in stimulus (i.e. where are
    # the edges of the minitor in a head-fixed recording)
    std_img[std_img < 20] = 0
    img_norm = img_norm * (std_img > 0)

    # Contrast
    contrast = np.zeros(np.size(stim, 0))
    for f in range(np.size(stim, 0)):
        contrast[f] = np.nanstd(img_norm[f,:,:])

    # contrast over time
    plt.figure()
    plt.plot(self.contrast[2000:3000])
    plt.xlabel('frames')
    plt.ylabel('worldcam contrast')
    if self.figs_in_pdf:
        self.diagnostic_pdf.savefig(); plt.close()
    elif not self.figs_in_pdf:
        plt.show()
    # std of worldcam image
    fig = plt.figure()
    plt.imshow(std_im)
    plt.colorbar()
    plt.title('worldcam std img')
    if self.figs_in_pdf:
        self.diagnostic_pdf.savefig(); plt.close()
    elif not self.figs_in_pdf:
        plt.show()

    ### Bin spike times into firing rate
    
    model_dT = 0.025 # sec
    modelT = np.arange(0, np.max(worldT), model_dT)

    sp_rates = {}

    for c, sps in spikes.items():
        sp_r, _ = np.histogram(sps, modelT)
        sp_rates[c] = sp_r / model_dT

    ### Get model stimulus video from worldcam

    # Create interpolator for movie data in same timebase
    # as spike rate
    ds = 0.5
    sz = np.shape(img_norm)

    world_ds = np.zeros((int(sz[0]), # num frames
                         np.int(sz[1]*ds),
                         np.int(sz[2]*ds)))

    target_sz = (np.int(sz[2]*ds), np.int(sz[1]*ds))

    for f in range(sz[0]):
        world_ds[f,:,:] = cv2.resize(img_norm[f,:,:], target_sz)

    world_interp = scipy.interpolate.interp1d(worldT, world_ds, axis=0, bounds_error=False)

    # Create `stim`, the worldcam stimulus aligned to spike
    # data. `stim` is the *flatened* worldcam values to use
    # for STA, STV, etc. for the rest of analysis.

    nks = np.shape(world_ds[0,:,:])
    nk = nks[0]*nks[1]
    stim = np.zeros((len(modelT), nk))

    for i, t in enumerate(modelT):
        stim[i,:] = np.reshape(world_interp(t + model_dT/2), nk)
    
    stim[np.isnan(stim)] = 0

    # Get active times based on model gyro_z_raw
    mgzr = scipy.interpolate.interp1d(imuT,
            (gyro_z_raw - np.nanmean(gyro_z_raw)*7.5),
            bounds_error=False)(modelT)

    model_active = np.convolve(np.abs(mgzr),
            np.ones(np.int(1/model_dT)), 'same')

    ### Movement PSTHs
    saccthresh = { # deg/sec
        'head_moved': 60,
        'gaze_stationary': 120,
        'gaze_moved': 240
    }

    dHead = scipy.interpolate.interp1d(imuT, gyro_z, bounds_error=False)(eyeT)[:-1]

    # All eye movements
    all_dHead_left = eyeT[(dHead > saccthresh['head_moved'])]
    all_dHead_right = eyeT[(dHead < -saccthresh['head_moved'])]

    left_allhead_psth = {}
    right_allhead_psth = {}
    for c, sps in spikes.items():
        left_allhead_psth[c] = calc_PSTH(sps, all_dHead_left)
        right_allhead_psth[c] = calc_PSTH(sps, all_dHead_right)

    if fm is True:

        dGaze = dEye + dHead

        # Gaze shifts
        left_gazeshift_times = eyeT[(dHead > saccthresh['head_moved'])      \
                    & (dGaze > saccthresh['gaze_moved'])]
        right_gazeshift_times = eyeT[(dHead < -saccthresh['head_moved'])    \
                    & (dGaze < -saccthresh['gaze_moved'])]

        left_gazeshift_psth = {}
        right_gazeshift_psth = {}
        for c, sps in spikes.items():
            left_gazeshift_psth[c] = calc_PSTH(sps, left_gazeshift_times)
            right_gazeshift_psth[c] = calc_PSTH(sps, right_gazeshift_times)

        # Compensatory
        left_compensatory_times = eyeT[(dHead > saccthresh['head_moved'])   \
                    & (dGaze < saccthresh['gaze_stationary'])               \
                    & (dGaze > -saccthresh['gaze_stationary'])]
        right_compensatory_times = eyeT[(dHead < -saccthresh['head_moved']) \
                    & (dGaze > -saccthresh['gaze_stationary'])              \
                    & (dGaze < saccthresh['gaze_stationary'])]

        left_compensatory_psth = {}
        right_compensatory_psth = {}
        for c, sps in spikes.items():
            left_compensatory_psth[c] = calc_PSTH(sps, left_compensatory_times)
            right_compensatory_psth[c] = calc_PSTH(sps, right_compensatory_times)





plt.figure()
plt.hist(self.dEye_dps, bins=21, density=True)
plt.xlabel('dTheta')
plt.tight_layout()

plt.figure()
plt.hist(self.dGaze, bins=21, density=True)
plt.xlabel('dGaze')
if self.figs_in_pdf:
    self.detail_pdf.savefig(); plt.close()
elif not self.figs_in_pdf:
    plt.show()

plt.figure()
plt.hist(self.dHead, bins=21, density=True)
plt.xlabel('dHead')
if self.figs_in_pdf:
    self.detail_pdf.savefig(); plt.close()
elif not self.figs_in_pdf:
    plt.show()

plt.figure()
plt.plot(self.dEye_dps[::20], self.dHead[::20], 'k.')
plt.xlabel('dEye'); plt.ylabel('dHead')
plt.xlim((-900,900)); plt.ylim((-900,900))
plt.plot([-900,900], [900,-900], 'r:')
if self.figs_in_pdf:
    self.detail_pdf.savefig(); plt.close()
elif not self.figs_in_pdf:
    plt.show()




def movement_tuning(self):
    if self.fm:
        # get active times only
        active_interp = interp1d(self.model_t, self.model_active, bounds_error=False)
        active_imu = active_interp(self.imuT.values)
        use = np.where(active_imu > 40)
        imuT_use = self.imuT[use]

        # spike rate vs gyro x
        gx_range = np.linspace(-400,400,10)
        active_gx = self.gyro_x[use]
        self.gyrox_tuning_bins, self.gyrox_tuning, self.gyrox_tuning_err = self.calc_tuning(active_gx, gx_range, imuT_use, 'gyro x')

        # spike rate vs gyro y
        gy_range = np.linspace(-400,400,10)
        active_gy = self.gyro_y[use]
        self.gyroy_tuning_bins, self.gyroy_tuning, self.gyroy_tuning_err = self.calc_tuning(active_gy, gy_range, imuT_use, 'gyro y')
        
        # spike rate vs gyro z
        gz_range = np.linspace(-400,400,10)
        active_gz = self.gyro_z[use]
        self.gyroz_tuning_bins, self.gyroz_tuning, self.gyroz_tuning_err = self.calc_tuning(active_gz, gz_range, imuT_use, 'gyro z')

        # roll vs spike rate
        roll_range = np.linspace(-30,30,10)
        active_roll = self.roll[use]
        self.roll_tuning_bins, self.roll_tuning, self.roll_tuning_err = self.calc_tuning(active_roll, roll_range, imuT_use, 'head roll')

        # pitch vs spike rate
        pitch_range = np.linspace(-30,30,10)
        active_pitch = self.pitch[use]
        self.pitch_tuning_bins, self.pitch_tuning, self.pitch_tuning_err = self.calc_tuning(active_pitch, pitch_range, imuT_use, 'head pitch')

        # subtract mean from roll and pitch to center around zero
        centered_pitch = self.pitch - np.mean(self.pitch)
        centered_roll = self.roll - np.mean(self.roll)

        # interpolate to match eye timing
        pitch_interp = interp1d(self.imuT, centered_pitch, bounds_error=False)(self.eyeT)
        roll_interp = interp1d(self.imuT, centered_roll, bounds_error=False)(self.eyeT)

        # pitch vs theta
        plt.figure()
        plt.plot(pitch_interp[::100], self.theta[::100], 'k.'); plt.xlabel('head pitch'); plt.ylabel('theta')
        plt.ylim([-60,60]); plt.xlim([-60,60]); plt.plot([-60,60],[-60,60], 'r:')
        if self.figs_in_pdf:
            self.diagnostic_pdf.savefig(); plt.close()
        elif not self.figs_in_pdf:
            plt.show()

        # roll vs phi
        plt.figure()
        plt.plot(roll_interp[::100], self.phi[::100], 'k.'); plt.xlabel('head roll'); plt.ylabel('phi')
        plt.ylim([-60,60]); plt.xlim([-60,60]); plt.plot([-60,60],[60,-60], 'r:')
        if self.figs_in_pdf:
            self.diagnostic_pdf.savefig(); plt.close()
        elif not self.figs_in_pdf:
            plt.show()

        # roll vs theta
        plt.figure()
        plt.plot(roll_interp[::100], self.theta[::100], 'k.'); plt.xlabel('head roll'); plt.ylabel('theta')
        plt.ylim([-60,60]); plt.xlim([-60,60])
        if self.figs_in_pdf:
            self.diagnostic_pdf.savefig(); plt.close()
        elif not self.figs_in_pdf:
            plt.show()

        # pitch vs phi
        plt.figure()
        plt.plot(pitch_interp[::100], self.phi[::100], 'k.'); plt.xlabel('head pitch'); plt.ylabel('phi')
        plt.ylim([-60,60]); plt.xlim([-60,60])
        if self.figs_in_pdf:
            self.diagnostic_pdf.savefig(); plt.close()
        elif not self.figs_in_pdf:
            plt.show()

        # histogram of pitch values
        plt.figure()
        plt.hist(centered_pitch, bins=50); plt.xlabel('head pitch')
        if self.figs_in_pdf:
            self.diagnostic_pdf.savefig(); plt.close()
        elif not self.figs_in_pdf:
            plt.show()

        # histogram of pitch values
        plt.figure()
        plt.hist(centered_roll, bins=50); plt.xlabel('head roll')
        if self.figs_in_pdf:
            self.diagnostic_pdf.savefig(); plt.close()
        elif not self.figs_in_pdf:
            plt.show()

        # histogram of th values
        plt.figure()
        plt.hist(self.theta, bins=50); plt.xlabel('theta')
        if self.figs_in_pdf:
            self.diagnostic_pdf.savefig(); plt.close()
        elif not self.figs_in_pdf:
            plt.show()

        # histogram of pitch values
        plt.figure()
        plt.hist(self.phi, bins=50); plt.xlabel('phi')
        if self.figs_in_pdf:
            self.diagnostic_pdf.savefig(); plt.close()
        elif not self.figs_in_pdf:
            plt.show()

    elif not self.fm:
        ball_speed_range = [0, 0.01, 0.1, 0.2, 0.5, 1.0]
        self.ballspeed_tuning_bins, self.ballspeed_tuning, self.ballspeed_tuning_err = self.calc_tuning(self.ball_speed, ball_speed_range, self.ballT, 'running speed')


