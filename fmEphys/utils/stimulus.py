"""
Ephys analysis that isn't stimulus-specific. just aligning everything.
"""

import os
import cv2
import numpy as np
import pandas as pd
import scipy.interpolate
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import fmEphys

def timing_alignment(ephysT0, eyeT, worldT,
                    topT=None, ballT=None, imuT=None):

    eyeT -= ephysT0
    worldT -= ephysT0

    # 8-hour offset for some data
    if eyeT[0] < -600:
        eyeT += 8*60*60
    if worldT[0] < -600:
        worldT += 8*60*60

    if topT is not None:
        topT -= ephysT0
    if ballT is not None:
        ballT -= ephysT0
    if imuT is not None:
        imuT -= ephysT0

def head_fixed(rpath):

    rname = make_recording_name(rpath)

    savepath = os.path.join(rpath, '{}_stim_responses.h5'.format(rname))

    # Sometimes the existing h5 cannot be overwritten, so it's better to delete it.
    if os.path.isfile(savepath):
        os.remove(savepath)

    # Paths of behavior data
    reye_path = os.path.join(rpath, '{}_Reye.h5'.format(rname))
    world_path = os.path.join(rpath, '{}_World.h5'.format(rname))
    treadmill_path = os.path.join(rpath, '{}_Treadmill.h5'.format(rname))

    # Paths of ephys data
    ephys_spike_path = os.path.join(rpath, '{}_ephys_merge.json'.format(rname))
    ephys_binary_path = os.path.join(rpath, '{}_Ephys.bin'.format(rname))
    
    # Open worldcam


def main(cfg):

    pdf_savepath = os.path.join(cfg['rpath'],
            '{}_stim_analysis.pdf'.format(cfg['rname']))
    pdf = PdfPages(pdf_savepath)

    
    fig1, [[ax1A,ax1B,ax1C],[ax1D,ax1E,ax1F]] = plt.subplots(2,3, figsize=(11,8.5), dpi=300)

    spikes ## dictionary of spike data for each cell with a 0:n_cells index as the key

    ### Plot spike raster
    for i, c in enumerate(sorted(spikes.keys())):
        sp = np.array(spikes[c])
        ax1A.vlines(sp[sp<10], i-0.25, i+0.25)

    ax1A.set_xlim(0, 10) # in sec    
    ax1A.set_xlabel('time (s)')
    ax1A.set_ylabel('cells')
    ax1A.set_ylim([len(spikes.keys()), 0])

    ### Plot scatter of horizontal and vertical
    # eye position
    frac_good = np.sum(~np.isnan(theta)) / len(theta)
    ax1B.plot(theta, phi, 'k.', markersize=4)
    ax1B.set_xlabel('theta')
    ax1B.set_ylabel('phi')
    ax1B.set_title('{:.3}% good'.format(frac_good*100))
    
    ### Plot theta
    # If deinterlacing shifted frames in the wrong direction,
    # the line will have a rough, zig-zag texture. Plot starts
    # 35 s into video and lasts 5 s.
    th_start = 35*60; th_stop = 40*60
    th_switch = np.zeros(np.shape(theta))
    th_switch[0:-1:2] = np.array(theta[1::2].copy())
    th_switch[1::2] = np.array(theta[0:-1:2].copy())
    ax1C.plot(theta[th_start:th_stop], color='k', label='actual')
    ax1C.plot(th_switch[th_start:th_stop], color='r', label='flipped')
    ax1C.legend()


    # figures specific to freely moving recordings
    fig2, [[ax2A,ax2B,ax2C],[ax2D,ax2E,ax2F]] = plt.subplots(2,3, figsize=(11,8.5), dpi=300)
    ### Plot IMU/eye alignment
    # offset
    ax2A.plot(eyeT[t1*60], offset)
    ax2A.xlabel('time (s)')
    ax2A.ylabel('offset (s)')
    ax2A.set_title('IMU/eye alignment')
    # max cross correlation
    ax2B.plot(eyeT[t1*60], ccmax)
    ax2B.xlabel('time (s)')
    ax2B.ylabel('max cc')
    ax2B.set_title('IMU/eye alignment')

    ### Plot regression timing fit
    dataT = dataT[~np.isnan(dataT)]
    offset = offset[~np.isnan(dataT)]
    ax2C.plot(dataT, offset, 'k.')
    ax2C.plot(dataT, ephys_offset + dataT * ephys_drift, color='r')
    ax2C.xlabel('time (s)')
    ax2C.ylabel('offset (s)')
    ax2C.set_title('offset={:.4} drift={:.4}'.format(ephys_offset, ephys_drift))

    ### Some eye/head diagnostics
    ax2D.plot(eyeT[:-1], np.diff(theta), label='dTheta')
    ax2D.plot(imuT-0.1, (gyro_z_raw-3)*10, label='raw gyro z')
    ax2D.legend()
    ax2D.set_xlabel('time (s)')
    ax2D.set_xlim(30,40)
    ax2D.set_ylim(-12,12)
    
    gyroZ_interp = scipy.interpolate.interp1d(imuT, gyro_z, bounds_error=False)(eyeT)
    ax2E.plot(eyeT[:-1], dEye, label='dEye')
    ax2E.plot(eyeT, gyroZ_interp, label='dHead')
    ax2E.legend()
    ax2E.set_xlim(37,39)
    ax2E.set_ylim(-10,10)
    ax2E.set_ylabel('deg')
    ax2E.set_xlabel('time (s)')

    ax2F.plot(eyeT, np.nancumsum(gyroZ_interp), label='head')
    ax2F.plot(eyeT, gyroZ_interp+dEye[:-1], label='gaze')
    ax2F.plot(eyeT, theta, label='theta')
    ax2F.set_xlim(35,40)
    ax2F.set_ylim(-30,30)
    ax2F.legend()
    ax2F.set_ylabel('deg')
    ax2F.set_xlabel('time (s)')

    fig2.tight_layout()
    pdf.savefig(); plt.close()

    fig3, [[ax3A,ax3B,ax3C],[ax3D,ax3E,ax3F]] = plt.subplots(2,3, figsize=(11,8.5), dpi=300)

    ### Plot eye timestamps
    ax3A.plot(np.diff(eyeT)[0:-1:10])
    ax3A.set_xticks(np.linspace(0, (len(eyeT)-1)/10, 10))
    ax3A.set_xlabel('frame')
    ax3A.set_ylabel('diff(eyeT)')

    ax3B.hist(np.diff(eyeT), bins=100)
    ax3B.set_xlabel('diff(eyeT)')







    plt.figure()
    plt.imshow(np.mean(self.world_vid, axis=0))
    plt.title('mean world image')
    if self.figs_in_pdf:
        self.diagnostic_pdf.savefig(); plt.close()
    elif not self.figs_in_pdf:
        plt.show()

    # world timestamps
    self.worldT = world_data.timestamps.copy()
    # plot timing
    fig = plt.subplots(1,2,figsize=(15,6))
    plt.subplot(1,2,1)
    plt.plot(np.diff(self.worldT)[0:-1:10])
    plt.xlabel('every 10th frame')
    plt.ylabel('deltaT')
    plt.title('worldcam')
    plt.subplot(1,2,2)
    plt.hist(np.diff(self.worldT), 100)
    plt.xlabel('deltaT')
    plt.tight_layout()
    if self.figs_in_pdf:
        self.diagnostic_pdf.savefig(); plt.close()
    elif not self.figs_in_pdf:
        plt.show()

    # figure of gyro z
    plt.figure()
    plt.plot(self.gyro_x[0:100*60])
    plt.title('gyro z (deg)')
    plt.xlabel('frame')
    if self.figs_in_pdf:
        self.diagnostic_pdf.savefig(); plt.close()
    elif not self.figs_in_pdf:
        plt.show()


    plt.figure()
    plt.plot(self.ballT, self.ball_speed)
    plt.xlabel('sec'); plt.ylabel('running speed (cm/sec)')
    if self.figs_in_pdf:
        self.diagnostic_pdf.savefig(); plt.close()
    elif not self.figs_in_pdf:
        plt.show()

def open_spike_data(ephys_json_path,
                    do_sorting=True):
    
    ephys_data = pd.read_json(ephys_json_path)
    
    # Sort units by shank and site order
    if do_sorting:
        ephys_data = ephys_data.sort_values(by='ch', axis=0, ascending=True)
        ephys_data = ephys_data.reset_index()
        ephys_data = ephys_data.drop('index', axis=1)
    
    # Select good cells from Phy2
    ephys_data = ephys_data.loc[ephys_data['group']=='good']
    
    # Start of aquisition
    t0 = ephys_data.iloc[0,12]

    # Spike times
    sps = ephys_data['spikeT'].copy()

    spikes = {}
    for ind, row in sps.iterrows():
        spikes[int(ind)] = row

    return spikes, t0

def open_eye_data(eye_data_path):

    eye_data = utils.file.read_h5(eye_data_path)
    eyeVid = eye_data['video'].astype(np.uint8)
    eyeT = eye_data['timestamps']
    
    # Eye orientation
    theta = np.rad2deg(eye_data['theta'])
    phi = np.rad2deg(eye_data['phi'])
    
    # Mean-center
    theta = theta - np.nanmean(theta)
    phi = phi - np.nanmean(phi) 
    phi = - phi # flip so up is positive and down is negative


def open_world_data(world_data_path):

    world_data = utils.file.read_h5(world_data_path)
    worldVid_raw = world_data['video'].astype(np.uint8)

    # Resize video if too large
    # Want it to be 60x80 pxls
    ds = 0.5
    if np.size(worldVid_raw, 1) >= 160:
        sz = np.shape(worldVid_raw)
        worldVid = np.zeros((int(sz[0]), # number of frames
                             int(sz[1]*ds),
                             int(sz[2]*ds)),
                             dtype='uint8')

        target_sz = (int(sz[2]*ds), int(sz[1]*ds))

        for f in range(sz[0]):
            worldVid[f,:,:] = cv2.resize(worldVid_raw[f,:,:], target_sz)
    else:
        worldVid = worldVid_raw.copy()

def open_IMU_data(imu_data_path):

    imu_data = utils.file.read_h5(imu_data_path)

    imuT = imu_data['timestamps']

    return imu_data

def open_top_data(top_data_path):

    top_data = utils.file.read_h5(top_data_path)

    topT = top_data['timestamps']