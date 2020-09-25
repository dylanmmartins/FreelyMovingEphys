"""
read_data.py

functions for reading in and manipulating data and time

Sept. 24, 2020
"""

# package imports
import pandas as pd
import numpy as np
import xarray as xr
from glob import glob
import os
import fnmatch
import dateutil
import cv2
from tqdm import tqdm

# glob for subdirectories
def find(pattern, path):
    result = [] # initialize the list as empty
    for root, dirs, files in os.walk(path): # walk though the path directory, and files
        for name in files:  # walk to the file in the directory
            if fnmatch.fnmatch(name,pattern):  # if the file matches the filetype append to list
                result.append(os.path.join(root,name))
    return result # return full list of file of a given type

# read in .h5 DLC files and manage column names
def open_h5(path):
    try:
        pts = pd.read_hdf(path)
    except ValueError:
        # read in .h5 file when there is a key set in corral_files.py
        pts = pd.read_hdf(path, key='data')
    # organize columns of pts
    pts.columns = [' '.join(col[:][1:3]).strip() for col in pts.columns.values]
    pts = pts.rename(columns={pts.columns[n]: pts.columns[n].replace(' ', '_') for n in range(len(pts.columns))})
    pt_loc_names = pts.columns.values

    return pts, pt_loc_names

# read in the timestamps for a camera and adjust to deinterlaced video length if needed
def open_time(path, dlc_len=None):
    # read in the timestamps if they've come directly from cameras
    read_time = pd.read_csv(open(path, 'rU'), encoding='utf-8', engine='c', header=None)
    time_in = pd.to_timedelta(read_time.squeeze(), unit='us', errors='coerce')

    # auto check if vids were deinterlaced
    if dlc_len is not None:
        # test length of the time just read in as it compares to the length of the data, correct for deinterlacing if needed
        timestep = np.median(np.diff(time_in, axis=0))
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

    return time_out

# convert xarray DataArray of DLC x and y positions and likelihood values into separate pandas data structures
def split_xyl(eye_names, eye_data, thresh):
    x_locs = []
    y_locs = []
    likeli_locs = []
    for loc_num in range(0, len(eye_names)):
        loc = eye_names[loc_num]
        if '_x' in loc:
            x_locs.append(loc)
        elif '_y' in loc:
            y_locs.append(loc)
        elif 'likeli' in loc:
            likeli_locs.append(loc)
    # get the xarray, split up into x, y,and likelihood
    for loc_num in range(0, len(likeli_locs)):
        pt_loc = likeli_locs[loc_num]
        if loc_num == 0:
            likeli_pts = eye_data.sel(point_loc=pt_loc)
        elif loc_num > 0:
            likeli_pts = xr.concat([likeli_pts, eye_data.sel(point_loc=pt_loc)], dim='point_loc', fill_value=np.nan)
    for loc_num in range(0, len(x_locs)):
        pt_loc = x_locs[loc_num]
        # threshold from likelihood
        eye_data.sel(point_loc=pt_loc)[eye_data.sel(point_loc=pt_loc) < thresh] = np.nan
        if loc_num == 0:
            x_pts = eye_data.sel(point_loc=pt_loc)
        elif loc_num > 0:
            x_pts = xr.concat([x_pts, eye_data.sel(point_loc=pt_loc)], dim='point_loc', fill_value=np.nan)
    for loc_num in range(0, len(y_locs)):
        pt_loc = y_locs[loc_num]
        # threshold from likelihood
        eye_data.sel(point_loc=pt_loc)[eye_data.sel(point_loc=pt_loc) < thresh] = np.nan
        if loc_num == 0:
            y_pts = eye_data.sel(point_loc=pt_loc)
        elif loc_num > 0:
            y_pts = xr.concat([y_pts, eye_data.sel(point_loc=pt_loc)], dim='point_loc', fill_value=np.nan)
    x_pts = xr.DataArray.squeeze(x_pts)
    y_pts = xr.DataArray.squeeze(y_pts)
    # convert to dataframe, transpose so points are columns
    x_vals = xr.DataArray.to_pandas(x_pts).T
    y_vals = xr.DataArray.to_pandas(y_pts).T

    return x_vals, y_vals, likeli_pts

# build an xarray DataArray of the a single camera's dlc point .h5 files and .csv timestamp corral_files
# function is used for any camera view regardless of type, though extension must be specified in 'view' argument
def h5_to_xr(pt_path, time_path, view):
    if pt_path is not None and pt_path != []:
        if isinstance(pt_path, list):
            pts, names = open_h5(pt_path[0])
        else:
            pts, names = open_h5(pt_path)
        if isinstance(time_path, list):
            time = open_time(time_path[0], len(pts))
        else:
            time = open_time(time_path, len(pts))
        xrpts = xr.DataArray(pts, dims=['frame', 'point_loc'])
        xrpts.name = view
        xrpts = xrpts.assign_coords(timestamps=('frame', time[1:])) # indexing [1:] into time because first row is the empty header, 0
    elif pt_path is None or pt_path == []:
        if time_path is not None and time_path != []:
            time = open_time(time_path)
            xrpts = xr.DataArray(np.zeros([len(time)-1]), dims=['frame'])
            xrpts = xrpts.assign_coords({'frame':range(0,len(xrpts))})
            xrpts = xrpts.assign_coords(timestamps=('frame', time[1:]))
            names = None
        elif time_path is None or time_path == []:
            xrpts = None; names = None

    return xrpts, names

# Sort out what the first timestamp in all DataArrays is so that videos can be set to start playing at the corresponding frame
def find_start_end(topdown_data, leftellipse_data, rightellipse_data, side_data):

    # bin the times
    topdown_binned = topdown_data.resample(time='10ms').mean()
    left_binned = leftellipse_data.resample(time='10ms').mean()
    right_binned = rightellipse_data.resample(time='10ms').mean()
    side_binned = side_data.resample(time='10ms').mean()

    # get binned times for each
    td_bintime = topdown_binned.coords['timestamps'].values
    le_bintime = left_binned.coords['timestamps'].values
    re_bintime = right_binned.coords['timestamps'].values
    sd_bintime = side_binned.coords['timestamps'].values

    print('topdown: ' + str(td_bintime[0]) + ' / ' + str(td_bintime[-1]))
    print('left: ' + str(le_bintime[0]) + ' / ' + str(le_bintime[-1]))
    print('right: ' + str(re_bintime[0]) + ' / ' + str(re_bintime[-1]))
    print('side: ' + str(sd_bintime[0]) + ' / ' + str(sd_bintime[-1]))

    # find the last timestamp to start a video
    first_real_time = max([td_bintime[0], le_bintime[0], re_bintime[0], sd_bintime[0]])

    # find the first end of a video
    last_real_time = min([td_bintime[-1], le_bintime[-1], re_bintime[-1], sd_bintime[-1]])

    # find which position contains the timestamp that matches first_real_time and last_real_time
    td_startframe = next(i for i, x in enumerate(td_bintime) if x == first_real_time)
    td_endframe = next(i for i, x in enumerate(td_bintime) if x == last_real_time)
    left_startframe = next(i for i, x in enumerate(le_bintime) if x == first_real_time)
    left_endframe = next(i for i, x in enumerate(le_bintime) if x == last_real_time)
    right_startframe = next(i for i, x in enumerate(re_bintime) if x == first_real_time)
    right_endframe = next(i for i, x in enumerate(re_bintime) if x == last_real_time)
    side_startframe = next(i for i, x in enumerate(sd_bintime) if x == first_real_time)
    side_endframe = next(i for i, x in enumerate(sd_bintime) if x == last_real_time)

    return td_startframe, td_endframe, left_startframe, left_endframe, right_startframe, right_endframe, side_startframe, side_endframe, first_real_time, last_real_time

# calculates xcorr ignoring NaNs without altering timing
# adapted from /niell-lab-analysis/freely moving/nanxcorr.m
def nanxcorr(x, y, maxlag=25):
    lags = range(-maxlag, maxlag)
    cc = []
    for i in range(0,len(lags)):
        # shift data
        yshift = np.roll(y, lags[i])
        # get index where values are usable in both x and yshift
        use = ~pd.isnull(x + yshift)
        # some restructuring
        x_arr = np.asarray(x, dtype=object); yshift_arr = np.asarray(yshift, dtype=object)
        x_use = x_arr[use]; yshift_use = yshift_arr[use]
        # normalize
        x_use = (x_use - np.mean(x_use)) / (np.std(x_use) * len(x_use))
        try:
            yshift_use = (yshift_use - np.mean(yshift_use)) / (np.std(yshift_use))
        except ZeroDivisionError:
            yshift_use = (yshift_use - np.mean(yshift_use))
        # get correlation
        cc.append(np.correlate(x_use, yshift_use))
    cc_out = np.hstack(np.stack(cc))
    return cc_out, lags

# add videos to xarray
# will downsample by ratio in config file and convert to black and white uint8
# might want to improve the way this is done -- requires lot of memory for each video
def format_frames(vid_path, config):
    print('formatting video into DataArray')
    vidread = cv2.VideoCapture(vid_path)
    all_frames = np.empty([int(vidread.get(cv2.CAP_PROP_FRAME_COUNT)),
                        int(vidread.get(cv2.CAP_PROP_FRAME_HEIGHT)*config['dwnsmpl']),
                        int(vidread.get(cv2.CAP_PROP_FRAME_WIDTH)*config['dwnsmpl'])])
    for frame_num in tqdm(range(0,int(vidread.get(cv2.CAP_PROP_FRAME_COUNT)))):
        ret, frame = vidread.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sframe = cv2.resize(frame, (0,0), fx=config['dwnsmpl'], fy=config['dwnsmpl'], interpolation=cv2.INTER_NEAREST)
        all_frames[frame_num,:,:] = sframe

    formatted_frames = xr.DataArray(all_frames.astype(np.int8), dims=['frame', 'height', 'width'])
    formatted_frames.assign_coords({'frame':range(0,len(formatted_frames))})
    del all_frames

    return formatted_frames

# align xarrays by time and merge
# first input will start at frame 0, the second input will be aligned to the first using timestamps in nanoseconds
# so that the first frame in a new dimension, 'merge_time', will start at either a positive or negative integer which
# is shifted forward or back from 0
def merge_xr_by_timestamps(xr1, xr2):
    # round the nanoseseconds in each xarray
    round1 = np.around(xr1['timestamps'].data.astype(np.int), -4)
    round2 = np.around(xr2['timestamps'].data.astype(np.int), -4)
    df2 = pd.DataFrame(round2)

    # where we'll put the index of the closest match in round2 for each value in round1
    ind = []
    for step in range(0,len(round1)):
        ind.append(np.argmin(abs(df2 - round1[step])))
    # here, a positive value means that round2 is delayed by that value
    # and a negative value means that round2 is ahead by that value
    delay_behind_other = int(round(np.mean([(i-ind[i]) for i in range(0,len(ind))])))

    # set the two dataarrays up with aligned timestamps
    new1 = xr1.expand_dims({'merge_time':range(0,len(xr1))})
    new2 = xr2.expand_dims({'merge_time':range(delay_behind_other, len(ind)+delay_behind_other)}).drop('timestamps')

    # merge into one dataset
    ds_out = xr.merge([new1,new2], dim='merge_time')

    return ds_out
    