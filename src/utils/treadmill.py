import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

from src import filter, path, time, file


def sparse_to_constant_timebase(sparse_time, fixedinter_time, speed, seek_win=0.030):
    """ Adjust optical mouse data to match timestamps with constat time base, filling zeros
    for steps in time with no recorded sample.


    # output sample rate (data will be set up to match this sample rate,
        # since there is not constant sample rate for optical mouse data
        # from Bonsai)
    
    Parameters
    --------
    sparse_time : np.array
        Timestamps (in seconds) for samples do not exist when there was no change in data.
    fixedinter_time : np.array
        Timestamps (in seconds) with a constant step size, where start and end match sparse_time.
    speed : np.array
        Array of values that match the timebase of sparse_time.
    
    Returns
    --------
    data_out : np.array
        data with constant time step size and with zeros filled in where both data and
        sparse_time previously had no values
    """
    data_out = np.zeros(len(fixedinter_time))
    for t in sparse_time:
        ind = np.searchsorted(fixedinter_time, t)
        if ind < len(fixedinter_time):
            data_out[ind] = (speed[ind] if t >= (fixedinter_time[ind]-seek_win) and t <= fixedinter_time[ind] else 0)
    return data_out

def treadmill_speed(path, savepath,
                    set_samprate=0.008, center_x=960, center_y=540,
                    optmouse_pxls2cm=2840):
    """ Track the movement of the ball for headfixed recordings.

    samprate
    # float in seconds, window in which a previous timestamp in
    # sparse_time must fall, otherwise a zero will be filled in
    """

    # read in one csv file with timestamps, x position, and y position in three columns
    csv_data = pd.read_csv(path)

    # from this, we can get the timestamps, as seconds since midnight before the recording
    time = time.fmt_time(csv_data['Timestamp.TimeOfDay'])

    # convert center-subtracted pixels into cm
    x_pos = (csv_data['Value.X']-center_x) / optmouse_pxls2cm
    y_pos = (csv_data['Value.Y']-center_y) / optmouse_pxls2cm

    # set up new time base
    t0 = time[0]; t_end = time[-1]
    arange_time = np.arange(t0, t_end, set_samprate)

    # interpolation of xpos, ypos 
    xinterp = interp1d(time, x_pos, bounds_error=False, kind='nearest')(arange_time)
    yinterp = interp1d(time, y_pos, bounds_error=False, kind='nearest')(arange_time)

    # if no timestamp within 30ms, set interpolated val to 0
    full_x = sparse_to_constant_timebase(time, arange_time, xinterp)
    full_y = sparse_to_constant_timebase(time, arange_time, yinterp)

    # cm per second
    xpersec = full_x[:-1] / np.diff(arange_time)
    ypersec = full_y[:-1] / np.diff(arange_time)

    # speed
    speed = filter.convfilt(np.sqrt(xpersec**2 + ypersec**2), 10)

    # collect all data
    data = {
        'timestamps': time,
        'cm_x': full_x,
        'cm_y': full_y,
        'x_persec': xpersec,
        'y_persec': ypersec,
        'speed_cmpersec': speed
    }

    file.write_h5(savepath, data)