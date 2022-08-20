import numpy as np
from datetime import datetime
import pandas as pd
 
def fmt_time(s):
       """ Read timestamps as a pd.Series and format time.
 
       Parameters
       --------
       s : pd.Series
           Timestamps as a Series.
           Expected to be formated as hours : minutes : seconds . microsecond
 
       Returns
       --------
       output_time : np.array
           Returned as the number of seconds that have passed since the
           previous midnight, with microescond precision, e.g. 700.000000
 
       """
       t_out = []
       fmt = '%H:%M:%S.%f'
 
       if s.dtype != np.float64:
           for t_in in s:
               t = str(t_in).strip()
 
               # If the timestamp has an unexpected precision
               try:
                   _ = datetime.strptime(t, fmt)
               except ValueError as v:
                   ulr = len(v.args[0].partition('unconverted data remains: ')[2])
                   if ulr:
                       t = t[:-ulr]
              
               try:
                   t_out.append((datetime.strptime(t, fmt) - datetime.strptime('00:00:00.000000', fmt)).total_seconds())
               except ValueError:
                   t_out.append(np.nan)
 
           t_out = np.array(t_out)
      
       else:
           t_out = s.values
 
       return t_out
 
def interp_time(t_in, use_medstep=False):
       """ Interpolate timestamps for double the number of frames. Compensates for video deinterlacing.
      
       Parameters
       --------
       camT : np.array
           Camera timestamps aquired at 30Hz
       use_medstep : bool
           When True, the median diff(camT) will be used as the timestep
           in interpolation. If False, the timestep between each frame will
           be used instead.
 
       Returns
       --------
       camT_out : np.array
           Timestamps of camera interpolated so that there are twice the number
           of timestamps in the array. Each timestamp in camT will be replaced by
           two, set equal distances from the original.
 
       """
       t_out = np.zeros(np.size(t_in, 0)*2)
 
       medstep = np.nanmedian(np.diff(t_in, axis=0))
 
       # Shift each deinterlaced frame by 0.5 frame periods forward/backwards
 
       if use_medstep:
           # Use the median timestep
           t_out[::2] = t_in - 0.25 * medstep
           t_out[1::2] = t_in + 0.25 * medstep
 
       elif not use_medstep:
           # Use the time step for each specific frame (this is preferred)
           steps = np.diff(t_in, axis=0, append=t_in[-1]+medstep)
           t_out[::2] = t_in
           t_out[1::2] = t_in + 0.5 * steps
 
       return t_out
 
def read_time(path, len_data=None, shift=False):
 
       """ Read timestamps from a .csv file.
 
       Parameters
       --------
       position_data_length : None or int
           Number of timesteps in data from deeplabcut. This is used to
           determine whether or not the number of timestamps is too short
           for the number of video frames.
           Eyecam and Worldcam will have half the number of timestamps as
           the number of frames, since they are aquired as an interlaced
           video and deinterlaced in analysis. To fix this, timestamps need
           to be interpolated.
 
       """
       # Read in times from the .csv file and format them as an array. Timestamps
       # should be one column. The first cell of the .csv (i.e. header) can either
       # be the first timestamp or the value `0` as a float or int.
       s = pd.read_csv(path, encoding='utf-8', engine='c', header=None).squeeze()
 
       # Deal with header.
       if s[0] == 0:
           s = s[1:]
 
       # Format time as string of seconds and convert it into a numpy array.
       t_out = fmt_time(s)
 
       # Compare timestamp length to its data (e.g. video) to see
       # if there is a 2x difference in length.
       # Or, shift without checking if `shift` is True.
       if ((len_data is not None) and (len_data > len(t_out))) or (shift is True):
           t_out = interp_time(t_out)
 
       return t_out