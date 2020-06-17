#####################################################################################
"""
load_from_DLC.py of FreelyMovingEphys/main/
(formerly: load_all_csv.py)

Loads in top-down camera and right and/or left eye from DeepLabCut .h5 file outputs.
The user provides to this script with a file path for all of the data, collected by a
glob function, and an xarray DataArray is built from each of these camera viewpoints
and combined into one DataArray for all trials. The topdown data are processing and
ellipse parameters are collected for each of the eyes. Then, camera videos are
played back and saved out as a combined aligned video with feeds stitched side-by-side.
Finally, the ellipse parameters and preened topdown points are saved into .nc files.

Requires the functions in: topdown_preening.py
                           eye_tracking.py
                           time_management.py
                           data_reading.py
                           check_tracking.py
other files that are used by these above functions include:
                           data_cleaning.py
                           find.py
to preview camera feeds side-by-side, there is the function:
                           play_videos.py
to systematically rename files in a more readable format, there is the script:
                           corral_files.py
and data can be reopened and previewed after being saved out from this script with
                           read_nc_files.py

Adapted from GitHub repository /niell-lab-analysis/freely moving/loadAllCsv.m

TO DO:
- get it working with video read-in changes
- transform worldcam with eye direction
- start using find() from find.py instead of glob()
- move the timestep correction into time_management.py
- change figure=True/False parameter to two seperate ones: showfig=True to print it out in a window and makefig=True to
save it out without running the line plt.show()

last modified: June 16, 2020 by Dylan Martins (dmartins@uoregon.edu)
"""
#####################################################################################
from glob import glob
import os.path
import numpy as np
import xarray as xr
import pandas as pd

from utilities.topdown_preening import preen_topdown_data
from utilities.eye_tracking import eye_angles
from utilities.time_management import read_time
from utilities.data_reading import read_data
from utilities.check_tracking import parse_data_for_playback

####################################
##          USER INPUTS           ##
####################################
# set savepath for DLC points and videos
savepath_input = '/Users/dylanmartins/data/Niell/PreyCapture/Cohort3Outputs/J463c(blue)_110719/analysis_test_02/'

# find list of all data
main_path = '/Users/dylanmartins/data/Niell/PreyCapture/Cohort3/J463c(blue)/*/CorralledApproachData/'
vid_path = '/Users/dylanmartins/data/Niell/PreyCapture/Cohort3/J463c(blue)/*/CorralledApproachVids/'

# find the files wanted from the given main_path
# DeepLabCut point locations
topdown_file_list = set(glob(main_path + '*top*DeepCut*.h5')) - set(glob(main_path + '*DeInter*.h5'))
righteye_file_list = set(glob(main_path + '*eye1r*DeepCut*.h5')) - set(glob(main_path + '*DeInter*.h5'))
lefteye_file_list = set(glob(main_path + '*eye2l*DeepCut*.h5')) - set(glob(main_path + '*DeInter*.h5'))
# video files that those points come from
righteye_vid_list = set(glob(vid_path + '*eye1r*.avi')) - set(glob(vid_path + '*DeInter*.avi'))
lefteye_vid_list = set(glob(vid_path + '*eye2l*.avi')) - set(glob(vid_path + '*DeInter*.avi'))
topdown_vid_list = set(glob(vid_path + '*top*.avi')) - set(glob(vid_path + '*DeInter*.avi'))
worldcam_vid_list = set(glob(vid_path + '*world*.avi')) - set(glob(vid_path + '*DeInter*.avi'))
# accelerometer files
acc_file_list = glob(main_path + '*acc*.dat')
# camera time files
righteye_time_file_list = glob(main_path + '*eye1r*TS*.csv')
lefteye_time_file_list = glob(main_path + '*eye2l*TS*.csv')
topdown_time_file_list = glob(main_path + '*topTS*.csv')

# for testing purposes, limit number of topdown files read in. error will be raised if limit_of_loops is not 2 or
# greater; align_head_from_DLC wants to index through trials and it can only do that if there is more than one
limit_of_loops = 2

# loop through each topdown file and find the associated files
# then, read the data in for each set, and build from it an xarray DataArray
loop_count = 0

trial_id_list = []

topdown_file_list = [i for i in topdown_file_list if '1_110719_01' not in i]

for file in topdown_file_list:
    if loop_count < limit_of_loops:
        split_path = os.path.split(file)
        file_name = split_path[1]
        mouse_key = file_name[0:5]
        # trial_key was [17:27], that only caught the first of two previously possible ending digits
        # now set to end at 28 because of corralled data
        trial_key = file_name[17:28]

        # get a the other recorded files that are associated with the topdown file currently being read
        # find the accelerometer file (these aren't yet used)
        acc_files = [i for i in acc_file_list if mouse_key and trial_key in i]
        # find the right/left eye DLC files
        righteye_files = [i for i in righteye_file_list if mouse_key and trial_key in i]
        lefteye_files = [i for i in lefteye_file_list if mouse_key and trial_key in i]
        # find the camera time files
        topdown_time_files = [i for i in topdown_time_file_list if mouse_key and trial_key in i]
        lefteye_time_files = [i for i in lefteye_time_file_list if mouse_key and trial_key in i]
        righteye_time_files = [i for i in righteye_time_file_list if mouse_key and trial_key in i]

        # this turns it into a string from a list of only one string item
        lefteye_file = lefteye_files[0]
        righteye_file = righteye_files[0]
        topdown_time_file = topdown_time_files[0]
        lefteye_time_file = lefteye_time_files[0]
        righteye_time_file = righteye_time_files[0]

        # read in the data from file locations
        topdown_pts, topdown_names, lefteye_pts, lefteye_names, righteye_pts, righteye_names = read_data(file, lefteye_file, righteye_file)

        # make a unique name for the mouse and the recording trial
        trial_id = 'mouse_' + str(mouse_key) + '_trial_' + str(trial_key)
        trial_id_list.append(trial_id)

        # read in the time stamp data of each camera for this trial
        # also: extrapolate to what the last timepoint shoudl be, since time files always have one fewer length than point data
        # TO DO: move the timestep correction into time_management.py
        if topdown_time_file is not None:
            topdown_time, topdown_start = read_time(topdown_time_file)
            topdown_timestep = topdown_time[-1] - topdown_time[-2]
            topdown_time.append(topdown_time[-1] + topdown_timestep)
        elif topdown_time_file is None:
            topdown_time = None
        if lefteye_time_file is not None:
            lefteye_time, lefteye_start = read_time(lefteye_time_file)
            lefteye_timestep = lefteye_time[-1] - lefteye_time[-2]
            lefteye_time.append(lefteye_time[-1] + lefteye_timestep)
        elif lefteye_time_file is None:
            lefteye_time = None
        if righteye_time_file is not None:
            righteye_time, righteye_start = read_time(righteye_time_file)
            righteye_timestep = righteye_time[-1] - righteye_time[-2]
            righteye_time.append(righteye_time[-1] + righteye_timestep)
        elif righteye_time_file is None:
            righteye_time = None

        print('building xarrays for trial ' + trial_id)
        # build one DataArray that stacks up all topdown trials in separate dimensions
        if topdown_time is not None:
            if loop_count == 0:
                topdown = xr.DataArray(topdown_pts, coords=[topdown_time, topdown_pts.columns], dims=['time', 'point_loc'])
                topdown['trial'] = trial_id
            elif loop_count > 0:
                topdown_trial = xr.DataArray(topdown_pts, coords=[topdown_time, topdown_pts.columns], dims=['time', 'point_loc'])
                topdown_trial['trial'] = trial_id
                topdown = xr.concat([topdown, topdown_trial], dim='trial', fill_value=np.nan)
        elif topdown_time is None:
            print('trial ' + trial_id + ' has no topdown time data')

        # build one DataArray that stacks up all trials in separate dimensions for each of two possible eyes
        if lefteye_pts is not None and lefteye_time is not None:
            if loop_count == 0:
                lefteye = xr.DataArray(lefteye_pts, coords=[lefteye_time, lefteye_pts.columns], dims=['time', 'point_loc'])
                lefteye['trial'] = trial_id
                # create a DaraFrame of all trial's timestamps for the left and right eye so that they can be added
                # back to the theta, phi, etc. ellipse parameter DataArrays
                lefteye_time_df = pd.DataFrame(lefteye_time,columns=[trial_id])
            elif loop_count > 0:
                lefteye_trial = xr.DataArray(lefteye_pts, coords=[lefteye_time, lefteye_pts.columns], dims=['time', 'point_loc'])
                lefteye_trial['trial'] = trial_id
                lefteye = xr.concat([lefteye, lefteye_trial], dim='trial', fill_value=np.nan)
                lefteye_time_df_to_append = pd.DataFrame(lefteye_time,columns=[trial_id])
                lefteye_time_df = lefteye_time_df.join(lefteye_time_df_to_append)
        elif lefteye_pts is None or lefteye_time is None:
            print('trial ' + trial_id + ' has no left eye camera data')

        if righteye_pts is not None and righteye_time is not None:
            if loop_count == 0:
                righteye = xr.DataArray(righteye_pts, coords=[righteye_time, righteye_pts.columns], dims=['time', 'point_loc'])
                righteye['trial'] = trial_id
                righteye_time_df = pd.DataFrame(righteye_time,columns=[trial_id])
            elif loop_count > 0:
                righteye_trial = xr.DataArray(righteye_pts, coords=[righteye_time, righteye_pts.columns], dims=['time', 'point_loc'])
                righteye_trial['trial'] = trial_id
                righteye = xr.concat([righteye, righteye_trial], dim='trial', fill_value=np.nan)
                righteye_time_df_to_append = pd.DataFrame(righteye_time,columns=[trial_id])
                righteye_time_df = righteye_time_df.join(righteye_time_df_to_append)
        elif righteye_pts is None or righteye_time is None:
            print('trial ' + trial_id + ' has no right eye camera data')

        loop_count = loop_count + 1

# run through each topdown trial, correct y-coordinates, and threshold point liklihoods
print('preening top-down points')
preened_topdown = preen_topdown_data(topdown, trial_id_list, topdown_names, savepath_input, figures=False, coord_correction_val=0)

print('getting left eye angles')
left_ellipse = eye_angles(lefteye, lefteye_names, trial_id_list, savepath_input, lefteye_time_df, figures=False, side='left')
print('getting right eye angles')
right_ellipse = eye_angles(righteye, righteye_names, trial_id_list, savepath_input, righteye_time_df, figures=False, side='right')

# playback the videos and save out a combined alinged video with feeds stitched side-by-side
print('parsing data and video files for plotting and playback')
parse_data_for_playback(savepath_input, trial_id_list, preened_topdown, left_ellipse, right_ellipse,
                        topdown_vid_list, lefteye_vid_list, righteye_vid_list, worldcam_vid_list,
                        topdown_names, lefteye_names, righteye_names)

# save out the xarrays as .nc files
print('saving out xarray data')
preened_topdown.to_netcdf(savepath_input + 'topdown_points.nc')
left_ellipse.to_netcdf(savepath_input + 'leftellipse_points.nc')
right_ellipse.to_netcdf(savepath_input + 'right_ellipse_points.nc')