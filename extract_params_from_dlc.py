"""
extract_params_from_dlc.py

extract mouse and prey parameters from videos and DeepLabCut outputs

Sept. 24, 2020
"""

# package imports
from glob import glob
import os.path
import numpy as np
import xarray as xr
import pandas as pd
import argparse
import warnings
import json
from multiprocessing import freeze_support

# module imports
from util.read_data import h5_to_xr, find, format_frames, merge_xr_by_timestamps
from util.track_topdown import topdown_tracking, head_angle, plot_top_vid, get_top_props
from util.track_eye import plot_eye_vid, eye_tracking
from util.track_world import adjust_world, find_pupil_rotation, pupil_rotation_wrapper
from util.analyze_jump import jump_gaze_trace
from util.ephys import format_spikes

# get user inputs
parser = argparse.ArgumentParser(description='extract mouse and prey parameters from DeepLabCut data and corresponding videos')
parser.add_argument('-c', '--json_config_path', help='path to .json config file')
args = parser.parse_args()

def main():
    print('config file: ' + args.json_config_path)

    # open config file
    with open(args.json_config_path, 'r') as fp:
        config = json.load(fp)

    # get trial name out of each h5 file and make a list of the unique entries
    trial_paths = []
    for h5 in find('*.h5', config['data_path']):
        split_name = h5.split('_')[:-1]
        trial = '_'.join(split_name)
        if trial not in trial_paths:
            trial_paths.append(trial)

    # go into each trial and get out the camera/ephys types according to what's listed in json file
    for trial_path in trial_paths:
        trial_cam_h5 = [(trial_path +'_{}.h5').format(name) for name in config['camera_names']]
        if config['use_BonsaiTS'] is True:
            trial_cam_csv = [(trial_path+'_{}_BonsaiTS.csv').format(name) for name in config['camera_names']]
        elif config['use_BonsaiTS'] is False:
            trial_cam_csv = [(trial_path+'_{}_FlirTS.csv').format(name) for name in config['camera_names']]
        trial_cam_avi = [(trial_path+'_{}.avi').format(name) for name in config['camera_names']]
        t_name = os.path.split(trial_path)[1]

        # make the save path if it doesn't exist
        if not os.path.exists(config['save_path']):
            os.makedirs(config['save_path'])

        # format the ephys data
        if config['has_ephys'] is True:
            print('formatting electrophysiology recordings for ' + t_name)
            # filter the list of files for the current tiral to get to the ephys data
            # presently, the phy2 files are expected to be in a subdirectory of 
            trial_spike_times = os.path.join(config['data_path'],'ephys','spike_times.npy')
            trial_spike_clusters = os.path.join(config['data_path'],'ephys','spike_clusters.npy')
            trial_cluster_group = os.path.join(config['data_path'],'ephys','cluster_group.tsv')
            trial_ephys_time = os.path.join(config['data_path'],t_name+'Ephys_BonsaiTS.csv')
            # read in the data and structure them in one xarray for all spikes during this trial
            ephys_xr = format_spikes(trial_spike_times, trial_spike_clusters, trial_cluster_group, trial_ephys_time, config)
            # save out the data
            ephys_xr.to_netcdf(os.path.join(config['save_path'], str(t_name+'ephys.nc')), engine='netcdf4')

        # analyze top views
        if 'TOP' in config['camera_names']:
            print('tracking TOP for ' + t_name)
            # filter the list of files for the current trial to get the topdown view
            top_h5 = list(filter(lambda a: 'TOP' in a, trial_cam_h5))[0]
            top_csv = list(filter(lambda a: 'TOP' in a, trial_cam_csv))[0]
            top_avi1 = list(filter(lambda a: 'TOP1' in a, trial_cam_avi))[0]
            top_avi2 = list(filter(lambda a: 'TOP2' in a, trial_cam_avi))[0]
            top_avi3 = list(filter(lambda a: 'TOP3' in a, trial_cam_avi))[0]
            # make an xarray of dlc point values out of the found .h5 files
            # also assign timestamps as coordinates of the xarray
            topdlc, topnames = h5_to_xr(top_h5, top_csv, 'TOP')
            # clean DLC points up
            pts, noseX, noseY = topdown_tracking(topdlc, config, t_name) #key_save_path, key, args.lik_thresh, args.coord_cor, args.topdown_pt_num, args.cricket
            # calculate head angle
            print('finding head angle and mouse/cricket properties')
            head_theta = head_angle(pts, noseX, noseY, config, t_name) # args.lik_thresh, key_save_path, args.cricket, key, nose_x, nose_y
            # get mouse properties (and cricket properties if there is one)
            top_props = get_top_props(pts, head_theta, config, t_name)
            # make videos (only saved if config says so)
            if config['save_vids'] is True:
                print('plotting points on top video')
                plot_top_vid(top_avi1, pts, head_theta, config, t_name)
            # make xarray of video frames
            xr_top_frames1 = format_frames(top_avi1, config); xr_top_frames1.name = 'top1_video'
            xr_top_frames2 = format_frames(top_avi2, config); xr_top_frames2.name = 'top2_video'
            xr_top_frames3 = format_frames(top_avi3, config); xr_top_frames3.name = 'top3_video'
            # name and organize data
            pts.name = 'top_dlc_pts'; head_theta.name = 'top_head_theta'; top_props = 'top_properties'
            trial_top1_data = xr.merge([pts, head_theta, top_props, xr_top_frames1])
            trial_top1_data.to_netcdf(os.path.join(config['save_path'], str(t_name+'top1.nc')), engine='netcdf4', encoding={'top1_video':{"zlib": True, "complevel": 9}})
            trial_top2_data = xr.merge([pts, head_theta, top_props, xr_top_frames2])
            trial_top2_data.to_netcdf(os.path.join(config['save_path'], str(t_name+'top2.nc')), engine='netcdf4', encoding={'top2_video':{"zlib": True, "complevel": 9}})
            trial_top3_data = xr.merge([pts, head_theta, top_props, xr_top_frames3])
            trial_top3_data.to_netcdf(os.path.join(config['save_path'], str(t_name+'top3.nc')), engine='netcdf4', encoding={'top3_video':{"zlib": True, "complevel": 9}})


        # analyze eye views
        eye_sides = []
        if 'REYE' in config['camera_names']:
            eye_sides.append('R')
        if 'LEYE' in config['camera_names']:
            eye_sides.append('L')
        for i in range(0,len(eye_sides)):
            eye_side = eye_sides[i]
            print('tracking ' + eye_side + 'EYE for ' + t_name)
            # filter the list of files for the current trial to get the eye of this side
            eye_h5 = list(filter(lambda a: (eye_side+'EYE') in a, trial_cam_h5))[0]
            eye_csv = list(filter(lambda a: (eye_side+'EYE') in a, trial_cam_csv))[0]
            eye_avi = list(filter(lambda a: (eye_side+'EYE') in a, trial_cam_avi))[0]
            # make an xarray of dlc point values out of the found .h5 files
            # also assign timestamps as coordinates of the xarray
            eyedlc, eyenames = h5_to_xr(eye_h5, eye_csv, (eye_side+'EYE'))
            # get ellipse parameters from dlc points
            print('fitting ellipse to pupil')
            eyeparams = eye_tracking(eyedlc, config)
            # get pupil rotation and plot video -- slow step
            # rfit, shift = pupil_rotation_wrapper(eyeparams, config, t_name, eye_side)
            # make videos (only if config says so)
            if config['save_vids'] is True:
                print('plotting parameters on video')
                plot_eye_vid(eye_avi, eyedlc, eyeparams, config, t_name, eye_side)
            # make xarray of video frames
            xr_eye_frames = format_frames(eye_avi, config)
            # name and organize data
            eyedlc.name = eye_side+'eye_dlc_pts'; eyeparams.name = eye_side+'eye_ellipse_params'
            # rfit.name = eye_side+'eye_radius_fit'; shift.name = eye_side+'eye_pupil_rotation'
            xr_eye_frames.name = eye_side+'EYEvideo'
            trial_eye_data = xr.merge([eyedlc, eyeparams, xr_eye_frames])
            trial_eye_data.to_netcdf(os.path.join(config['save_path'], str(t_name+eye_side+'eye.nc')), engine='netcdf4', encoding={eye_side+'EYEvideo':{"zlib": True, "complevel": 9}})

        # analyze world views
        world_sides = []
        if 'WORLD' in config['camera_names']:
            print('tracking WORLD for ' + t_name)
            # filter the list of files for the current trial to get the world view of this side
            world_csv = list(filter(lambda a: ('WORLD') in a, trial_cam_csv))[0]
            world_avi = list(filter(lambda a: ('WORLD') in a, trial_cam_avi))[0]
            # make an xarray of timestamps without dlc points, since there aren't any for a world camera
            worlddlc, worldnames = h5_to_xr(pt_path=None, time_path=world_csv, view=('WORLD'))
            worlddlc.name = 'WORLDtimes'
            # make xarray of video frames
            xr_world_frames = format_frames(world_avi, config); xr_world_frames.name = 'WORLDvideo'
            trial_world_data = xr.merge([worlddlc, xr_world_frames])
            trial_world_data.to_netcdf(os.path.join(config['save_path'], str(t_name+'world.nc')), engine='netcdf4', encoding={'WORLDvideo':{"zlib": True, "complevel": 9}})

    print('done with ' + str(len(trial_paths)) + ' queued trials')

if __name__ == '__main__':
    main()
