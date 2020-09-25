"""
track_eye.py

utilities for tracking the pupil of the mouse and fitting an ellipse to the DeepLabCut points

last modified September 20, 2020
"""

# package imports
import pandas as pd
import numpy as np
from skimage import measure
import xarray as xr
import matplotlib.pyplot as plt
import os
import cv2
from skimage import measure
from itertools import product
from tqdm import tqdm
from numpy import *
import json
from math import e as e
from numpy.linalg import eig
import math

# module imports
from util.read_data import split_xyl

# finds the best fit to an ellipse for the given set of points in a single frame
# inputs should be x and y values of points aroudn pupil as numpy arrays
# outputs dictionary of ellipse parameters for a single frame
# adapted from /niell-lab-analysis/freely moving/fit_ellipse2.m
def fit_ellipse(x,y):

    orientation_tolerance = 1*np.exp(-3)
    
    # remove bias of the ellipse
    meanX = np.mean(x)
    meanY = np.mean(y)
    x = x - meanX
    y = y - meanY
    
    # estimation of the conic equation
    X = np.array([x**2, x*y, y**2, x, y])
    X = np.stack(X).T
    a = dot(np.sum(X, axis=0), linalg.pinv(np.matmul(X.T,X)))
    
    # extract parameters from the conic equation
    a, b, c, d, e = a[0], a[1], a[2], a[3], a[4]
    
    # eigen decomp
    Q = np.array([[a, b/2],[b/2, c]])
    eig_val, eig_vec = eig(Q)
    
    # get angle to long axis
    angle_to_x = np.arctan2(eig_vec[1,0], eig_vec[0,0])
    angle_from_x = angle_to_x
    

    orientation_rad = 0.5 * np.arctan2(b, (c-a))
    cos_phi = np.cos(orientation_rad)
    sin_phi = np.sin(orientation_rad)
    a, b, c, d, e = [a*cos_phi**2 - b*cos_phi*sin_phi + c*sin_phi**2,
                     0,
                     a*sin_phi**2 + b*cos_phi*sin_phi + c*cos_phi**2,
                     d*cos_phi - e*sin_phi,
                     d*sin_phi + e*cos_phi]
    meanX, meanY = [cos_phi*meanX - sin_phi*meanY,
                    sin_phi*meanX + cos_phi*meanY]
    
    # check if conc expression represents an ellipse
    test = a*c
    if test > 0:
        # make sure coefficients are positive as required
        if a<0:
            a, c, d, e = [-a, -c, -d, -e]
        
        # final ellipse parameters
        X0 = meanX - d/2/a
        Y0 = meanY - e/2/c
        F = 1 + (d**2)/(4*a) + (e**2)/(4*c)
        a = np.sqrt(F/a)
        b = np.sqrt(F/c)
        long_axis = 2*np.maximum(a,b)
        short_axis = 2*np.minimum(a,b)
        
        # rotate axes backwards to find center point of original tilted ellipse
        R = np.array([[cos_phi, sin_phi], [-sin_phi, cos_phi]])
        P_in = R @ np.array([[X0],[Y0]])
        X0_in = P_in[0][0]
        Y0_in = P_in[1][0]
        
        # organize parameters in dictionary to return
        # makes some final modifications to values here, maybe those should be done above for cleanliness
        ellipse_dict = {'X0':X0, 'Y0':Y0, 'F':F, 'a':a, 'b':b, 'long_axis':long_axis/2, 'short_axis':short_axis/2,
                        'angle_to_x':np.rad2deg(angle_to_x), 'angle_from_x':np.rad2deg(angle_from_x), 'cos_phi':cos_phi, 'sin_phi':sin_phi,
                        'X0_in':X0_in, 'Y0_in':Y0_in, 'phi':orientation_rad}
        
    else:
        # if the conic equation didn't return an ellipse, don't return any real values and fill the dictionary with NaNs
        ellipse_dict = {'X0':np.nan, 'Y0':np.nan, 'F':np.nan, 'a':np.nan, 'b':np.nan, 'long_axis':np.nan, 'short_axis':np.nan,
                        'angle_to_x':np.nan, 'angle_from_x':np.nan, 'cos_phi':np.nan, 'sin_phi':np.nan,
                        'X0_in':np.nan, 'Y0_in':np.nan, 'phi':np.nan}
        
    return ellipse_dict

# get the ellipse parameters from DeepLabCut points and save into an xarray
# equivilent to /niell-lab-analysis/freely moving/EyeCameraCalc1.m
def eye_tracking(eye_data, config):
    
    # names of th different points
    pt_names = list(eye_data['point_loc'].values)

    x_vals, y_vals, likeli_vals = split_xyl(pt_names, eye_data, config['lik_thresh'])
    likelihood = likeli_vals.values
    # drop tear
    # these points ought to be used, this will be addressed later
    if config['tear'] is True:
        x_vals = x_vals.iloc[:,:-2]
        y_vals = y_vals.iloc[:,:-2]

    # get bools of when a frame is usable with the right number of points above threshold
    usegood = np.sum(likelihood,0) >= config['num_ellipse_pts_needed']

    ellipse_params = np.empty([len(x_vals), 14])

    # step through each frame, fit an ellipse to points, and add ellipse parameters to array with data for all frames together
    for step in tqdm(range(0,len(x_vals))):
        if usegood[step] == True:
            e_t = fit_ellipse(x_vals.iloc[step].values, y_vals.iloc[step].values)
            ellipse_params[step] = [e_t['X0'], e_t['Y0'], e_t['F'], e_t['a'], e_t['b'],
                                    e_t['long_axis'], e_t['short_axis'], e_t['angle_to_x'], e_t['angle_from_x'],
                                    e_t['cos_phi'], e_t['sin_phi'], e_t['X0_in'], e_t['Y0_in'], e_t['phi']]
        elif usegood[step] == False:
            e_t = {'X0':np.nan, 'Y0':np.nan, 'F':np.nan, 'a':np.nan, 'b':np.nan, 'long_axis':np.nan, 'short_axis':np.nan,
                            'angle_to_x':np.nan, 'angle_from_x':np.nan, 'cos_phi':np.nan, 'sin_phi':np.nan,
                            'X0_in':np.nan, 'Y0_in':np.nan, 'phi':np.nan}
            ellipse_params[step] = [e_t['X0'], e_t['Y0'], e_t['F'], e_t['a'], e_t['b'],
                                    e_t['long_axis'] ,e_t['short_axis'], e_t['angle_to_x'], e_t['angle_from_x'],
                                    e_t['cos_phi'], e_t['sin_phi'], e_t['X0_in'], e_t['Y0_in'], e_t['phi']]

    # list of all places where the ellipse meets threshold
    R = np.linspace(0,2*np.pi, 100)
    list1 = np.where((ellipse_params[:,6] / ellipse_params[:,5]) < config['ell_thresh']) # short axis / long axis
    list2 = np.where((usegood == True) & ((ellipse_params[:,6] / ellipse_params[:,5]) < config['ell_thresh']))

    # find camera center
    A = np.vstack([np.cos(ellipse_params[list2,7]),np.sin(ellipse_params[list2,7])])
    b = np.expand_dims(np.diag(A.T@np.squeeze(ellipse_params[list2,11:13].T)),axis=1)
    cam_cent = np.linalg.inv(A@A.T)@A@b

    # ellipticity and scale
    ellipticity = (ellipse_params[list2,6] / ellipse_params[list2,5]).T
    scale = np.nansum(np.sqrt(1-(ellipticity)**2)*(np.linalg.norm(ellipse_params[list2,11:13]-cam_cent.T,axis=0)))/np.sum(1-(ellipticity)**2)

    # angles
    theta = np.arcsin((ellipse_params[:,11]-cam_cent[0])/scale)
    phi = np.arcsin((ellipse_params[:,12]-cam_cent[1])/np.cos(theta)/scale)

    # organize data to return as an xarray of most essential parameters
    # note: here, we subtract 45 degrees (in radians) from phi
    ellipse_df = pd.DataFrame({'theta':list(theta), 'phi':list(phi-0.7854), 'longaxis':list(ellipse_params[:,6]), 'shortaxis':list(ellipse_params[:,5]),
                               'X0':list(ellipse_params[:,11]), 'Y0':list(ellipse_params[:,12])})
    ellipse_params = ['theta', 'phi', 'longaxis', 'shortaxis', 'X0', 'Y0']
    ellipse_out = xr.DataArray(ellipse_df, coords=[('frame', range(0, len(ellipse_df))), ('ellipse_params', ellipse_params)], dims=['frame', 'ellipse_params'])
    ellipse_out.attrs['cam_center_x'] = cam_cent[0,0]
    ellipse_out.attrs['cam_center_y'] = cam_cent[1,0]

    return ellipse_out

# plot the ellipse and dlc points on the video frames
# then, save the video out as an .avi file
def plot_eye_vid(vid_path, dlc_data, ell_data, config, trial_name, eye_letter):

    # read topdown video in
    # setup the file to save out of this
    vidread = cv2.VideoCapture(vid_path)
    width = int(vidread.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidread.get(cv2.CAP_PROP_FRAME_HEIGHT))
    savepath = os.path.join(config['save_path'], (trial_name + '_' + eye_letter + 'EYE.avi'))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_vid = cv2.VideoWriter(savepath, fourcc, 60.0, (width, height))

    for frame_num in tqdm(range(0,int(vidread.get(cv2.CAP_PROP_FRAME_COUNT)))):
        ret, frame = vidread.read()

        if not ret:
            break

        if dlc_data is not None and ell_data is not None:
            # get current frame number to be displayed, so that it can be used to slice DLC data
            try:
                ell_data_thistime = ell_data.sel(frame=frame_num)
                # get out ellipse parameters and plot them on the video
                ellipse_axes = (int(ell_data_thistime.sel(ellipse_params='longaxis').values), int(ell_data_thistime.sel(ellipse_params='shortaxis').values))
                ellipse_phi = int(np.rad2deg(ell_data_thistime.sel(ellipse_params='phi').values))
                ellipse_cent = (int(ell_data_thistime.sel(ellipse_params='X0').values), int(ell_data_thistime.sel(ellipse_params='Y0').values))
                frame = cv2.ellipse(frame, ellipse_cent, ellipse_axes, ellipse_phi, 0, 360, (255,0,0), 2) # ellipse in blue
            except (ValueError, KeyError):
                pass

            # get out the DLC points and plot them on the video
            try:
                pts = dlc_data.sel(frame=frame_num)
                for k in range(0, len(pts), 3):
                    pt_cent = (int(pts.isel(point_loc=k).values), int(pts.isel(point_loc=k+1).values))
                    if pts.isel(point_loc=k+2).values < 0.90: # bad points in red
                        frame = cv2.circle(frame, pt_cent, 3, (0,0,255), -1)
                    elif pts.isel(point_loc=k+2).values >= 0.90: # good points in green
                        frame = cv2.circle(frame, pt_cent, 3, (0,255,0), -1)

            except (ValueError, KeyError):
                pass

        elif dlc_data is None or ell_data is None:
            pass

        out_vid.write(frame)

    out_vid.release()