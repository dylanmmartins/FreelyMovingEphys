"""Tracking mouse pupil from head-mounted eye-facing camera.

Notes:
    * Haven't used the cyclotorsion code in a long time, and it
      has since been reorganized/refactored. So, it may error

"""

import os
import json
import multiprocessing
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2
import astropy
import matplotlib.pyplot as plt
from scipy import stats, signal, optimize
from matplotlib.backends.backend_pdf import PdfPages

from fmEphys import utils

def fit_ellipse(x, y):
    """ Fit an ellipse to points labeled around the perimeter of pupil.

    Parameters
    --------
    x : np.array
        Positions of points along the x-axis for a single video frame.
    y : np.array
        Positions of labeled points along the y-axis for a single video frame.

    Returns
    --------
    ellipse_dict : dict
        Parameters of the ellipse.
        X0: center at the x-axis of the non-tilt ellipse
        Y0: center at the y-axis of the non-tilt ellipse
        a: radius of the x-axis of the non-tilt ellipse
        b: radius of the y-axis of the non-tilt ellipse
        long_axis: radius of the long axis of the ellipse
        short_axis: radius of the short axis of the ellipse
        angle_to_x: angle from long axis to horizontal plane
        angle_from_x: angle from horizontal plane to long axis
        X0_in: center at the x-axis of the tilted ellipse
        Y0_in: center at the y-axis of the tilted ellipse
        phi: tilt orientation of the ellipse in radians
    """

    # Remove bias
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    x = x - mean_x
    y = y - mean_y

    # Estimate conic equation
    X = np.array([x**2, x*y, y**2, x, y])
    X = np.stack(X).T
    a,b,c,d,e = np.dot(np.sum(X, axis=0),  \
                       np.linalg.pinv(np.matmul(X.T, X)))

    # Eigen decomp
    Q = np.array([[a, b/2], [b/2, c]])
    eig_val, eig_vec = np.linalg.eig(Q)

    # Get angle to long axis
    if eig_val[0] < eig_val[1]:
        angle_to_x = np.arctan2(eig_vec[1,0], eig_vec[0,0])
    else:
        angle_to_x = np.arctan2(eig_vec[1,1], eig_vec[0,1])

    angle_from_x = angle_to_x

    orientation_rad = 0.5 * np.arctan2(b, (c-a))
    cos_phi = np.cos(orientation_rad)
    sin_phi = np.sin(orientation_rad)

    a, b, c, d, e = [a*cos_phi**2 - b*cos_phi*sin_phi + c*sin_phi**2,
                    0,
                    a*sin_phi**2 + b*cos_phi*sin_phi + c*cos_phi**2,
                    d*cos_phi - e*sin_phi,
                    d*sin_phi + e*cos_phi]

    mean_x, mean_y = [cos_phi*mean_x - sin_phi*mean_y,
                    sin_phi*mean_x + cos_phi*mean_y]

    # Check if conc expression represents an ellipse
    test = a*c

    if test > 0:
        # Make sure coefficients are positive as required
        if a<0:
            a, c, d, e = [-a, -c, -d, -e]

        # Final ellipse parameters
        X0 = mean_x - d/2/a
        Y0 = mean_y - e/2/c
        F = 1 + (d**2)/(4*a) + (e**2)/(4*c)
        a = np.sqrt(F/a)
        b = np.sqrt(F/c)

        long_axis = 2*np.maximum(a,b)
        short_axis = 2*np.minimum(a,b)

        # Rotate axes backwards to find center point of the
        # original tilted ellipse
        R = np.array([[cos_phi, sin_phi], [-sin_phi, cos_phi]])
        P_in = R @ np.array([[X0],[Y0]])
        X0_in = P_in[0][0]
        Y0_in = P_in[1][0]

        ellipse_dict = {'X0':X0, 'Y0':Y0, 'F':F, 'a':a, 'b':b,
                        'long_axis':long_axis/2, 'short_axis':short_axis/2,
                        'angle_to_x':angle_to_x, 'angle_from_x':angle_from_x,
                        'cos_phi':cos_phi, 'sin_phi':sin_phi,
                        'X0_in':X0_in, 'Y0_in':Y0_in, 'phi':orientation_rad}
    else:
        # If the conic equation didn't return an ellipse,
        # don't return any real values and fill the
        # dictionary with NaNs
        dict_keys = ['X0','Y0','F','a','b','long_axis','short_axis',
                     'angle_to_x','angle_from_x','cos_phi','sin_phi',
                     'X0_in','Y0_in','phi']
        dict_vals = list(np.ones([len(dict_keys)]) * np.nan)

        ellipse_dict = dict(zip(dict_keys, dict_vals))
    
    return ellipse_dict

def calc_theta_phi(cfg, dlc_path=None):
    """
    cfg can be None and default options will be used
    """

    # Find files
    if dlc_path is None:
        dlc_path = utils.path.find('*{}*DLC*.h5'.format(cfg['dname']), cfg['rpath'])
        dlc_path = utils.path.most_recent(dlc_path)

    pdf_savepath = os.path.join(cfg['rpath'],
            '{}_{}_diagnostics.pdf'.format(cfg['rname'], cfg['cname']))
    pdf = PdfPages(pdf_savepath)
        
    # If this is a hf recording, read in existing fm
    # camera center, scale, etc.
    # It should run all fm recordings first, so it
    # will be possible to read in fm camera calibration
    # parameters for every hf recording
    if not cfg['overwrite_eyeparams'] and ('hf' in cfg['rname']):
        eyeparams_files = sorted(utils.path.find('*eyeparams.json', cfg['apath']))
        eyeparams_path = utils.path.most_recent(eyeparams_files)
        with open(eyeparams_path, 'r') as fp:
            eyeparams = json.load(fp)
    elif cfg['overwrite_params'] or ('fm' in cfg['rname']):
        eyeparams = None

    # Read in x/y positions and likelihood from DeepLabCut
    x_pos, y_pos, likeli = utils.file.read_dlc_positions(dlc_path, split_xyl=True)
    all_dlc_data = utils.file.read_dlc_positions(dlc_path, split_xyl=False)

    # subtract center of IR light reflection from all other pts
    if cfg['IRspot_is_labeled'] and cfg['do_eyecam_spotsub']:

        spot_xcent = np.mean(x_pos.iloc[:,-5:], 1)
        spot_ycent = np.mean(y_pos.iloc[:,-5:], 1)
        spot_likeli = likeli[:,-5:].copy()

        pupil_likeli = likeli[:,:-5].copy()

        x_pos = x_pos.iloc[:,:-5].subtract(spot_xcent, axis=0)
        y_pos = y_pos.iloc[:,:-5].subtract(spot_ycent, axis=0)

    # drop the IR spot points without doing their subtraction
    elif cfg['IRspot_is_labeled'] and not cfg['do_eyecam_spotsub']:

        x_pos = x_pos.iloc[:,:-5]
        y_pos = y_pos.iloc[:,:-5]
        pupil_likeli = likeli[:,:-5]

    # drop tear/outer eye points
    if cfg['canthus_is_labeled']:

        x_pos = x_pos.iloc[:,:-2]
        y_pos = y_pos.iloc[:,:-2]
        pupil_likeli = likeli[:,:-2]

    else:
        pupil_likeli = likeli.copy()

    # get bools of when a frame is usable with the right
    # number of points above threshold
    if cfg['do_eyecam_spotsub']:

        # if spot subtraction is being done, we should only include
        # frames where all five pts marked around the ir spot are
        # good (centroid would be off otherwise)
        pupil_count = np.sum(pupil_likeli >= cfg['likeli_thresh'], 1)
        spot_count = np.sum(spot_likeli >= cfg['likeli_thresh'], 1)

        use_pupil = (pupil_count >= cfg['ellipse_useN'])  \
                    & (spot_count >= cfg['reflection_useN'])
        cal_pupil = (pupil_count >= cfg['ellipse_calN'])  \
                    & (spot_count >= cfg['reflection_useN'])

        use_spot = (spot_count >= cfg['reflection_useN'])

    elif not cfg['do_eyecam_spotsub']:

        pupil_count = np.sum(pupil_likeli >= cfg['likeli_thresh'], 1)

        use_pupil = (pupil_count >= cfg['ellipse_useN'])
        cal_pupil = (pupil_count >= cfg['ellipse_calN'])

    # how well did eye track?
    fig, [[ax0,ax1],[ax2,ax3]] = plt.subplots(2,2, figsize=(11,8.5), dpi=300)

    ax0.plot(pupil_count[::10], color='k')
    ax0.set_title('{:.3}% good'.format(np.mean(use_pupil)*100))
    ax0.set_ylabel('# good pupil pts')
    ax0.set_xlabel('every 10th frame')

    ax1.hist(pupil_count, color='k', bins=9, range=(0,9), density=True)
    ax1.set_xlabel('# good pupil pts')
    ax1.set_ylabel('frac. frames')

    if cfg['do_eyecam_spotsub']:
        ax2.plot(spot_count[0::10])
        ax2.set_title('{:.3}% good'.format(np.mean(use_spot)*100))
        ax2.set_ylabel('# good reflec. pts')
        ax2.set_xlabel('every 10th frame')

        ax3.hist(spot_count, color='k', bins=9, range=(0,9), density=True)
        ax3.set_xlabel('# good reflec. pts')
        ax3.set_ylabel('frac. frames')
    else:
        ax2.axis('off')
        ax3.axis('off')

    fig.tight_layout()
    pdf.savefig()
    plt.close()

    # Threshold out pts more than a given distance
    # away from nanmean of that point
    std_thresh_x = np.empty(np.shape(x_pos))
    for f in range(np.size(x_pos, 1)):
        std_thresh_x[:,f] = (np.abs(np.nanmean(x_pos.iloc[:,f])  \
            - x_pos.iloc[:,f]) / cfg['pupil_pxl2cm']) > cfg['distance_thresh']
    
    std_thresh_y = np.empty(np.shape(y_pos))
    for f in range(np.size(y_pos, 1)):
        std_thresh_y[:,f] = (np.abs(np.nanmean(y_pos.iloc[:,f])  \
            - y_pos.iloc[:,f]) / cfg['pupil_pxl2cm']) > cfg['distance_thresh']
    
    std_thresh_x = np.nanmean(std_thresh_x, 1)
    std_thresh_y = np.nanmean(std_thresh_y, 1)

    x_pos[std_thresh_x > 0] = np.nan
    y_pos[std_thresh_y > 0] = np.nan
        
    # step through each frame, fit an ellipse to points, and
    # add ellipse parameters to array with data for all frames together
    cols = ['X0','Y0', # 0 1
            'F','a','b', # 2 3 4
            'long_axis','short_axis', # 5 6
            'angle_to_x','angle_from_x', # 7 8
            'cos_phi','sin_phi', # 9 10
            'X0_in','Y0_in', # 11 12
            'phi'] # 13
    ellipse = pd.DataFrame(np.zeros([len(use_pupil), len(cols)])*np.nan, columns=cols)

    linalg_errCount = 0
    for f in tqdm(range(len(use_pupil))):
        if use_pupil[f] is True:
            try:
                ef = fit_ellipse(x_pos.iloc[f].values,
                                  y_pos.iloc[f].values)
                for k,v in ef.items():
                    ellipse.at[f,k] = v

            except np.linalg.LinAlgError:
                linalg_errCount += 1
                pass

    print('ellipse fit encounted {} linalg errors (# frames={})'.format(  \
    linalg_errCount, len(use_pupil)))
        
    # list of all places where the ellipse meets threshold
    # (short axis / long axis) < thresh
    ellcal = np.where(cal_pupil & ((ellipse['short_axis'] / ellipse['long_axis'])  \
                    < cfg['ellipticity_thresh'])).flatten()
    
    # limit number of frames used for calibration
    # make a shorter version of the list
    if len(ellcal) > 50000:
        ellcal_s = sorted(np.random.choice(ellcal, size=50000, replace=False))
    else:
        ellcal_s = ellcal

    # find camera center
    cam_A = np.vstack([np.cos(ellipse.loc[ellcal_s, 'angle_to_x']),  \
                       np.sin(ellipse.loc[ellcal_s, 'angle_to_x'])])

    cam_b = np.expand_dims(np.diag(cam_A.T @ \
            np.squeeze(ellipse.loc[ellcal_s,['X0_in','Y0_in']].T)), axis=1)

    # but only use the camera center from this recording if
    # values were not read in from a json
    # in practice, this means hf recordings have their
    # cam center thrown out and use the fm values read in
    if eyeparams is None:
        cam_center = np.linalg.inv(cam_A @ cam_A.T) @ cam_A @ cam_b

    elif eyeparams is not None:
        cam_center = np.array([[float(eyeparams['cam_cent_x'])],  \
                               [float(eyeparams['cam_cent_y'])]])
        
    # ellipticity and scale
    ellipticity = (ellipse.loc[ellcal_s,'short_axis'] / ellipse.loc[ellcal_s,'long_axis']).T

    if eyeparams is None:
        try:
            scale = np.nansum(np.sqrt(1 - (ellipticity)**2) *  \
                    (np.linalg.norm(ellipse.loc[ellcal_s,['X0_in','Y0_in']] -  \
                    cam_center.T, axis=0))) / np.sum(1 - (ellipticity)**2 )
        
        except ValueError:
            # swap axis that linalg.norm is calculated over (from axis=0 to axis=1)
            # I don't remember why I did this try/except or how often
            # the code is going into this exception
            # I should debug this and see why it would ever happen...
            scale = np.nansum(np.sqrt(1 - (ellipticity)**2) *  \
                    (np.linalg.norm(ellipse.loc[ellcal_s,['X0_in','Y0_in']] -  \
                    cam_center.T, axis=1))) / np.sum(1 - (ellipticity)**2 )
    
    elif eyeparams is not None:
        scale = float(eyeparams['scale'])
        
    # horizontal angle (rad)
    theta = np.arcsin((ellipse['X0_in'] - cam_center[0]) / scale)

    # vertical angle (rad)
    phi = np.arcsin((ellipse['Y0_in'] - cam_center[1]) / np.cos(theta) / scale)

    # FIGURE: theta & phi
    fig, [[ax0,ax1,ax2],[ax3,ax4,ax5]] = plt.subplots(2,3, figsize=(11,8.5), dpi=300)
    
    ax0.plot(np.rad2deg(theta)[::10], color='k')
    ax0.set_ylabel('theta (deg)')
    ax0.set_xlabel('every 10th frame')

    ax1.plot(np.rad2deg(phi)[::10], color='k')
    ax1.set_ylabel('phi (deg)')
    ax1.set_xlabel('every 10th frame')

    ax2.plot(ellipticity[::10], color='k')
    ax2.set_ylabel('ellipticity')
    ax2.set_xlabel('every 10th frame')

    ax3.hist(np.rad2deg(theta), density=True, color='k')
    ax3.set_ylabel('theta (deg)')
    ax3.set_xlabel('frac. frames')

    ax4.hist(np.rad2deg(phi), density=True, color='k')
    ax4.set_ylabel('')
    ax4.set_xlabel('frac. frames')

    ax5.hist(ellipticity, density=True, color='k')
    ax5.set_ylabel('ellipticity')
    ax5.set_xlabel('frac. frames')

    fig.tight_layout()
    pdf.savefig(); plt.close()
        
    # Eye axes relative to center
    ds = 100
    w = ellipse['angle_to_x'].copy().to_numpy()
    x = ellipse['X0_in'].copy().to_numpy() # 11
    y = ellipse['Y0_in'].copy().to_numpy() # 12

    fig, [[ax0,ax1],[ax3,ax4]] = plt.subplots(2,2, figsize=(11,8.5), dpi=300)
    
    # Position for frames
    for f in ellcal[::ds]:
        ax0.plot(
            (x[f] + [-5 * np.cos(w[f]), 5 * np.cos(w[f])]),
            (y[f] + [-5 * np.sin(w[f]), 5 * np.sin(w[f])]),
            '.', markersize=1.5)
    # Camera center
    ax0.plot(cam_center[0], cam_center[1], 'r*')
    ax0.set_title('eye axes relative to center')
            
    # Check calibration
    xvals = np.linalg.norm(ellipse.loc[cal_pupil, ['X0_in','Y0_in']].copy().to_numpy().T  \
            - cam_center, axis=0)
    yvals = scale * np.sqrt(1 - (ellipse.loc[cal_pupil,'short_axis'].copy().to_numpy()  \
            / ellipse.loc[cal_pupil,'long_axis'].copy().to_numpy()) **2 )
    tmp_mask = ~np.isnan(xvals) & ~np.isnan(yvals)

    slope, _, r_value, _, _ = stats.linregress(xvals[tmp_mask], yvals[tmp_mask].T)
    
    # Scale and center
    ax1.plot(xvals[::ds], yvals[::ds], '.', markersize=1)
    ax1.plot(np.linspace(0,np.max(xvals[::ds])), np.linspace(0,np.max(yvals[::ds])), 'r')
    ax1.set_title('scale={:.3} r={:.3} m={:.3}'.format(scale, r_value, slope))
    ax1.set_xlabel('pupil camera dist')
    ax1.set_ylabel('scale * ellipticity')

    # Calibration of camera center
    delta = (cam_center - ellipse[['X0_in','Y0_in']].copy().to_numpy().T)
    show_cal = cal_pupil[::ds]
    show_use = use_pupil[::ds]
    ang2x = ellipse['angle_to_x'].copy().to_numpy()
    # Plot pts used for calibration
    ax2.plot(
        np.linalg.norm(delta[:,show_cal], 2, axis=0),  \
        ((delta[0,show_cal].T * np.cos(ang2x[show_cal]))  \
        + (delta[1,show_cal].T * np.sin(ang2x[show_cal])))  \
        / np.linalg.norm(delta[:,show_cal], 2, axis=0).T,  \
        'r.', markersize=1, label='cal')
    # Plot all pts
    ax2.plot(
        np.linalg.norm(delta[:,show_use], 2, axis=0),  \
        ((delta[0,show_use].T * np.cos(ang2x[show_use]))  \
        + (delta[1,show_use].T * np.sin(ang2x[show_use])))  \
        / np.linalg.norm(delta[:, show_use], 2, axis=0).T,  \
        'k.', markersize=1, label='all')
    ax2.set_title('camera center calibration')
    ax2.set_ylabel('abs([PC-EC]).[cosw;sinw]')
    ax2.set_xlabel('abs(PC-EC)')

    ax3.axis('off')

    fig.tight_layout()
    pdf.savefig()
    plt.close()

    pdf.close() # close and save the pdf

    # save out camera center and scale as np array
    # (but only if this is a freely moving recording)
    if 'fm' in cfg['rname']:
        caldict = {
            'cam_cent_x': float(cam_center[0]),
            'cam_cent_y': float(cam_center[1]),
            'scale': float(scale),
            'regression_r': float(r_value),
            'regression_m': float(slope)}
        caldict_savepath = os.path.join(cfg['rpath'],
                '{}_{}_eyeparams.json'.format(cfg['rfname'], cfg['dname']))

        with open(caldict_savepath, 'w') as f:
            json.dump(caldict, f)

    # save out the data
    ell_dict = {
        'theta': list(theta),
        'phi': list(phi),
        'longaxis': list(ellipse['long_axis'].values),
        'shortaxis': list(ellipse['short_axis'].values),
        'X0': list(ellipse['X0_in'].values),
        'Y0': list(ellipse['Y0_in'].values),
        'ellipse_rotation': list(ellipse['angle_to_x'].values),
        'camera_center': cam_center[:,0] # contains X and Y in zeroth dim
    }
    pos_dict = all_dlc_data.to_dict('list')
    savedata = {**ell_dict, **pos_dict}

    return savedata

def preprocess_pupil(cfg):
    """Main preprocessing.
    """

    # Read in timestamps
    time_path = utils.path.find('{}*{}.avi'.format(cfg['rfname'],  \
        cfg['dname']), cfg['rpath'])
    time_path = utils.path.most_recent(time_path)
    eyeT = utils.time.read_time(time_path)

    # read in the .avi and load it into a numpy array
    vid_path = utils.path.find('{}*{}deinter.avi'.format(cfg['rfname'],  \
        cfg['dname']), cfg['rpath'])
    vid_path = utils.path.most_recent(vid_path)
    vid = utils.video.avi_to_arr(vid_path)

    # calc theta and phi
    pos_ell_dict = calc_theta_phi(cfg)

    addtl_dict = {
        'eyeT': eyeT,
        'video': vid
    }

    savedata = {**pos_ell_dict, **addtl_dict}
    savepath = os.path.join(cfg['rpath'],  \
        '{}_{}_pupil.h5'.format(cfg['rfname'], cfg['dname']))

    utils.file.write_h5(savepath, savedata)


def sigmoid_curve(xval, a, b, c):
    """Sigmoid curve function."""

    return a + (b-a) / (1 + 10**( (c - xval) * 2))


def sigmoid_fit(d):
    """ Fit sigmoid.
    popt: fit
    ci: confidence interval
    """
    try:
        popt, pcov = optimize.curve_fit(sigmoid_curve,
                        xdata=range(1,len(d)+1),
                        ydata=d,
                        p0=[100.0,200.0,len(d)/2],
                        method='lm',
                        xtol=10**-3,
                        ftol=10**-3)
        ci = np.sqrt(np.diagonal(pcov))

    except RuntimeError:
        popt = np.nan*np.zeros(4)
        ci = np.nan*np.zeros(4)

    return (popt, ci)


def get_torsion_from_ridges(cfg, ell_dict, vidpath=None):
    """ Get torsion (omega) from rotation of ridges along the edge of the pupil.
    """

    if vidpath is None:
        vidpath = utils.path.find('{}*{}deinter.avi'.format(cfg['rfname'],  \
            cfg['dname']), cfg['rpath'])
        vidpath = utils.path.most_recent(vidpath)
    
    pdf_savepath = os.path.join(cfg['rpath'],
            '{}_{}_cyclotorsion.pdf'.format(cfg['rname'], cfg['cname']))
    pdf = PdfPages(pdf_savepath)
    
    # Set up range of degrees in radians
    rad_range = np.deg2rad(np.arange(360))

    # Get the ellipse parameters for this trial from the time-interpolated xarray
    eye_longaxis = ell_dict['longaxis']
    eye_shortaxis = ell_dict['shortaxis']
    eye_centX = ell_dict['X0']
    eye_centY = ell_dict['Y0']

    # Set up for the read-in video
    eyevid = cv2.VideoCapture(self.video_path)
    # this can be changed to a small number of frames for testing
    totalF = int(eyevid.get(cv2.CAP_PROP_FRAME_COUNT))

    set_size = (int(eyevid.get(cv2.CAP_PROP_FRAME_WIDTH)),  \
                int(eyevid.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # set up for the multiprocessing for sigmoid fit
    n_proc = multiprocessing.cpu_count()
    print('Found {} as CPU count for multiprocessing'.format(n_proc))
    pool = multiprocessing.Pool(processes=n_proc)

    range_r = 10

    print('Calculating pupil cross-section and fitting sigmoid (slow)')
    errCount = 0

    rfit_out = np.zeros(totalF, 360)
    rfit_conv_out = np.zeros(totalF, 360)

    for f in tqdm(np.arange(totalF)):
        try:
            # Read frame
            ret, img = eyevid.read()
            if not ret:
                break

            # Convert to grey image
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Mean radius
            meanr = 0.5 * (eye_longaxis[f] + eye_shortaxis[f])
            # Range of values over mean radius (meanr)
            r = range(int(meanr - range_r), int(meanr + range_r))
            # Empty array that the calculated edge of the pupil
            # will be put into
            pupil_edge = np.zeros([360, len(r)])

            # Get cross-section of pupil at each angle 1-360 and
            # fit to sigmoid
            rad_range = np.deg2rad(np.arange(360))
            
            for i in range(len(r)):
                pupil_edge[:,i] = img[(  \
                    (eye_centY[f] + r[i] * (np.sin(rad_range))).astype(int),  \
                    (eye_centY[f] + r[i] * (np.cos(rad_range))).astype(int)  \
                    )]

            d = pupil_edge[:,:]

            # Apply sigmoid fit with multiprocessing
            param_mp = [pool.apply_async(sigmoid_fit, args=(d[n,:],)) for n in range(360)]
            params_output = [result.get() for result in param_mp]

            # Unpack outputs of sigmoid fit
            params = []; ci = []
            for vals in params_output:
                params.append(vals[0])
                ci.append(vals[1])
            params = np.stack(params)
            ci = np.stack(ci)

            # Extract radius variable from parameters
            rfit = params[:,2] - 1

            # If confidence interval in estimate is > fit_thresh pix, set to to NaN
            ci_temp = (ci[:,0] > 5) | (ci[:,1] > 5)  | (ci[:,2]>0.75)
            rfit[ci_temp] = np.nan

            # Remove if luminance goes the wrong way (e.g. from reflectance)
            rfit[(params[:,1] - params[:,0]) < 10] = np.nan
            rfit[params[:,1] > 250] = np.nan

            try:
                # Median filter
                rfit_filt = utils.filter.nanmedfilt(rfit, 5)

                # Subtract baseline because our points aren't perfectly centered on ellipse
                filtsize = 31
                rfit_conv = rfit_filt - astropy.convolution.convolve(rfit_filt,  \
                                np.ones(filtsize)/filtsize, boundary='wrap')

            except ValueError as e:
                # In case every value in rfit is NaN
                rfit = np.nan*np.zeros(360)
                rfit_conv = np.nan*np.zeros(360)
                
        except (KeyError, ValueError):
            errCount += 1
            rfit = np.nan*np.zeros(360)
            rfit_conv = np.nan*np.zeros(360)

        # Get rid of outlier points
        rfit_conv[np.abs(rfit_conv) > 1.5] = np.nan

        # Save this out
        rfit_out[f,:] = rfit
        rfit_conv_out[f,:] = rfit_conv

    ##############

    # Save out pupil edge data
    edgedata_dict = {
        'rfit': rfit,
        'rfit_conv': rfit_conv
    }

    # Threshold out any frames with large or small rfit_conv distributions
    for frame in range(0,np.size(rfit_conv_xr,0)):
        if np.min(rfit_conv_xr[frame,:]) < -10 or np.max(rfit_conv_xr[frame,:]) > 10:
            rfit_conv_xr[frame,:] = np.nan

    # correlation across first minute of recording
    timepoint_corr_rfit = pd.DataFrame(rfit_conv_xr.isel(frame=range(0,3600)).values).T.corr()

    # plot the correlation matrix of rfit over all timepoints
    plt.figure()
    fig, ax = plt.subplots()
    im = ax.imshow(timepoint_corr_rfit)
    ax.set_title('correlation of radius fit during first min. of recording')
    fig.colorbar(im, ax=ax)
    pdf.savefig(); plt.close()

    n = np.size(rfit_conv_xr.values, 0)
    pupil_update = rfit_conv_xr.values.copy()
    total_shift = np.zeros(n); peak = np.zeros(n)
    c = total_shift.copy()
    template = np.nanmean(rfit_conv_xr.values, 0)

    ### ***

        # calculate mean as template
        try:
            template_rfitconv_cc, template_rfit_cc_lags = nanxcorr(rfit_conv_xr[7].values, template, 30)
            template_nanxcorr = True
        except ZeroDivisionError:
            template_nanxcorr = False

        plt.figure()
        plt.plot(template)
        plt.title('mean as template')
        pdf.savefig(); plt.close()

        if template_nanxcorr is True:
            plt.figure()
            plt.plot(template_rfitconv_cc)
            plt.title('rfit_conv template cross correlation')
            pdf.savefig(); plt.close()

        # xcorr of two random timepoints
        try:
            t0 = np.random.random_integers(0,totalF-1); t1 = np.random.random_integers(0,totalF-1)
            rfit2times_cc, rfit2times_lags = nanxcorr(rfit_conv_xr.isel(frame=t0).values, rfit_conv_xr.isel(frame=t1).values, 10)
            rand_frames = True
        except ZeroDivisionError:
            rand_frames = False
        if rand_frames is True:
            plt.figure()
            plt.plot(rfit2times_cc, 'b-')
            plt.title('nanxcorr of frames ' + str(t0) + ' and ' + str(t1))
            pdf.savefig(); plt.close()

        num_rfit_samples_to_plot = 100
        ind2plot_rfit = sorted(np.random.randint(0,totalF-1,num_rfit_samples_to_plot))

        # iterative fit to alignment
        # start with mean as template
        # on each iteration, shift individual frames to max xcorr with template
        # then recalculate mean template
        print('doing iterative fit for alignment of each frame')
        for rep in tqdm(range(0,12)): # twelve iterations
            # for each frame, get correlation, and shift
            for frame_num in range(0,n): # do all frames
                try:
                    xc, lags = nanxcorr(template, pupil_update[frame_num,:], 20)
                    c[frame_num] = np.amax(xc) # value of max
                    peaklag = np.argmax(xc) # position of max
                    peak[frame_num] = lags[peaklag]
                    total_shift[frame_num] = total_shift[frame_num] + peak[frame_num]
                    pupil_update[frame_num,:] = np.roll(pupil_update[frame_num,:], int(peak[frame_num]))
                except ZeroDivisionError:
                    total_shift[frame_num] = np.nan
                    pupil_update[frame_num,:] = np.nan

            template = np.nanmean(pupil_update, axis=0) # update template

            # plot template with pupil_update for each iteration of fit
            plt.figure()
            plt.title('pupil_update of rep='+str(rep)+' in iterative fit')
            plt.plot(pupil_update[ind2plot_rfit,:].T, alpha=0.2)
            plt.plot(template, 'k--', alpha=0.8)
            pdf.savefig(); plt.close()

            # histogram of correlations
            plt.figure()
            plt.title('correlations of rep='+str(rep)+' in iterative fit')
            plt.hist(c[c>0], bins=300) # gets rid of NaNs in plot
            pdf.savefig(); plt.close()

        win = 5
        shift_nan = -total_shift
        shift_nan[c < 0.35] = np.nan
        shift_nan = shift_nan - np.nanmedian(shift_nan)
        shift_nan[shift_nan >= 20] = np.nan; shift_nan[shift_nan <= -20] = np.nan # get rid of very large shifts
        shift_smooth = signal.medfilt(shift_nan,3)  # median filt to get rid of outliers
        shift_smooth = astropy.convolution.convolve(shift_nan, np.ones(win)/win)  # convolve to smooth and fill in nans
        shift_smooth = shift_smooth - np.nanmedian(shift_smooth)

        plt.figure()
        plt.plot(shift_nan)
        plt.title('shift nan')
        pdf.savefig(); plt.close()

        plt.figure()
        plt.plot(shift_smooth)
        plt.title('shift smooth')
        pdf.savefig(); plt.close()

        plt.figure()
        plt.plot(shift_smooth[:3600])
        plt.title('shift smooth for first 1min of recording')
        pdf.savefig(); plt.close()

        plt.figure()
        plt.plot(shift_smooth, linewidth = 4, label = 'shift_smooth')
        plt.plot(-total_shift,linewidth=1, alpha = 0.5, label='raw total_shift')
        plt.legend()
        plt.title('shift_smooth and raw total shift')
        pdf.savefig(); plt.close()

        plt.figure()
        plt.plot(rfit_xr.isel(frame=ind2plot_rfit).T, alpha=0.2)
        plt.plot(np.nanmean(rfit_xr.T,1), 'b--', alpha=0.8)
        plt.title('rfit for 100 random frames')
        pdf.savefig(); plt.close()

        plt.figure()
        plt.plot(rfit_conv_xr.isel(frame=ind2plot_rfit).T, alpha=0.2)
        plt.plot(np.nanmean(rfit_conv_xr.T,1), 'b--', alpha=0.8)
        plt.title('rfit_conv for 100 random frames')
        pdf.savefig(); plt.close()

        # plot of 5 random frames' rfit_conv
        plt.figure()
        fig, axs = plt.subplots(5,1)
        axs = axs.ravel()
        for i in range(0,5):
            rand_num = np.random.randint(0,totalF-1)
            axs[i].plot(rfit_conv_xr.isel(frame=rand_num))
            axs[i].set_title(('rfit conv; frame ' + str(rand_num)))
        pdf.savefig()
        plt.close()

        shift_smooth1 = xr.DataArray(shift_smooth, dims=['frame'])

        if self.config['internals']['diagnostic_preprocessing_videos'] is True:
            eyevid = cv2.VideoCapture(self.video_path)
            vidsavepath = os.path.join(self.recording_path,str(self.recording_name+'_pupil_rotation_rep'+str(rep)+'_'+self.camname+'.avi'))
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            vidout = cv2.VideoWriter(vidsavepath, fourcc, 60.0, (int(eyevid.get(cv2.CAP_PROP_FRAME_WIDTH))*2, int(eyevid.get(cv2.CAP_PROP_FRAME_HEIGHT))))
            
            if self.config['internals']['video_frames_to_save'] > int(eyevid.get(cv2.CAP_PROP_FRAME_COUNT)):
                num_save_frames = int(eyevid.get(cv2.CAP_PROP_FRAME_COUNT))
            else:
                num_save_frames = self.config['internals']['video_frames_to_save']

            print('plotting pupil rotation on eye video')
            for step in tqdm(range(num_save_frames)):
                eye_ret, eye_frame = eyevid.read()
                eye_frame0 = eye_frame.copy()
                if not eye_ret:
                    break

                # get ellipse parameters for this time
                current_longaxis = eye_longaxis.sel(frame=step).values
                current_shortaxis = eye_shortaxis.sel(frame=step).values
                current_centX = eye_centX.sel(frame=step).values
                current_centY = eye_centY.sel(frame=step).values

                # plot the ellipse edge
                rmin = 0.5 * (current_longaxis + current_shortaxis) - ranger
                for deg_th in range(0,360):
                    rad_th = rad_range[deg_th]
                    edge_x = np.round(current_centX+(rmin+rfit_xr.isel(frame=step,deg=deg_th).values)*np.cos(rad_th))
                    edge_y = np.round(current_centY+(rmin+rfit_xr.isel(frame=step,deg=deg_th).values)*np.sin(rad_th))
                    if pd.isnull(edge_x) is False and pd.isnull(edge_y) is False:
                        eye_frame1 = cv2.circle(eye_frame, (int(edge_x),int(edge_y)), 1, (235,52,155), thickness=-1)

                # plot the rotation of the eye as a vertical line made up of many circles
                for d in np.linspace(-0.5,0.5,100):
                    rot_x = np.round(current_centX + d*(np.rad2deg(np.cos(np.deg2rad(shift_smooth1.isel(frame=step).values)))))
                    rot_y = np.round(current_centY + d*(np.rad2deg(np.sin(np.deg2rad(shift_smooth1.isel(frame=step).values)))))
                    if pd.isnull(rot_x) is False and pd.isnull(rot_y) is False:
                        eye_frame1 = cv2.circle(eye_frame1, (int(rot_x),int(rot_y)),1,(255,255,255),thickness=-1)

                # plot the center of the eye on the frame as a larger dot than the others
                if pd.isnull(current_centX) is False and pd.isnull(current_centY) is False:
                    eye_frame1 = cv2.circle(eye_frame1, (int(current_centX),int(current_centY)),3,(0,255,0),thickness=-1)

                frame_out = np.concatenate([eye_frame0, eye_frame1], axis=1)

                vidout.write(frame_out)

            vidout.release()

        shift = xr.DataArray(pd.DataFrame(shift_smooth), dims=['frame','shift'])
        print('key/value error count during sigmoid fit: ' + str(key_error_count))

        # plotting omega on some random frames to be saved into the pdf
        eyevid = cv2.VideoCapture(self.video_path)
        rand_frame_nums = list(np.random.randint(0,int(eyevid.get(cv2.CAP_PROP_FRAME_COUNT)), size=20))
        
        for step in rand_frame_nums:
            eyevid.set(cv2.CAP_PROP_POS_FRAMES, step)
            eye_ret, eye_frame = eyevid.read()
            if not eye_ret:
                break
            plt.subplots(2,2)
            plt.subplot(221)
            plt.imshow(eye_frame.astype(np.uint8), cmap='gray')

            # get ellipse parameters for this time
            current_longaxis = eye_longaxis.sel(frame=step).values
            current_shortaxis = eye_shortaxis.sel(frame=step).values
            current_centX = eye_centX.sel(frame=step).values
            current_centY = eye_centY.sel(frame=step).values
            
            # plot the ellipse edge
            rmin = 0.5 * (current_longaxis + current_shortaxis) - ranger
            plt.subplot(222)
            plt.imshow(eye_frame.astype(np.uint8), cmap='gray')
            for deg_th in range(0,360):
                rad_th = rad_range[deg_th]
                edge_x = np.round(current_centX+(rmin+rfit_xr.isel(frame=step,deg=deg_th).values)*np.cos(rad_th))
                edge_y = np.round(current_centY+(rmin+rfit_xr.isel(frame=step,deg=deg_th).values)*np.sin(rad_th))
                if pd.isnull(edge_x) is False and pd.isnull(edge_y) is False:
                    plt.plot(edge_x, edge_y, color='orange', marker='.',markersize=1,alpha=0.1)
            
            # plot the rotation of the eye as a vertical line made up of many circles
            plt.subplot(223)
            plt.imshow(eye_frame.astype(np.uint8), cmap='gray')
            for d in np.linspace(-0.5,0.5,100):
                rot_x = np.round(current_centX + d*(np.rad2deg(np.cos(np.deg2rad(shift_smooth1.isel(frame=step).values)))))
                rot_y = np.round(current_centY + d*(np.rad2deg(np.sin(np.deg2rad(shift_smooth1.isel(frame=step).values)))))
                if pd.isnull(rot_x) is False and pd.isnull(rot_y) is False:
                    plt.plot(rot_x, rot_y, color='white',marker='.',markersize=1,alpha=0.1)

            plt.subplot(223)
            plt.imshow(eye_frame.astype(np.uint8), cmap='gray')
            # plot the center of the eye on the frame as a larger dot than the others
            if pd.isnull(current_centX) is False and pd.isnull(current_centY) is False:
                plt.plot(int(current_centX),int(current_centY), color='blue', marker='o')

            plt.subplot(224)
            plt.imshow(eye_frame.astype(np.uint8), cmap='gray')
            for deg_th in range(0,360):
                rad_th = rad_range[deg_th]
                edge_x = np.round(current_centX+(rmin+rfit_xr.isel(frame=step,deg=deg_th).values)*np.cos(rad_th))
                edge_y = np.round(current_centY+(rmin+rfit_xr.isel(frame=step,deg=deg_th).values)*np.sin(rad_th))
                if pd.isnull(edge_x) is False and pd.isnull(edge_y) is False:
                    plt.plot(edge_x, edge_y, color='orange', marker='.',markersize=1,alpha=0.1)
            for d in np.linspace(-0.5,0.5,100):
                rot_x = np.round(current_centX + d*(np.rad2deg(np.cos(np.deg2rad(shift_smooth1.isel(frame=step).values)))))
                rot_y = np.round(current_centY + d*(np.rad2deg(np.sin(np.deg2rad(shift_smooth1.isel(frame=step).values)))))
                if pd.isnull(rot_x) is False and pd.isnull(rot_y) is False:
                    plt.plot(rot_x, rot_y, color='white',marker='.',markersize=1,alpha=0.1)
            # plot the center of the eye on the frame as a larger dot than the others
            if pd.isnull(current_centX) is False and pd.isnull(current_centY) is False:
                plt.plot(int(current_centX),int(current_centY), color='blue', marker='o')

            pdf.savefig()
            plt.close()

        pdf.close()

        self.shift = shift
        self.rfit = rfit_xr
        self.rfit_conv = rfit_conv_xr

    # def get_torsion_from_markers(self):

    def save_params(self):
        self.xrpts.name = self.camname+'_pts'
        self.xrframes.name = self.camname+'_video'
        self.ellipse_params.name = self.camname+'_ellipse_params'
        merged_data = [self.xrpts, self.ellipse_params, self.xrframes]

        if self.config['internals']['get_torsion_from_ridges']:
            self.rfit.name = self.camname+'_pupil_radius'
            self.shift.name = self.camname+'_omega'
            self.rfit_conv.name = self.camname+'_conv_pupil_radius'
            merged_data = merged_data + [self.rfit, self.shift, self.rfit_conv]
        if self.config['internals']['get_torsion_from_markers']:
            print('Torsion from markers not implemented.')
            sys.exit()

        self.safe_merge(merged_data)
        self.data.to_netcdf(os.path.join(self.recording_path,str(self.recording_name+'_'+self.camname+'.nc')),
                    engine='netcdf4', encoding={self.camname+'_video':{"zlib": True, "complevel": 4}})

    def process(self):
        if self.config['main']['deinterlace'] and not self.config['internals']['flip_headcams']['run']:
            self.deinterlace()
        elif not self.config['main']['deinterlace'] and self.config['internals']['flip_headcams']['run']:
            self.flip_headcams()


        if self.config['internals']['apply_gamma_to_eyecam']:
            self.auto_contrast()

        if self.config['main']['pose_estimation']:
            self.pose_estimation()
