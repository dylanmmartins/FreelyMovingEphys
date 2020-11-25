"""
analyze_jump.py

jump tracking utilities

Oct. 26, 2020
"""

# package imports
import xarray as xr
import cv2
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

# module imports
from util.read_data import nanxcorr

# get cross-correlation
def jump_cc(global_data_path, global_save_path, trial_name, REye_ds, LEye_ds, top_ds, side_ds):
    # open pdf file to save plots in
    pdf = PdfPages(os.path.join(config['data_path'], (config['recording_name'] + '_jump_cc.pdf')))
    # organize data
    REye_now = REye_ds.REYE_ellipse_params
    LEye_now = LEye_ds.LEYE_ellipse_params
    head_theta = side_ds.SIDE_theta

    RTheta = (REye_now.sel(ellipse_params='theta') - np.nanmedian(REye_now.sel(ellipse_params='theta')))
    RPhi = (REye_now.sel(ellipse_params='phi') - np.nanmedian(REye_now.sel(ellipse_params='phi')))
    LTheta = (LEye_now.sel(ellipse_params='theta') -  np.nanmedian(LEye_now.sel(ellipse_params='theta')))
    LPhi = (LEye_now.sel(ellipse_params='phi') - np.nanmedian(LEye_now.sel(ellipse_params='phi')))

    # zero-center head theta, and get rid of wrap-around effect (mod 360)
    th = head_theta * 180 / np.pi; th = (th + 360) % 360
    th = th - np.nanmean(th); th = -th

    # eye divergence (theta)
    div = 0.5 * (RTheta - LTheta)
    # gaze (mean theta of eyes)
    gaze_th = (RTheta + LTheta) * 0.5
    # gaze (mean phi of eyes)
    gaze_phi = (RPhi + LPhi) * 0.5

    # calculate xcorrs
    th_gaze, lags = nanxcorr(th.values, gaze_th.values, 30)
    th_div, lags = nanxcorr(th.values, div.values, 30)
    th_phi, lags = nanxcorr(th.values, gaze_phi.values, 30)

    # plots
    plt.figure()
    plt.title(config['recording_name'])
    plt.ylabel('deg'); plt.xlabel('frames')
    plt.legend(['head_theta', 'eye_theta','eye_divergence','eye_phi'])
    plt.plot(th); plt.plot(gaze_th); plt.plot(div); plt.plot(gaze_phi)
    pdf.savefig()
    plt.close()

    plt.figure()
    plt.title('head theta xcorr')
    plt.plot(lags, th_gaze); plt.plot(lags, th_div); plt.plot(lags, th_phi)
    plt.legend(['gaze', 'div', 'phi'])
    pdf.savefig()
    plt.close()

    plt.figure()
    plt.ylabel('eye div deg'); plt.xlabel('head th deg')
    plt.plot([-40,40],[40,-40], 'r:')
    plt.xlim([-40,40]); plt.ylim([-40,40])
    plt.scatter(th, div)
    pdf.savefig()
    plt.close()

    plt.figure()
    plt.ylabel('eye phi deg'); plt.xlabel('head th deg')
    plt.plot([-40,40],[-40,40], 'r:')
    plt.xlim([-40,40]); plt.ylim([-40,40])
    plt.scatter(th, gaze_phi)
    pdf.savefig()
    plt.close()

    # # plot pooled data
    # # head theta, phi
    # plt.figure()
    # plt.plot(all_theta, all_phi, '.')
    # plt.xlabel('head theta'); plt.ylabel('phi')
    # plt.xlim([-60,60]); plt.ylim([-60,60])
    # pdf.savefig()
    # plt.close()
    # # head theta, eye theta divergence
    # plt.figure()
    # plt.plot(all_theta, all_div, '.')
    # plt.xlabel('head theta'); plt.ylabel('eye theta div')
    # plt.xlim([-60,60]); plt.ylim([-60,60])
    # pdf.savefig()
    # plt.close()
    # # xcorr with head angle
    # plt.figure()
    # plt.errorbar(lags, np.nanmean(all_th_gaze), np.std(all_th_gaze)/np.sqrt(np.size(all_th_gaze)))
    # plt.errorbar(lags, np.nanmean(all_th_div), np.std(all_th_div)/np.sqrt(np.size(all_th_div)))
    # plt.errorbar(lags, np.nanmean(all_th_phi), np.std(all_th_phi)/np.sqrt(np.size(all_th_phi)))
    # plt.ylim([-1,1]); plt.ylabel('correlation'); plt.title('xcorr with head angle')
    # plt.legend(['mean theta', 'mean theta divergence', 'mean phi'])
    # pdf.savefig()
    # plt.close()

    pdf.close()

# create movies of pursuit with eye positions
def jump_gaze_trace(REye, LEye, TOP, SIDE, Svid, config):
    
    REye_params = REye.REYE_ellipse_params
    LEye_params = LEye.LEYE_ellipse_params
    Side_pts = SIDE.SIDE_pts
    Side_params = SIDE.SIDE_theta

    savepath = str(savepath) + '/' + str(trial_name) + '_side_gaze_trace.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_vid = cv2.VideoWriter(savepath, fourcc, 20.0, (width, height))

    sidecap = cv2.VideoCapture(Svid) #.set(cv2.CAP_PROP_POS_FRAMES, int(side_startframe))

    # find the first shared frame for the four video feeds and play them starting at that shared frame
    td_startframe, td_endframe, left_startframe, left_endframe, right_startframe, right_endframe, side_startframe, side_endframe, first_real_time, last_real_time = find_start_end(TOP, LEye, REye, SIDE)

    savepath = os.path.join(config['data_path'], (config['recording_name'] + '_side_gaze_trace.avi'))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vid_out = cv2.VideoWriter(savepath, fourcc, 60.0, (width, height))

    # interpolate time
    REye_interp = REye_params.interp_like(other=TOP, method='linear')
    LEye_interp = LEye_params.interp_like(other=TOP, method='linear')
    SIDE_par_interp = np.deg2rad(Side_params.interp_like(other=TOP, method='linear')) # plus, convert to radians
    SIDE_pts_interp = Side_pts.interp_like(other=TOP, method='linear')

    for frame_num in tqdm(range(0,int(sidecap.get(cv2.CAP_PROP_FRAME_COUNT)))):
        # read in videos
        SIDE_ret, SIDE_frame = sidevid.read()
        TOP_ret, TOP_frame = sidevid.read()
        REye_ret, REye_frame = sidevid.read()
        LEye_ret, LEye_frame = sidevid.read()

        if not SIDE_ret:
            break
        if not TOP_ret:
            break
        if not REye_ret:
            break
        if not LEye_ret:
            break

        # get current ellipse parameters
        REye_now = REye_interp.sel(frame=frame_num)
        LEye_now = LEye_interp.sel(frame=frame_num)
        SIDE_par_now = SIDE_par_interp.sel(frame=frame_num)
        SIDE_pts_now = SIDE_pts_interp.sel(frame=frame_num)

        # # scale
        # REye_now = REye_now * SIDE_par_now.sel(head_param='scaleR') / 50
        # LEye_now = LEye_now * SIDE_par_now.sel(head_param='scaleL') / 50

        # split apart parameters
        RTheta = REye_now.sel(ellipse_params='theta').values
        RPhi = REye_now.sel(ellipse_params='phi').values
        LTheta = LEye_now.sel(ellipse_params='theta').values
        LPhi = LEye_now.sel(ellipse_params='phi').values
        head_theta = SIDE_par_now.values

        # zero-center head theta, and get rid of wrap-around effect (mod 360)
        # add pi/8 since this is roughly head tilt in movies relative to mean theta
        th = head_theta - np.nanmedian(SIDE_par_interp) + np.pi + np.pi/8

        # eye divergence (theta)
        div = 0.5 * (RTheta - LTheta)
        # gaze (mean theta of eyes)
        gaze_th = (RTheta + LTheta) * 0.5
        # gaze (mean phi of eyes)
        gaze_phi = (RPhi + LPhi) * 0.5

        # plot mouse head poisiton with 'tracers'
        for i in range(0,15):
            frame_before = frame_num - i
            if frame_before >= 0:
                head_x = SIDE_pts_interp.sel(point_loc='LEye_x', frame=frame_before).values
                head_y = SIDE_pts_interp.sel(point_loc='LEye_y', frame=frame_before).values
                try:
                    SIDE_frame = cv2.circle(SIDE_frame, (int(head_x),int(head_y)), 2, (255,0,0), -1)
                except ValueError:
                    pass

        # blue circle over the current position of the eye
        eyecent_x = SIDE_pts_now.sel(point_loc='LEye_x').values
        eyecent_y = SIDE_pts_now.sel(point_loc='LEye_y').values
        try:
            SIDE_frame = cv2.circle(SIDE_frame, (int(eyecent_x),int(eyecent_y)), 3, (255,0,0), -1)
        except ValueError:
            pass

        # calculate and plot head vector
        headV_x1 = SIDE_pts_now.sel(point_loc='LEye_x').values
        headV_y1 = SIDE_pts_now.sel(point_loc='LEye_y').values
        headV_x2 = SIDE_pts_now.sel(point_loc='LEye_x').values + 200 * np.cos(th)
        headV_y2 = SIDE_pts_now.sel(point_loc='LEye_y').values + 200 * np.sin(th)
        # black line of the head vector
        try:
            SIDE_frame = cv2.line(SIDE_frame, (int(headV_x1),int(headV_y1)), (int(headV_x2),int(headV_y2)), (255,255,255), thickness=2)
        except ValueError:
            pass

        # calculate gaze direction (head and eyes)
        # subtract off the pi/8 that was added above
        rth = th - div * np.pi/180 - np.pi/8
        gazeV_x1 = SIDE_pts_now.sel(point_loc='LEye_x').values
        gazeV_y1 = SIDE_pts_now.sel(point_loc='LEye_y').values
        gazeV_x2 = SIDE_pts_now.sel(point_loc='LEye_x').values + 200 * np.cos(rth)
        gazeV_y2 = SIDE_pts_now.sel(point_loc='LEye_y').values + 200 * np.sin(rth)
        # cyan line of gaze direction
        try:
            SIDE_frame = cv2.line(SIDE_frame, (int(gazeV_x1),int(gazeV_y1)), (int(gazeV_x2),int(gazeV_y2)), (255,255,0), thickness=2)
        except ValueError:
            pass

        vidout.write(SIDE_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        out_vid.release()
        cv2.destroyAllWindows()
