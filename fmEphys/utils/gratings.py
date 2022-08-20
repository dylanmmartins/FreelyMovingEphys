import numpy as np
from tqdm import tqdm
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
 
from fmEphys.filter import nanmedfilt
from ephys_base import calc_PSTH
 
 
def calc_stim(worldVid, worldT):
 
   # Drop static worldcam pixels
   std_im = np.std(worldVid, axis=0)
   std_im[std_im < 20] = 0
 
   # Normalize video
   norm_vid = (worldVid - np.mean(worldVid, axis=0)) / std_im
   norm_vid = norm_vid * (std_im > 0)
 
   # Setup empty arrays
   n_frames = np.size(worldVid, 0) - 1
 
   u_mn = np.zeros((n_frames, 1))
   v_mn = np.zeros((n_frames, 1))
 
   sx_mn = np.zeros((n_frames, 1))
   sy_mn = np.zeros((n_frames, 1))
 
   flow_norm = np.zeros((n_frames, np.size(worldVid, 1), np.size(worldVid, 2), 2))
  
   # Find screen
   meanx = np.mean(std_im>0, axis=0)
   xcent = np.int(np.sum(meanx * np.arange(len(meanx))) / np.sum(meanx))
 
   meany = np.mean(std_im>0, axis=1)
   ycent = np.int(np.sum(meany * np.arange(len(meany))) / np.sum(meany))
 
   print('Calculate optic flow of gratings stimulus.')
  
   # Pixel range to define monitor
   xrg = 40
   yrg = 25
 
   for f in tqdm(range(n_frames)):
 
       frame_0 = np.uint8(32 * (norm_vid[f, :, :] + 4))
       frame_1 = np.uint8(32 * (norm_vid[f+1, :, :] + 4))
 
       flow_norm[f, :, :, :] = cv2.calcOpticalFlowFarneback(prev=frame_0, next=frame_1,
                                                         flow=None, pyr_scale=0.5, levels=3,
                                                         winsize=30, iterations=3, poly_n=7,
                                                         poly_sigma=1.5, flags=0)
 
       # Negative flow_norm for `v` to fix sign for y axis in images.
       u = flow_norm[f, :, :, 0]
       v = -flow_norm[f, :, :, 1]
 
       sx = cv2.Sobel(frame_0, cv2.CV_64F, 1, 0, ksize=11)
       sy = -cv2.Sobel(frame_0, cv2.CV_64F, 0, 1, ksize=11)
 
       # Get rid of values outside of monitor.
       sx[std_im < 20] = 0
       sy[std_im < 20] = 0
 
       # Make vectors point in positive x direction so that opposite sides of
       # grating do not cancel.
       sy[sx<0] = -sy[sx<0]
       sx[sx<0] = -sx[sx<0]
 
       sy[np.abs(sx / sy) < 0.15] = np.abs(sy[np.abs(sx / sy) < 0.15])
 
       u_mn[f] = np.mean(u[ycent-yrg : ycent+yrg, xcent-xrg : xcent+xrg])
       v_mn[f]= np.mean(v[ycent-yrg : ycent+yrg, xcent-xrg : xcent+xrg])
 
       sx_mn[f] = np.mean(sx[ycent-yrg : ycent+yrg, xcent-xrg : xcent+xrg])
       sy_mn[f] = np.mean(sy[ycent-yrg : ycent+yrg, xcent-xrg : xcent+xrg])
 
   scr_contrast = np.empty(worldT.size)
 
   for i in range(worldT.size):
       scr_contrast[i] = np.nanmean(np.abs(norm_vid[i, ycent-25:ycent+25, xcent-40:xcent+40]))
 
   scr_contrast = nanmedfilt(scr_contrast, 11)
  
   stim_is_on = np.double(scr_contrast > 0.5)
   stim_onset = np.array(worldT[np.where(np.diff(stim_is_on) > 0)])
 
   print('Calculate properties of each stimulus presentation.')
 
   stim_end = np.array(worldT[np.where(np.diff(stim_is_on) < 0)])
   stim_end = stim_end[stim_end > stim_onset[0]]
   stim_onset = stim_onset[stim_onset < stim_end[-1]]
 
   grating_ori = np.zeros(len(stim_onset)) # ori
   grating_mag = np.zeros(len(stim_onset)) # sf
   grating_dir = np.zeros(len(stim_onset)) # direction
   dI = np.zeros(len(stim_onset)) # tf
 
   for i in range(len(stim_onset)):
 
       tpts = np.where((worldT > stim_onset[i] + 0.025) & (worldT < stim_end[i] - 0.025))
       # mag = np.sqrt(sx_mn[tpts]**2 + sy_mn[tpts]**2)
       # use_pts = np.array(tpts)[0, np.where(mag[:, 0] > np.percentile(mag, 25))]
 
       stim_sx = np.nanmedian(sx_mn[tpts])
       stim_sy = np.nanmedian(sy_mn[tpts])
       stim_u = np.nanmedian(u_mn[tpts])
       stim_v = np.nanmedian(v_mn[tpts])
 
       # Orientation of gratings
       grating_ori[i] = np.arctan2(stim_sy, stim_sx)
 
       # Spatial frequency
       grating_mag[i] = np.sqrt(stim_sx**2 + stim_sy**2)
 
       # Direction (dot product of gratient and flow)
       grating_dir[i] = np.sign(stim_u*stim_sx + stim_v*stim_sy)
 
       # Temporal frequency (rate of change of image)
       dI[i] = np.mean(np.diff(norm_vid[tpts, ycent, xcent])**2)
 
   grating_ori[grating_dir < 0] = grating_ori[grating_dir < 0] + np.pi
   grating_ori = grating_ori - np.min(grating_ori)
   ori_labels = np.floor((grating_ori + np.pi / 16) / (np.pi / 4))
 
   # Temporal frequency categories:
   # 0 = low, 2 cps
   # 1 = high, 8 cps
   grating_tf = np.zeros(len(stim_onset))
   grating_tf[dI > 0.5] = 1
 
   km = KMeans(n_clusters=3)
   km.fit(np.reshape(grating_mag, (-1, 1)))
   sf_labels = km.labels_
   sf_kcenters = km.cluster_centers_
 
   order = np.argsort(np.reshape(sf_kcenters, 3))
 
   sf_catnew = sf_labels.copy()
   for i in range(3):
       sf_catnew[sf_labels == order[i]] = i
   sf_labels = sf_catnew.copy()
 
   ntrial = np.zeros((3 ,8))
   for i in range(3): # sf
       for j in range(8): # ori
           ntrial[i, j] = np.sum((sf_labels==i) & (ori_labels==j))
  
   # -> figure
   fig, [ax0, ax1] = plt.subplots(2, 1, figsize=(5,3), dpi=100)
 
   ax0.scatter(grating_mag, grating_ori, c=ori_labels)
   ax0.set_xlabel('grating magnitude'); ax0.set_ylabel('theta')
  
   colorbar_img = ax1.imshow(ntrial, vmin=0, vmax=(2 * np.mean(ntrial)))
   ax1.colorbar(colorbar_img, ax=ax1)
   ax1.set_xlabel('orientations')
   ax1.set_ylabel('sfs')
   ax1.set_title('trials per condition')
 
   fig.savefig()
   fig.close()
 
def calc_tuning(cells, stim_props):
   """
   cells is a df
   stim_props is a dict that came out of the calc_stim() func
   """
   edge_win = 0.025
 
   n_cells = len(cells.index.values)
 
   stim_onset = stim_props['stim_onset']
   stim_end = stim_props['stim_end']
 
   ori_labels = stim_props['ori_labels']
   sf_labels = stim_props['sf_labels']
   tf_labels = stim_props['tf_labels']
  
   stim_rate = np.zeros((len(n_cells), len(stim_onset)))
   spont_rate = np.zeros((len(n_cells), len(stim_onset)))
   drift_spont = np.zeros(len(n_cells))
 
   tuning = np.zeros((len(n_cells), 8, 3, 2))
 
   for cell_i, cell_ind in enumerate(cells.index.values): # was c, ind
      
       # Spike times for this unit
       sp = cells.at[cell_ind, 'Gt_spikeT'].copy()
 
       # Stimulus spike rate
       for stim_i in range(len(stim_onset)):
 
           spike_count = np.sum((sp > stim_onset[stim_i] + edge_win) & (sp < stim_end[stim_i]))
           duration = (stim_end[stim_i] - stim_onset[stim_i] - edge_win)
 
           stim_rate[cell_i, stim_i] = spike_count / duration
 
       # Spontanious spike rate
       for stim_i in range(len(stim_onset) - 1):
          
           spike_count = np.sum((sp > stim_end[stim_i] + edge_win) & (sp < stim_onset[stim_i+1]))
           duration = (stim_onset[stim_i+1] - stim_end[stim_i] - edge_win)
 
           spont_rate[cell_i, stim_i] = spike_count / duration
 
       # Orientation tuning
       for ori in range(8):
           for sf in range(3):
               for tf in range(2):
                  
                   tuning[cell_i, ori, sf, tf] = np.mean(stim_rate[cell_i, (ori_labels==ori) & (sf_labels ==sf) & (tf_labels==tf)])
      
       drift_spont[cell_i] = np.mean(spont_rate[cell_i, :])
 
       # Shift orientation to make rightward gratings 0deg
       tuning = np.roll(tuning, shift=-1, axis=1)
 
       # One more roll to correct an offset
       tuning = np.roll(tuning, 1, axis=1)
 
       tuning_props = {
           'drift_spont': drift_spont,
           'tuning': tuning,
           'stim_rate': stim_rate,
           'spont_rate': spont_rate,
           'drift_spont': drift_spont,
           'ori_labels': ori_labels,
           'ori'
           'tuning': tuning
       }
 
def analysis(cells):
   ### Gratings stimulus
   stim_props =
 
 
   ### PSTH for each cell
 
   stim_onset = stim_props['stim_onset']
 
   # Iterate through cells in the dataset.
   for ind, spikeT in cells['Gt_spikeT'].iteritems():
 
       cells.at[ind, 'DfGt_stim_onset_PSTH'] = calc_PSTH(spikeT, stim_onset, edgedrop=30, win=1500)
 
      
 
   ### Gratings tuning.