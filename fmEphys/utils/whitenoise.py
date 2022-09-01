import os

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from scipy.signal import medfilt
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import fmEphys

def main():




    def glm_save(self):
        unit_data = pd.DataFrame([])
        stim = 'wn'
        for unit_num, ind in enumerate(self.cells.index):
            cols = [stim+'_'+i for i in ['contrast',
                                        'contrast_tuning_bins',
                                        'contrast_tuning',
                                        'contrast_tuning_err',
                                        'spike_triggered_average',
                                        'spike_triggered_variance',
                                        'rightsacc_avg',
                                        'leftsacc_avg',
                                        'pupilradius_tuning_bins',
                                        'pupilradius_tuning',
                                        'pupilradius_tuning_err',
                                        'theta_tuning_bins',
                                        'theta_tuning',
                                        'theta_tuning_err',
                                        'phi_tuning_bins',
                                        'phi_tuning',
                                        'phi_tuning_err',
                                        'eyeT',
                                        'dEye_dpf',
                                        'dEye_dps',
                                        'theta',
                                        'phi',
                                        'ballspeed',
                                        'ballspeed_tuning_bins',
                                        'ballspeed_tuning',
                                        'ballspeed_tuning_err',
                                        'lfp_power',
                                        'layer5cent_from_lfp',
                                        'glm_receptive_field',
                                        'glm_cc']]
            unit_df = pd.DataFrame(pd.Series([self.contrast,
                                    self.contrast_tuning_bins,
                                    self.contrast_tuning[unit_num],
                                    self.contrast_tuning_err[unit_num],
                                    self.sta[unit_num],
                                    self.stv[unit_num],
                                    self.rightsacc_avg[unit_num],
                                    self.leftsacc_avg[unit_num],
                                    self.pupilradius_tuning_bins,
                                    self.pupilradius_tuning[unit_num],
                                    self.pupilradius_tuning_err[unit_num],
                                    self.theta_tuning_bins,
                                    self.theta_tuning[unit_num],
                                    self.theta_tuning_err[unit_num],
                                    self.phi_tuning_bins,
                                    self.phi_tuning[unit_num],
                                    self.phi_tuning_err[unit_num],
                                    self.eyeT,
                                    self.dEye,
                                    self.dEye_dps,
                                    self.theta,
                                    self.phi,
                                    self.ball_speed,
                                    self.ballspeed_tuning_bins,
                                    self.ballspeed_tuning[unit_num],
                                    self.ballspeed_tuning_err[unit_num],
                                    self.lfp_power_profiles,
                                    self.lfp_layer5_centers,
                                    self.glm_rf[unit_num],
                                    self.glm_cc[unit_num]]),dtype=object).T
            unit_df.columns = cols
            unit_df.index = [ind]
            unit_df['session'] = self.session_name
            unit_data = pd.concat([unit_data, unit_df], axis=0)
        data_out = pd.concat([self.cells, unit_data], axis=1)
        data_out.to_hdf(os.path.join(self.recording_path, (self.recording_name+'_ephys_props.h5')), 'w')

    def save(self):
        unit_data = pd.DataFrame([])
        stim = 'Wn'
        for unit_num, ind in enumerate(self.cells.index):
            cols = [stim+'_'+i for i in ['contrast',
                                        'contrast_tuning_bins',
                                        'contrast_tuning',
                                        'contrast_tuning_err',
                                        'spike_triggered_average',
                                        'spike_triggered_variance',
                                        'rightsacc_avg',
                                        'leftsacc_avg',
                                        'pupilradius_tuning_bins',
                                        'pupilradius_tuning',
                                        'pupilradius_tuning_err',
                                        'theta_tuning_bins',
                                        'theta_tuning',
                                        'theta_tuning_err',
                                        'phi_tuning_bins',
                                        'phi_tuning',
                                        'phi_tuning_err',
                                        'eyeT',
                                        'dEye_dpf',
                                        'dEye_dps',
                                        'theta',
                                        'phi',
                                        'ballspeed',
                                        'ballspeed_tuning_bins',
                                        'ballspeed_tuning',
                                        'ballspeed_tuning_err',
                                        'lfp_power',
                                        'layer5cent_from_lfp']]
            unit_df = pd.DataFrame(pd.Series([self.contrast,
                                    self.contrast_tuning_bins,
                                    self.contrast_tuning[unit_num],
                                    self.contrast_tuning_err[unit_num],
                                    self.sta[unit_num],
                                    self.stv[unit_num],
                                    self.rightsacc_avg[unit_num],
                                    self.leftsacc_avg[unit_num],
                                    self.pupilradius_tuning_bins,
                                    self.pupilradius_tuning[unit_num],
                                    self.pupilradius_tuning_err[unit_num],
                                    self.theta_tuning_bins,
                                    self.theta_tuning[unit_num],
                                    self.theta_tuning_err[unit_num],
                                    self.phi_tuning_bins,
                                    self.phi_tuning[unit_num],
                                    self.phi_tuning_err[unit_num],
                                    self.eyeT,
                                    self.dEye,
                                    self.dEye_dps,
                                    self.theta,
                                    self.phi,
                                    self.ball_speed,
                                    self.ballspeed_tuning_bins,
                                    self.ballspeed_tuning[unit_num],
                                    self.ballspeed_tuning_err[unit_num],
                                    self.lfp_power_profiles,
                                    self.lfp_layer5_centers]),dtype=object).T
            unit_df.columns = cols
            unit_df.index = [ind]
            unit_df['session'] = self.session_name
            unit_data = pd.concat([unit_data, unit_df], axis=0)
        data_out = pd.concat([self.cells, unit_data], axis=1)
        data_out.to_hdf(os.path.join(self.recording_path, (self.recording_name+'_ephys_props.h5')), 'w')

    def save_glm_model_inputs(self):
        """ Save an npz file out that has inputs needed for post-processing glm.
        Just do this to avoid duplicating videos, etc. for all units, when the stim is shared.
        """
        np.savez(file=os.path.join(self.recording_path, 'glm_model_inputs.h5'),
                 model_t=self.model_t,
                 model_video=self.model_vid,
                 model_rough_correction_video=self.glm_model_vid,
                 model_nsp=self.model_nsp,
                 model_theta=self.model_theta,
                 model_phi=self.model_phi,
                 model_use=self.model_use,
        )

    def analyze(self):
        # delete the existing h5 file, so that a new one can be written
        if os.path.isfile(os.path.join(self.recording_path, (self.recording_name+'_ephys_props.h5'))):
            os.remove(os.path.join(self.recording_path, (self.recording_name+'_ephys_props.h5')))

        # self.overview_pdf = PdfPages(os.path.join(self.recording_path, (self.recording_name + '_overview_analysis_figures.pdf')))
        self.detail_pdf = PdfPages(os.path.join(self.recording_path, (self.recording_name + '_detailed_analysis_figures.pdf')))
        self.diagnostic_pdf = PdfPages(os.path.join(self.recording_path, (self.recording_name + '_diagnostic_analysis_figures.pdf')))

        print('starting ephys analysis for '+self.recording_name)
        self.base_ephys_analysis()

        print('making summary and overview figures')
        # self.overview_fig()
        # self.summary_fig()

        print('closing pdfs')
        # self.overview_pdf.close();
        self.detail_pdf.close(); self.diagnostic_pdf.close()

        print('saving ephys file')
        if self.do_rough_glm_fit:
            self.glm_save()
        elif not self.do_rough_glm_fit:
            self.save()
        
        if self.do_glm_model_preprocessing:
            print('saving inputs to full glm model')
            self.save_glm_model_inputs()


class HeadFixedSparseNoise(Ephys):
    def __init__(self, config, recording_name, recording_path):
        Ephys.__init__(self, config, recording_name, recording_path)

        self.fm = False
        self.stim = 'sn'

    def save(self):
        unit_data = pd.DataFrame([])
        stim = 'Sn'
        for unit_num, ind in enumerate(self.cells.index):
            cols = [stim+'_'+i for i in ['contrast',
                                        'contrast_tuning_bins',
                                        'contrast_tuning',
                                        'contrast_tuning_err',
                                        'spike_triggered_average',
                                        'spike_triggered_variance',
                                        'rightsacc_avg',
                                        'leftsacc_avg',
                                        'pupilradius_tuning_bins',
                                        'pupilradius_tuning',
                                        'pupilradius_tuning_err',
                                        'theta_tuning_bins',
                                        'theta_tuning',
                                        'theta_tuning_err',
                                        'phi_tuning_bins',
                                        'phi_tuning',
                                        'phi_tuning_err',
                                        'eyeT',
                                        'dEye_dpf',
                                        'dEye_dps',
                                        'theta',
                                        'phi',
                                        'ballspeed',
                                        'ballspeed_tuning_bins',
                                        'ballspeed_tuning',
                                        'ballspeed_tuning_err']]
            unit_df = pd.DataFrame(pd.Series([self.contrast,
                                    self.contrast_tuning_bins,
                                    self.contrast_tuning[unit_num],
                                    self.contrast_tuning_err[unit_num],
                                    self.sta[unit_num],
                                    self.stv[unit_num],
                                    self.rightsacc_avg[unit_num],
                                    self.leftsacc_avg[unit_num],
                                    self.pupilradius_tuning_bins,
                                    self.pupilradius_tuning[unit_num],
                                    self.pupilradius_tuning_err[unit_num],
                                    self.theta_tuning_bins,
                                    self.theta_tuning[unit_num],
                                    self.theta_tuning_err[unit_num],
                                    self.phi_tuning_bins,
                                    self.phi_tuning[unit_num],
                                    self.phi_tuning_err[unit_num],
                                    self.eyeT,
                                    self.dEye,
                                    self.dEye_dps,
                                    self.theta,
                                    self.phi,
                                    self.ball_speed,
                                    self.ballspeed_tuning_bins,
                                    self.ballspeed_tuning[unit_num],
                                    self.ballspeed_tuning_err[unit_num]]),dtype=object).T
            unit_df.columns = cols
            unit_df.index = [ind]
            unit_df['session'] = self.session_name
            unit_data = pd.concat([unit_data, unit_df], axis=0)
        data_out = pd.concat([self.cells, unit_data], axis=1)
        data_out.to_hdf(os.path.join(self.recording_path, (self.recording_name+'_ephys_props.h5')), 'w')

    def analyze(self):
        # delete the existing h5 file, so that a new one can be written
        if os.path.isfile(os.path.join(self.recording_path, (self.recording_name+'_ephys_props.h5'))):
            os.remove(os.path.join(self.recording_path, (self.recording_name+'_ephys_props.h5')))

        self.detail_pdf = PdfPages(os.path.join(self.recording_path, (self.recording_name + '_detailed_analysis_figures.pdf')))
        self.diagnostic_pdf = PdfPages(os.path.join(self.recording_path, (self.recording_name + '_diagnostic_analysis_figures.pdf')))
        
        print('starting ephys analysis for '+self.recording_name)
        self.base_ephys_analysis()

        print('making summary and overview figures')
        # self.summary_fig()

        print('closing pdfs')
        self.detail_pdf.close(); self.diagnostic_pdf.close()

        print('saving ephys file')
        self.save()
        
class HeadFixedGratings(Ephys):
    def __init__(self, config, recording_name, recording_path):
        Ephys.__init__(self, config, recording_name, recording_path)
        self.fm = False
        self.stim = 'gt'

        self.ori_x = np.arange(8)*45


    def stim_psth(self, lower=-0.5, upper=1.5, dt=0.1):
        """ Calculate and plot PSTH relative to stimulus onset
        """
        bins = np.arange(lower, upper+dt, dt)
        fig = plt.figure(figsize=(10, int(np.ceil(self.n_cells / 2))))
        # empty array into which psth will be saved
        psth = np.zeros([self.n_cells, len(bins)-1])
        # iterate through units
        for i, ind in enumerate(self.cells.index):
            plt.subplot(int(np.ceil(self.n_cells/4)), 4, i+1)
            # empty array for psth of this unit
            this_psth = np.zeros(len(bins)-1)
            for t in self.stim_start:
                # get a histogram of spike times in each of the stimulus bins
                hist, edges = np.histogram(self.cells.at[ind,'spikeT']-t, bins)
                # make this cumulative
                this_psth = this_psth + hist
            # normalize spikes in bins to the number of times the stim had an onset
            this_psth = this_psth / len(self.stim_start)
            # then normalize to length of time for each bin
            this_psth = this_psth / dt
            # plot histogram as a line
            plt.plot(bins[0:-1] + dt / 2, this_psth)
            plt.ylim(0, np.nanmax(this_psth) * 1.2)
            # add psth from this unit to array of all units
            psth[i,:] = this_psth
        plt.xlabel('time'); plt.ylabel('sp/sec')
        plt.title('gratings psth')
        plt.tight_layout(); plt.close()
        self.grating_psth = psth
        plt.tight_layout(); self.detail_pdf.savefig(); plt.close()

    def gratings_analysis(self, xrg=40, yrg=25):
        """
        xrg, yrg -- pixel range to define monitor
        """
        # setup
        nf = np.size(self.img_norm, 0) - 1
        u_mn = np.zeros((nf, 1)); v_mn = np.zeros((nf, 1))
        sx_mn = np.zeros((nf, 1)); sy_mn = np.zeros((nf, 1))
        flow_norm = np.zeros((nf, np.size(self.img_norm,1), np.size(self.img_norm,2), 2))
        
        # find screen
        meanx = np.mean(self.std_im>0, axis=0)
        xcent = np.int(np.sum(meanx * np.arange(len(meanx))) / np.sum(meanx))
        meany = np.mean(self.std_im>0, axis=1)
        ycent = np.int(np.sum(meany * np.arange(len(meany))) / np.sum(meany))

        # animation of optic flow
        fig, ax = plt.subplots(1,1,figsize = (16,8))
        for f in tqdm(range(nf)):
            frm = np.uint8(32*(self.img_norm[f,:,:]+4))
            frm2 = np.uint8(32*(self.img_norm[f+1,:,:]+4))
            flow_norm[f,:,:,:] = cv2.calcOpticalFlowFarneback(frm,frm2, None, 0.5, 3, 30, 3, 7, 1.5, 0)
            u = flow_norm[f,:,:,0]; v = -flow_norm[f,:,:,1] # negative to fix sign for y axis in images
            sx = cv2.Sobel(frm, cv2.CV_64F, 1, 0, ksize=11)
            sy = -cv2.Sobel(frm, cv2.CV_64F, 0, 1, ksize=11) # negative to fix sign for y axis in images
            sx[self.std_im<20] = 0; sy[self.std_im<20] = 0; # get rid of values outside of monitor
            sy[sx<0] = -sy[sx<0] # make vectors point in positive x direction (so opposite sides of grating don't cancel)
            sx[sx<0] = -sx[sx<0]
            sy[np.abs(sx/sy)<0.15] = np.abs(sy[np.abs(sx/sy)<0.15])
            u_mn[f] = np.mean(u[ycent-yrg:ycent+yrg, xcent-xrg:xcent+xrg])
            v_mn[f]= np.mean(v[ycent-yrg:ycent+yrg, xcent-xrg:xcent+xrg]); 
            sx_mn[f] = np.mean(sx[ycent-yrg:ycent+yrg, xcent-xrg:xcent+xrg])
            sy_mn[f] = np.mean(sy[ycent-yrg:ycent+yrg, xcent-xrg:xcent+xrg])
        scr_contrast = np.empty(self.worldT.size)
        for i in range(self.worldT.size):
            scr_contrast[i] = np.nanmean(np.abs(self.img_norm[i, ycent-25:ycent+25, xcent-40:xcent+40]))
        scr_contrast = medfilt(scr_contrast, 11)
        stimOn = np.double(scr_contrast>0.5)
        self.stim_start = np.array(self.worldT[np.where(np.diff(stimOn)>0)])
        
        self.stim_psth()

        stim_end = np.array(self.worldT[np.where(np.diff(stimOn)<0)])
        stim_end = stim_end[stim_end>self.stim_start[0]]
        self.stim_start = self.stim_start[self.stim_start<stim_end[-1]]
        grating_th = np.zeros(len(self.stim_start))
        grating_mag = np.zeros(len(self.stim_start))
        grating_dir = np.zeros(len(self.stim_start))
        dI = np.zeros(len(self.stim_start))
        for i in range(len(self.stim_start)):
            tpts = np.where((self.worldT>self.stim_start[i] + 0.025) & (self.worldT<stim_end[i]-0.025))
            mag = np.sqrt(sx_mn[tpts]**2 + sy_mn[tpts]**2)
            this = np.where(mag[:,0] > np.percentile(mag,25))
            goodpts = np.array(tpts)[0,this]
            stim_sx = np.nanmedian(sx_mn[tpts])
            stim_sy = np.nanmedian(sy_mn[tpts])
            stim_u = np.nanmedian(u_mn[tpts])
            stim_v = np.nanmedian(v_mn[tpts])
            grating_th[i] = np.arctan2(stim_sy, stim_sx)
            grating_mag[i] = np.sqrt(stim_sx**2 + stim_sy**2)
            grating_dir[i] = np.sign(stim_u*stim_sx + stim_v*stim_sy) # dot product of gratient and flow gives direction
            dI[i] = np.mean(np.diff(self.img_norm[tpts, ycent, xcent])**2) # rate of change of image give temporal frequency
        self.grating_ori = grating_th.copy()
        self.grating_ori[grating_dir<0] = self.grating_ori[grating_dir<0] + np.pi
        self.grating_ori = self.grating_ori - np.min(self.grating_ori)
        grating_tf = np.zeros(len(self.stim_start))
        grating_tf[dI>0.5] = 1;  # spatial frequencies: 0=low, 1=high
        ori_cat = np.floor((self.grating_ori+np.pi/16) / (np.pi/4))
        
        plt.figure()
        plt.plot(range(15), ori_cat[:15]); plt.xlabel('first 15 stims'); plt.ylabel('ori cat')
        self.diagnostic_pdf.savefig()

        km = KMeans(n_clusters=3).fit(np.reshape(grating_mag, (-1,1)))
        sf_cat = km.labels_
        order = np.argsort(np.reshape(km.cluster_centers_, 3))
        sf_catnew = sf_cat.copy()
        for i in range(3):
            sf_catnew[sf_cat == order[i]] = i
        self.sf_cat = sf_catnew.copy()

        plt.figure(figsize=(8,8))
        plt.scatter(grating_mag, self.grating_ori, c=ori_cat)
        plt.xlabel('grating magnitude'); plt.ylabel('theta')
        self.diagnostic_pdf.savefig(); plt.close()

        ntrial = np.zeros((3,8))
        for i in range(3):
            for j in range(8):
                ntrial[i,j] = np.sum((sf_cat==i) & (ori_cat==j))
        plt.figure()
        plt.imshow(ntrial, vmin=0, vmax=2*np.mean(ntrial))
        plt.colorbar(); plt.xlabel('orientations')
        plt.ylabel('sfs'); plt.title('trials per condition')
        self.diagnostic_pdf.savefig(); plt.close()

        # plotting grating orientation and tuning curves
        edge_win = 0.025
        self.grating_rate = np.zeros((len(self.cells), len(self.stim_start)))
        self.spont_rate = np.zeros((len(self.cells), len(self.stim_start)))
        self.ori_tuning = np.zeros((len(self.cells), 8, 3))
        self.ori_tuning_tf = np.zeros((len(self.cells), 8, 3, 2))
        self.drift_spont = np.zeros(len(self.cells))
        plt.figure(figsize=(12, self.n_cells*2))
        for c, ind in enumerate(self.cells.index):
            sp = self.cells.at[ind,'spikeT'].copy()
            for i in range(len(self.stim_start)):
                self.grating_rate[c, i] = np.sum((sp > self.stim_start[i]+edge_win) & (sp < stim_end[i])) / (stim_end[i] - self.stim_start[i] - edge_win)
            for i in range(len(self.stim_start)-1):
                self.spont_rate[c, i] = np.sum((sp > stim_end[i]+edge_win) & (sp < self.stim_start[i+1])) / (self.stim_start[i+1] - stim_end[i] - edge_win)  
            for ori in range(8):
                for sf in range(3):
                    self.ori_tuning[c, ori, sf] = np.mean(self.grating_rate[c, (ori_cat==ori) & (sf_cat==sf)])
                    for tf in range(2):
                        self.ori_tuning_tf[c, ori, sf, tf] = np.mean(self.grating_rate[c, (ori_cat==ori) & (sf_cat ==sf) & (grating_tf==tf)])
            self.drift_spont[c] = np.mean(self.spont_rate[c, :])
            plt.subplot(self.n_cells, 4, 4*c+1)
            plt.scatter(self.grating_ori, self.grating_rate[c,:], c=sf_cat)
            plt.plot(3*np.ones(len(self.spont_rate[c,:])), self.spont_rate[c,:], 'r.')
            
            plt.subplot(self.n_cells, 4, 4*c+2)
            plt.plot(self.ori_x, self.ori_tuning[c,:,0], label='low sf')
            plt.plot(self.ori_x, self.ori_tuning[c,:,1], label='mid sf')
            plt.plot(self.ori_x, self.ori_tuning[c,:,2], label='high sf')
            plt.plot([np.min(self.ori_x), np.max(self.ori_x)], [self.drift_spont[c], self.drift_spont[c]], 'r:', label='spont')
            try:
                plt.ylim(0, np.nanmax(self.ori_tuning_tf[c,:,:]*1.2))
            except ValueError:
                plt.ylim(0,1)
            plt.legend()

            plt.subplot(self.n_cells, 4, 4*c+3)
            plt.plot(self.ori_x, self.ori_tuning_tf[c,:,0,0], label='low sf')
            plt.plot(self.ori_x, self.ori_tuning_tf[c,:,1,0], label='mid sf')
            plt.plot(self.ori_x, self.ori_tuning_tf[c,:,2,0], label='high sf')
            plt.plot([np.min(self.ori_x), np.max(self.ori_x)], [self.drift_spont[c], self.drift_spont[c]], 'r:',label ='spont')
            try:
                plt.ylim(0, np.nanmax(self.ori_tuning_tf[c,:,:]*1.2))
            except ValueError:
                plt.ylim(0,1)
            plt.legend()

            plt.subplot(self.n_cells, 4, 4*c+4)
            plt.plot(self.ori_x, self.ori_tuning_tf[c,:,0,1], label='low sf')
            plt.plot(self.ori_x, self.ori_tuning_tf[c,:,1,1], label='mid sf')
            plt.plot(self.ori_x, self.ori_tuning_tf[c,:,2,1], label='high sf')
            plt.plot([np.min(self.ori_x), np.max(self.ori_x)], [self.drift_spont[c], self.drift_spont[c]], 'r:', label='spont')
            try:
                plt.ylim(0, np.nanmax(self.ori_tuning_tf[c,:,:]*1.2))
            except ValueError:
                plt.ylim(0,1)
            plt.legend()
        
        plt.tight_layout(); self.detail_pdf.savefig(); plt.close()

        # roll orientation tuning curves
        # ori_cat maps orientations so that ind=0 is the bottom-right corner of the monitor
        # index of sf_cat ascend moving counter-clockwise
        # ind=1 are rightward gratings; ind=5 are leftward gratings

        # shape is (cell, ori, sf), so rolling axis=1 shifts orientations so make rightward gratings 0deg
        self.ori_tuning_meantf = np.roll(self.ori_tuning, shift=-1, axis=1)
        # shape is (cell, ori, sf, tf), so again roll axis=1 to fix gratings orientations
        self.ori_tuning_tf = np.roll(self.ori_tuning_tf, shift=-1, axis=1)

    def overview_fig(self):
        plt.figure(figsize=(5, int(np.ceil(self.n_cells/2))), dpi=50)

        for i, ind in enumerate(self.cells.index):
            # plot waveform
            plt.subplot(self.n_cells, 4, i*4+1)
            wv = self.cells.at[ind,'waveform']
            plt.plot(np.arange(len(wv))*1000/self.ephys_samprate, wv)
            plt.xlabel('msec'); plt.title(str(ind)+' '+self.cells.at[ind,'KSLabel']+' cont='+str(self.cells.at[ind,'ContamPct']))
            
            # plot orientation tuning curve
            plt.subplot(self.n_cells, 4, i*4+2)
            plt.scatter(self.grating_ori, self.grating_rate[i,:], c=self.sf_cat)
            plt.plot(3*np.ones(len(self.spont_rate[i,:])), self.spont_rate[i,:], 'r:')

            # plot tuning curve
            plt.subplot(self.n_cells, 4, i*4+3)
            plt.plot(self.ori_x, self.ori_tuning_meantf[i,:,0], label='low sf')
            plt.plot(self.ori_x, self.ori_tuning_meantf[i,:,1], label='mid sf')
            plt.plot(self.ori_x, self.ori_tuning_meantf[i,:,2], label='high sf')
            plt.plot([np.min(self.ori_x), np.max(self.ori_x)], [self.drift_spont[i], self.drift_spont[i]], 'r:', label='spont')
            try:
                plt.ylim(0, np.nanmax(self.ori_tuning_meantf[i,:,:]*1.2))
            except ValueError:
                plt.ylim(0,1)
            plt.xlabel('orientation (deg)')
            
            # plot eye movements
            plt.subplot(self.n_cells, 4, i*4+4)
            plt.plot(self.trange_x, self.rightsacc_avg[i,:], color='tab:blue',label='right')
            plt.plot(self.trange_x, self.leftsacc_avg[i,:],color='red',label='left')
            maxval = np.max(np.maximum(self.rightsacc_avg[i,:], self.leftsacc_avg[i,:]))
            plt.vlines(0, 0, maxval*1.5, linestyles='dotted', colors='k')
            plt.ylim([0, maxval*1.2]); plt.ylabel('sp/sec'); plt.legend()
        
        plt.tight_layout(); self.overview_pdf.savefig(); plt.close()

    def save(self):
        unit_data = pd.DataFrame([])
        stim = 'Gt'
        for unit_num, ind in enumerate(self.cells.index):
            cols = [stim+'_'+i for i in ['contrast',
                                        'contrast_tuning_bins',
                                        'contrast_tuning',
                                        'contrast_tuning_err',
                                        'spike_triggered_average',
                                        'spike_triggered_variance',
                                        'rightsacc_avg',
                                        'leftsacc_avg',
                                        'pupilradius_tuning_bins',
                                        'pupilradius_tuning',
                                        'pupilradius_tuning_err',
                                        'theta_tuning_bins',
                                        'theta_tuning',
                                        'theta_tuning_err',
                                        'phi_tuning_bins',
                                        'phi_tuning',
                                        'phi_tuning_err',
                                        'eyeT',
                                        'dEye_dpf',
                                        'dEye_dps',
                                        'theta',
                                        'phi',
                                        'ballspeed',
                                        'ballspeed_tuning_bins',
                                        'ballspeed_tuning',
                                        'ballspeed_tuning_err',
                                        'grating_psth',
                                        'grating_ori',
                                        'ori_tuning_mean_tf',
                                        'ori_tuning_tf',
                                        'drift_spont',
                                        'spont_rate',
                                        'grating_rate',
                                        'sf_cat']]
            unit_df = pd.DataFrame(pd.Series([self.contrast,
                                    self.contrast_tuning_bins,
                                    self.contrast_tuning[unit_num],
                                    self.contrast_tuning_err[unit_num],
                                    self.sta[unit_num],
                                    self.stv[unit_num],
                                    self.rightsacc_avg[unit_num],
                                    self.leftsacc_avg[unit_num],
                                    self.pupilradius_tuning_bins,
                                    self.pupilradius_tuning[unit_num],
                                    self.pupilradius_tuning_err[unit_num],
                                    self.theta_tuning_bins,
                                    self.theta_tuning[unit_num],
                                    self.theta_tuning_err[unit_num],
                                    self.phi_tuning_bins,
                                    self.phi_tuning[unit_num],
                                    self.phi_tuning_err[unit_num],
                                    self.eyeT,
                                    self.dEye,
                                    self.dEye_dps,
                                    self.theta,
                                    self.phi,
                                    self.ball_speed,
                                    self.ballspeed_tuning_bins,
                                    self.ballspeed_tuning[unit_num],
                                    self.ballspeed_tuning_err[unit_num],
                                    self.grating_psth[unit_num],
                                    self.grating_ori,
                                    self.ori_tuning_meantf[unit_num],
                                    self.ori_tuning_tf[unit_num],
                                    self.drift_spont[unit_num],
                                    self.spont_rate[unit_num],
                                    self.grating_rate[unit_num],
                                    self.sf_cat[unit_num]]),dtype=object).T
            unit_df.columns = cols
            unit_df.index = [ind]
            unit_df['session'] = self.session_name
            unit_data = pd.concat([unit_data, unit_df], axis=0)
        data_out = pd.concat([self.cells, unit_data], axis=1)
        data_out.to_hdf(os.path.join(self.recording_path, (self.recording_name+'_ephys_props.h5')), 'w')

    def analyze(self):
        # delete the existing h5 file, so that a new one can be written
        if os.path.isfile(os.path.join(self.recording_path, (self.recording_name+'_ephys_props.h5'))):
            os.remove(os.path.join(self.recording_path, (self.recording_name+'_ephys_props.h5')))

        # self.overview_pdf = PdfPages(os.path.join(self.recording_path, (self.recording_name + '_overview_analysis_figures.pdf')))
        self.detail_pdf = PdfPages(os.path.join(self.recording_path, (self.recording_name + '_detailed_analysis_figures.pdf')))
        self.diagnostic_pdf = PdfPages(os.path.join(self.recording_path, (self.recording_name + '_diagnostic_analysis_figures.pdf')))

        print('starting ephys analysis for '+self.recording_name)
        self.base_ephys_analysis()

        print('running analysis for gratings stimulus')
        self.gratings_analysis()

        # print('making summary and overview figures')
        # self.overview_fig()
        # self.summary_fig()

        print('closing pdfs')
        self.detail_pdf.close(); self.diagnostic_pdf.close()

        print('saving ephys file')
        self.save()