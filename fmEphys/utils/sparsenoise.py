import numpy as np
from tqdm import tqdm
import fmEphys

dStim_thresh = 1e5


def calc_RF_stim(unit_STA, vid):
    flat_unit_sta = unit_STA.copy().flatten()
    on_y, on_x = np.unravel_index(np.argmax(flat_unit_sta), unit_STA.shape)
    off_y, off_x = np.unravel_index(np.argmin(flat_unit_sta), unit_STA.shape)

    on_stim_history = vid[:,on_y*2,on_x*2]
    off_stim_history = vid[:,off_y*2,off_x*2]

    return on_stim_history, (on_x, on_y), off_stim_history, (off_x, off_y)

def sort_luminance_direction(stim, eventT, eyeT, flips,
                    frameshift=4, change_thresh_RF=30):
    event_eyeT = np.zeros(len(eventT))
    for i, t in enumerate(eventT):
        event_eyeT[i] = eyeT[np.argmin(np.abs(t-eyeT))]
    
    gray = np.nanmedian(stim)

    shifted_flips = flips + frameshift
    if np.max(shifted_flips) > (stim.size-frameshift):
        shifted_flips = shifted_flips[:-1]
        event_eyeT = event_eyeT[:-1]
        
    rf_off = event_eyeT.copy()
    rf_on = event_eyeT.copy()
    only_global = event_eyeT.copy()

    off_bool = stim[shifted_flips]<(gray-change_thresh_RF)
    offT = rf_off[off_bool] # light-to-dark transitions, as a timestamp in ephys eyeT timebase
    # offInds = flips[np.where(off_bool)[0]]
    
    on_bool = stim[shifted_flips]>(gray+change_thresh_RF)
    onT = rf_on[on_bool] # same for dark-to-light transitions
    # onInds = flips[np.where(on_bool)[0]]
    
    background_bool = (stim[shifted_flips]>(gray-change_thresh_RF))     \
                       & (stim[shifted_flips]<(gray+change_thresh_RF))
    backgroundT = only_global[background_bool] # stim did not change from baseline enough
    # backgroundInds = flips[np.where(background_bool)[0]]
    
    return event_eyeT, offT, onT, backgroundT
    
def calc_Sn_psth(self):
    vid = self.Sn_world.WORLD_video.values.astype(np.uint8).astype(float)
    worldT = self.Sn_world.timestamps.values
    eyeT = self.Sn_ephys['Sn_eyeT'].iloc[0]
    ephysT0 = self.Sn_ephys['t0'].iloc[0]

    # when does the stimulus change?
    dStim = np.sum(np.abs(np.diff(vid, axis=0)), axis=(1,2))
    flips = np.argwhere((dStim[1:]>self.Sn_dStim_thresh) * (dStim[:-1]<self.Sn_dStim_thresh)).flatten()

    eventT = worldT[flips+1] - ephysT0

    rf_xy = np.zeros([len(self.Sn_ephys.index.values),4]) # [unit#, on x, on y, off x, off y]
    on_Sn_psth = np.zeros([len(self.Sn_ephys.index.values), 2001, 4]) # shape = [unit#, time, all/ltd/on/not_rf]
    off_Sn_psth = np.zeros([len(self.Sn_ephys.index.values), 2001, 4])
    for i, ind in tqdm(enumerate(self.Sn_ephys.index.values)):
        unit_sta = self.Sn_ephys.loc[ind, 'Sn_spike_triggered_average']
        on_stim_history, on_xy, off_stim_history, off_xy = self.calc_RF_stim(unit_sta, vid)
        rf_xy[i,0] = on_xy[0]; rf_xy[i,1] = on_xy[1]
        rf_xy[i,2] = off_xy[0]; rf_xy[i,3] = off_xy[1]
        # spikes
        unit_spikeT = self.Sn_ephys.loc[ind, 'spikeT']
        if len(unit_spikeT)<10: # if a unit never fired during revchecker
            continue
        # on subunit
        all_eventT, offT, onT, backgroundT = self.sort_lum(on_stim_history, eventT, eyeT, flips)
        if len(offT)==0 or len(onT)==0:
            on_Sn_psth[i,:,:] = np.nan
            continue
        # print('all={} off={}, on={}, background={}'.format(len(all_eventT), len(offT), len(onT), len(backgroundT)))
        on_Sn_psth[i,:,0] = calc_kde_sdf(unit_spikeT, all_eventT, shift_half=True)
        on_Sn_psth[i,:,1] = calc_kde_sdf(unit_spikeT, offT, shift_half=True)
        on_Sn_psth[i,:,2] = calc_kde_sdf(unit_spikeT, onT, shift_half=True)
        on_Sn_psth[i,:,3] = calc_kde_sdf(unit_spikeT, backgroundT, shift_half=True)
        # off subunit
        all_eventT, offT, onT, backgroundT = self.sort_lum(off_stim_history, eventT, eyeT, flips)
        if len(offT)==0 or len(onT)==0:
            on_Sn_psth[i,:,:] = np.nan
            continue
        # print('all={} off={}, on={}, background={}'.format(len(all_eventT), len(offT), len(onT), len(backgroundT)))
        off_Sn_psth[i,:,0] = calc_kde_sdf(unit_spikeT, all_eventT, shift_half=True)
        off_Sn_psth[i,:,1] = calc_kde_sdf(unit_spikeT, offT, shift_half=True)
        off_Sn_psth[i,:,2] = calc_kde_sdf(unit_spikeT, onT, shift_half=True)
        off_Sn_psth[i,:,3] = calc_kde_sdf(unit_spikeT, backgroundT, shift_half=True)

    self.on_Sn_psth = on_Sn_psth
    self.off_Sn_psth = off_Sn_psth
    self.rf_xy = rf_xy