import numpy as np
import matplotlib.pyplot as plt

from src import utils

def tuning()
    for i, c in enumerate(spikes.keys()):
        plt.subplot(np.ceil(self.n_cells/7).astype('int'), 7, i+1)
        plt.errorbar(var_cent, tuning[i,:], yerr=tuning_err[i,:])
        try:
            plt.ylim(0,np.nanmax(tuning[i,:]*1.2))
        except ValueError:
            plt.ylim(0,1)
        plt.xlim([variable_range[0], variable_range[-1]]); plt.title(ind, fontsize=5)
        plt.xlabel(label, fontsize=5); plt.ylabel('sp/sec', fontsize=5)
        plt.xticks(fontsize=5); plt.yticks(fontsize=5)
    plt.tight_layout()

def tuning_gratings()

def plot_PSTH(ax, psth,
              norm=False, xtype=''
              tightx=False):
    if gratings is True:
        psth_bins = np.arange(-1500, 1501)

    if tightx is True:

def plot_PSTH_hist():
    plt.subplot(np.ceil(self.n_cells/7).astype('int'), 7, i+1)
        plt.plot(self.trange_x, rightavg[i,:], color='tab:blue')
        plt.plot(self.trange_x, leftavg[i,:], color='tab:red')
        maxval = np.max(np.maximum(rightavg[i,:], leftavg[i,:]))
        plt.vlines(0, 0, maxval*1.5, linestyles='dotted', colors='k')
        plt.xlim([-0.5, 0.5])
        plt.ylim([0, maxval*1.2])
        plt.ylabel('sp/sec')
        plt.xlabel('sec')
        plt.title(str(ind)+' '+label)

def sLag_STA():

def mLag_STA()

def prelim_STA():
    plt.subplot(np.ceil(self.n_cells/7).astype('int'), 7, c+1)
        ch = int(self.cells.at[ind,'ch'])
        if self.num_channels == 64 or self.num_channels == 128:
            shank = np.floor(ch/32); site = np.mod(ch,32)
        else:
            shank = 0; site = ch
        plt.title(f'ind={ind!s} nsp={nsp!s}\n ch={ch!s} shank={shank!s}\n site={site!s}',fontsize=5)
        plt.axis('off')
    plt.imshow(sta, vmin=-0.3 ,vmax=0.3, cmap='seismic')

def quick_GLM_RFs():
    # Figure of receptive fields
    fig = plt.figure(figsize=(10,np.int(np.ceil(n_cells/3))),dpi=50)
    for celln in tqdm(range(n_cells)):
        for lag_ind, lag in enumerate(lag_list):
            crange = np.max(np.abs(sta_all[celln,:,:,:]))
            plt.subplot(n_cells, 6, (celln*6)+lag_ind+1)
            plt.imshow(sta_all[celln, lag_ind, :, :], vmin=-crange, vmax=crange, cmap='seismic')
            plt.title('cc={:.2f}'.format(cc_all[celln,lag_ind]), fontsize=5)

def STV():

def scatter_columns(ax, data
                        labels=None, colors=None, use_median=False, markersize=2,
                        template=None):
    """ Categorical scatter plot
    With categories as columns

    data should be a list of arrays, where each list item is an array of values for that category
    not an array of arrays ebcause we want to be able to have different numbers of samples in each cat
    
    """

    labels = np.arange(0, len(data), 1)

    if template == 'eyemov':
        labels = ['early', 'late', 'biphasic', 'negative']

    for l, label in enumerate(labels):

        d = data[l]

        x_jitter = add_jitter(center=l, size=np.size(d,0), scale=0.2)

        ax.plot(x_jitter, d, '.', color=colors[c], markersize=markersize)

        # Add a horizontal line for the median
        if use_median:
            hline = np.nanmedian(cluster_data)
        elif not use_median:
            hline = np.nanmean(cluster_data)
        ax.hlines(hline, c-0.2, c+0.2, color='k', linewidth=2)

        # Add a vertical line for the std err
        err = stderr(cluster_data)
        ax.vlines(c, hline-err, hline+err, color='k', linewidth=2)

        ax.set_xticks(range(4), ['early','late','biphasic','negative'])

def write_spike_audio(self, start=0):
    units = self.cells.index.values
    # timerange
    tr = [start, start+15]
    sp = np.array(self.cells.at[units[self.highlight_neuron],'spikeT']) - tr[0]
    sp = sp[sp>0]
    datarate = 30000
    # compute waveform samples
    tmax = tr[1] - tr[0]
    t = np.linspace(0, tr[1]-tr[0], (tr[1]-tr[0])*self.ephys_samprate, endpoint=False)
    x = np.zeros(np.size(t))
    for spt in sp[sp<tmax]:
        x[np.int64(spt*self.ephys_samprate) : np.int64(spt*self.ephys_samprate +30)] = 1
        x[np.int64(spt*self.ephys_samprate)+31 : np.int64(spt*self.ephys_samprate +60)] = -1
    # write the samples to a file
    self.diagnostic_audio_path = os.path.join(self.recording_path, (self.recording_name+'_unit'+str(self.highlight_neuron)+'.wav'))
    wavio.write(self.diagnostic_audio_path, x, self.ephys_samprate, sampwidth=1)

def merge_aud_vid(self):
        merge_mp4_name = os.path.join(self.recording_path, (self.recording_name+'_unit'+str(self.highlight_neuron)+'_merge.mp4'))
        subprocess.call(['ffmpeg', '-i', self.diagnostic_video_path, '-i', self.diagnostic_audio_path, '-c:v', 'copy', '-c:a', 'aac', '-y', merge_mp4_name])

def summary_fig(self, hist_dt=1):
        hist_t = np.arange(0, np.max(self.worldT), hist_dt)

        plt.subplots(self.n_cells+3, 1,figsize=(12, int(np.ceil(self.n_cells/2))))

        if not self.fm:
            # running speed
            plt.subplot(self.n_cells+3, 1, 1)
            plt.plot(self.ballT, self.ball_speed, 'k')
            plt.xlim(0, np.max(self.worldT)); plt.ylabel('cm/sec'); plt.title('running speed')
        elif self.fm:
            plt.subplot(self.n_cells+3, 1, 1)
            plt.plot(self.topT, self.top_speed, 'k')
            plt.xlim(0, np.max(self.worldT)); plt.ylabel('cm/sec'); plt.title('running speed')
        
        # pupil diameter
        plt.subplot(self.n_cells+3, 1, 2)
        plt.plot(self.eyeT, self.longaxis, 'k')
        plt.xlim(0, np.max(self.worldT)); plt.ylabel('pxls'); plt.title('pupil radius')
        
        # worldcam contrast
        plt.subplot(self.n_cells+3, 1, 3)
        plt.plot(self.worldT, self.contrast)
        plt.xlim(0, np.max(self.worldT)); plt.ylabel('contrast a.u.'); plt.title('contrast')
        
        # raster
        for i, ind in enumerate(self.cells.index):
            rate, bins = np.histogram(self.cells.at[ind,'spikeT'], hist_t)
            plt.subplot(self.n_cells+3, 1, i+4)
            plt.plot(bins[0:-1], rate, 'k')
            plt.xlim(bins[0], bins[-1]); plt.ylabel('unit ' + str(ind))

        plt.tight_layout()
        if self.figs_in_pdf:
            self.detail_pdf.savefig(); plt.close()
        elif not self.figs_in_pdf:
            plt.show()