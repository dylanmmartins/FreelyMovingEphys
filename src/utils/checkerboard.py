def analyze_checkerboard():
 
      
      
       vid = self.Rc_world.WORLD_video.values.astype(np.uint8)
       worldT = self.Rc_world.timestamps.values
       eyeT = self.Rc_ephys['Rc_eyeT'].iloc[0]
       ephysT0 = self.Rc_ephys['t0'].iloc[0]
 
       criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
       num_frames = np.size(vid, 0); vid_width = np.size(vid, 1); vid_height = np.size(vid, 2)
       kmeans_input = vid.reshape(num_frames, vid_width*vid_height)
       _, labels, _ = cv2.kmeans(kmeans_input.astype(np.float32), 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
       label_diff = np.diff(np.ndarray.flatten(labels))
 
       stim_state = interp1d(worldT[:-1]-ephysT0, label_diff, bounds_error=False)(eyeT)
       eventT = eyeT[np.where((stim_state<-0.1)+(stim_state>0.1))]
 
       Rc_psth = np.zeros([len(self.Rc_ephys.index.values), 2001]) # shape = [unit#, time]
       for i, ind in tqdm(enumerate(self.Rc_ephys.index.values)):
           unit_spikeT = self.Rc_ephys.loc[ind, 'spikeT']
           if len(unit_spikeT)<10: # if a unit never fired during revchecker
               continue
           Rc_psth[i,:] = calc_kde_sdf(unit_spikeT, eventT)
 
       self.Rc_psth = Rc_psth