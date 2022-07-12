# params

from src import video, path

def gather_camera_files(recpath, camname):
    
    # Position data from DLC only exists for the topcam and
    # eyecam. For these cameras, get the .h5 files saved out
    # from DeepLabCut.
    dlc_path = []
    if any(x in camname.lower() for x in ['eye','top']):
        dlc_path = path.find(('*{}*DLC*.h5'.format(camname)), recpath)
    if (type(dlc_path)==list) and (len(dlc_path>1)):
        dlc_path = dlc_path[0]

    # Get the .avi for the camera
    avi_path = path.find(('*{}*.avi'.format(camname)), recpath)
    avi_path = [v for v in avi_path if 'plot' not in v and 'speed_yaw' not in v]
    if 'eye' in camname.lower():
        avi_path = [v for v in avi_path if 'deinter' in v]
    elif 'world' in camname.lower():
        avi_path = [v for v in avi_path if 'calib' in v]
    if (type(avi_path)==list) and (len(avi_path>1)):
        avi_path = avi_path[0]

    # Timestamps, saved out by Bonsai as a .csv
    csv_path = path.find('*{}*BonsaiTS*.csv'.format(camname), recpath)
    csv_path = [v for v in csv_path if 'formatted' not in v]
    if (type(csv_path)==list) and (len(csv_path>1)):
        csv_path = csv_path[0]

def main():
    

    # Ball
    csv_path = find(self.recording_name+'_BALLMOUSE_BonsaiTS_X_Y.csv', self.recording_path)[0]