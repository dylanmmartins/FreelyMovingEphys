import argparse, os, yaml
import PySimpleGUI as sg
from datetime import datetime

import fmEphys

def get_rdict(cfg):
    """Make dict of recording paths.

    Args:
        cfg:

    Returns:

    """

    rname_list = fmEphys.utils.path.list_subdirs(cfg, name_only=True)

    if cfg['use_recordings'] != []:
        rname_list = [name for name in rname_list if name in cfg['use_recordings'] ]
    if cfg['ignore_recordings'] != []:
        rname_list = [name for name in rname_list if name not in cfg['ignore_recordings'] ]

    rpath_list = [os.path.join(cfg['apath'], name) for name in rname_list]

    # sort dictionary of {name: path} so freely-moving recordings are always handled first
    rdict = dict(zip(rname_list, rpath_list))
    sorting_key = sorted(rdict, key=lambda x:('fm' not in x, x))
    rdict = dict(zip(sorting_key, [rdict[k] for k in sorting_key]))

    return rdict

def get_ddict(cfg):
    """Make dict of data input paths.

    Args:
        cfg:

    Returns:

    """

    ddict = {
        'reye': None,
        'leye': None,
        'world': None,
        'top1': None,
        'top2': None,
        'top3': None,
        'side': None
    }

    # Fill in the camera names
    for cam in ddict.keys():
        # try combinations of lower/upper caps
        for test_c in fmEphys.utils.base.get_all_caps(cam):
            # if an avi video exists, save that as the name
            test_list = fmEphys.utils.path.find('{}_{}.avi'.format(cfg['rfname'], test_c), cfg['rpath'])
            if len(test_list) > 0:
                # Add the name (e.g. 'Reye')
                ddict[cam]['name'] = test_c
                # Add the .avi video
                avi_paths = fmEphys.utils.path.find('{}_{}.avi'.format(cfg['rfname'], test_c), cfg['rpath'])
                temp_path = [p for p in avi_paths if any(['deinter','calib'] not in p)]
                if len(temp_path) > 0:
                    ddict[cam]['raw_avi'] = fmEphys.utils.path.most_recent(temp_path)
                temp_path[0]
                dlc_path = fmEphys.utils.path.find('*{}*DLC*.h5'.format(cfg['dname']), cfg['rpath'])
                dlc_path = fmEphys.utils.path.most_recent(dlc_path)
                # Add the .csv timestamps


                continue

    # if the imu file exists
    ddict['imu'] = None
    if len(fmEphys.utils.path.find('{}_IMU.bin'.format(cfg['rfname']), cfg['rpath'])) > 0:
        ddict['imu'] = True

    # if the treadmill file exists
    ddict['treadmill'] = None
    if len(fmEphys.utils.path.find('*BALLMOUSE_BonsaiTS_X_Y.csv'.format(cfg['rfname']), cfg['rpath'])) > 0:
        ddict['treadmill'] = True

    if cfg['use_data'] != []:
        bad = [b for b in ddict.keys() if b not in cfg['use_data']]
        for b in bad:
            del ddict[b]
    if cfg['ignore_data'] != []:
        bad = [b for b in ddict.keys() if b in cfg['ignore_data']]
        for b in bad:
            del ddict[b]

    return ddict
    
def main(cfg):

    rdict = get_rdict(cfg)

    for rname, rpath in rdict.items():

        cfg['rname'] = rname
        cfg['rpath'] = rpath

        if cfg['delete_dlc_files']:
            fmEphys.utils.path.delete_dlc_files(cfg['rpath'])

        # Recording file base name
        cfg['rfname'] = fmEphys.utils.path.get_rfname(cfg['rpath'])

        # Get a list of the data inputs
        # i.e. is there a worldcam, eyecam, topcam, imu, treadmill, etc.?
        ddict = get_ddict(cfg)

        ### Deinterlace
        if cfg['steps']['deinterlace']:

            if ddict['reye'] is not None:

                fmEphys.utils.video.deinterlace(cfg, path, rotate=cfg['rotate_eyecam'])



            if ddict['reye'] is not None:
                cfg['dname'] = ddict['reye']
                utils.pupil.calc_theta_phi(cfg)

            if ddict['world'] is not None:
                cfg['dname'] = ddict['world']



            

            # eyecam

            utils.video.deinterlace(eye_avi, rotate180=cfg['rotate_eyecam'])

            # worldcam


        ### Behavior preprocessing

        if cfg['steps']['behavior']:

            # IMU
            utils.imu.preprocess_IMU(cfg)
            if cfg['includes_TTL']:
                utils.imu.preprocess_TTL(cfg)

            # Treadmill
            utils.treadmill.preprocess_treadmill(cfg)



        


            # get a list of cameras in the current recording
            recording_cams = []
            for p in ['REYE','LEYE','Reye','Leye','Side','SIDE','TOP1','TOP2','TOP3','WORLD','World']:
                date_str = recording_name.split('_')[0]
                animal_str = recording_name.split('_')[1]
                rec_str = recording_name.split('_')[3]
                if find(recording_name+'_'+p+'.avi', recording_path) != []:
                    recording_cams.append(p)
                elif self.config['internals']['eye_corners_first'] and (find('{}_{}_*_{}_{}.avi'.format(date_str, animal_str, rec_str, p), recording_path) != []):
                    recording_cams.append(p)

            for camname in recording_cams:
                if camname.lower() in ['reye','leye']:
                    print(recording_name + ' for input: ' + camname)
                    ec = Eyecam(self.config, recording_name, recording_path, camname)
                    ec.safe_process(show=True)
                elif camname.lower() in ['world']:
                    print(recording_name + ' for input: ' + camname)
                    wc = Worldcam(self.config, recording_name, recording_path, camname)
                    wc.safe_process(show=True)
                elif camname.lower() in ['top1','top2','top3'] and 'dark' not in recording_name:
                    print(recording_name + ' for input: ' + camname)
                    tc = Topcam(self.config, recording_name, recording_path, camname)
                    tc.safe_process(show=True)
                elif camname.lower() in ['side']:
                    sc = Sidecam(self.config, recording_name, recording_path, camname)
                    sc.safe_process(show=True)
            if find(recording_name+'_IMU.bin', recording_path) != []:
                print(recording_name + ' for input: IMU')
                imu = Imu(self.config, recording_name, recording_path)
                imu.process()
            if find(recording_name+'_BALLMOUSE_BonsaiTS_X_Y.csv', recording_path) != []:
                print(recording_name + ' for input: head-fixed running ball')
                rb = RunningBall(self.config, recording_name, recording_path)
                rb.process()

    def ephys_analysis(self):
        self.get_session_recordings()
        for _, recording_path in self.recordings_dict.items():
            recording_name = auto_recording_name(recording_path)
            if ('fm' in recording_name and 'light' in recording_name) or ('fm' in recording_name and 'light' not in recording_name and 'dark' not in recording_name):
                ephys = FreelyMovingLight(self.config, recording_name, recording_path)
                ephys.analyze()
            elif 'fm' in recording_name and 'dark' in recording_name:
                ephys = FreelyMovingDark(self.config, recording_name, recording_path)
                ephys.analyze()
            elif 'wn' in recording_name:
                ephys = HeadFixedWhiteNoise(self.config, recording_name, recording_path)
                ephys.analyze()
            elif 'grat' in recording_name:
                ephys = HeadFixedGratings(self.config, recording_name, recording_path)
                ephys.analyze()
            elif 'sp' in recording_name and 'noise' in recording_name:
                ephys = HeadFixedSparseNoise(self.config, recording_name, recording_path)
                ephys.analyze()
            elif 'revchecker' in recording_name:
                ephys = HeadFixedReversingCheckboard(self.config, recording_name, recording_path)
                ephys.analyze()

    def run_main(self):
        if self.config['main']['deinterlace'] or self.config['main']['undistort'] or self.config['main']['pose_estimation'] or self.config['main']['parameters']:
            self.preprocessing()
        if self.config['main']['ephys']:
            self.ephys_analysis()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str)
    args = parser.parse_args()

    sg.theme('Default1')
    if args.cfg is None:
        # if no path was given as an argument, open a dialog box
        cfg_path = sg.popup_get_file('Choose animal ephys_cfg.yaml')
    else:
        cfg_path = args.cfg

    with open(cfg_path, 'r') as infile:
        cfg = yaml.load(infile, Loader=yaml.FullLoader)

    date_str, time_str = utils.base.str_today()
    log_path = os.path.join(cfg['apath'], 'errlog_{}_{}.txt'.format(date_str, time_str))
    logging = utils.log.Log(log_path)

    main(cfg)