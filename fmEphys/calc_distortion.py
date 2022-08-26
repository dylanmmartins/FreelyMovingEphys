# calculate distortion in a camera image
import argparse, os, json
import PySimpleGUI as sg

import fmEphys

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--vid', type=str, default=None)
    parser.add_argument('-s', '--save', type=str, default=None)
    args = parser.parse_args()

    if args.vid is not None and args.save is not None:
        vidpath = args.vid
        savepath = args.save
    else:
        vidpath = sg.popup_get_file('Checkerboard video file')
        savepath = sg.popup_get_folder('Save folder')

    if '.npz' not in savepath:
        str_date, _ = fmEphys.utils.base.str_today()
        savepath = os.path.join(savepath, 'cam_mtx_{}.npz'.format(str_date))

    fmEphys.utils.video.calc_distortion(vidpath, savepath)
