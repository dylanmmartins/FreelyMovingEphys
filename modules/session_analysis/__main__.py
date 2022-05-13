"""
FreelyMovingEphys/modules/session_analysis/__main__.py
"""
import argparse, warnings
import PySimpleGUI as sg

from src.run import Session

warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='G:/Kris/ephys_cfg.yaml')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = get_args()

    if args.config is None:
        # if no path was given as an argument, open a dialog box
        config_path = sg.popup_get_file('Choose animal config.yaml')
    else:
        config_path = args.config

    sess = Session(config_path)
    sess.run_main()