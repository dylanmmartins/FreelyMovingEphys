"""
__main__.py
"""
from utils.prelim import PrelimWhiteNoise
import PySimpleGUI as sg
import argparse

def make_window(theme):
    sg.theme(theme)
    options_layout =  [[sg.Text('Choose a probe.')],
                       [sg.Combo(values=('default16', 'NN_H16', 'default64', 'NN_H64-LP', 'DB_P64-3', 'DB_P64-8', 'DB_P128-6'), default_value='default16', readonly=True, k='-COMBO-', enable_events=True)],
                       [sg.Text('Choose an ephys .bin file.')],
                       [sg.Button('Open .bin file')]]
    logging_layout = [[sg.Text('Run this module')],
                      [sg.Button('Run module')]]
    layout = [[sg.Text('Preliminary whitenoise receptive field mapping', size=(38, 1), justification='center', font=("Times", 16), relief=sg.RELIEF_RIDGE, k='-TEXT HEADING-', enable_events=True)]]
    layout +=[[sg.TabGroup([[sg.Tab('Options', options_layout),
               sg.Tab('Run', logging_layout)]], key='-TAB GROUP-')]]
    return sg.Window('Preliminary whitenoise receptive field mapping', layout)

def main():
    window = make_window(sg.theme())
    while True:
        event, values = window.read(timeout=100)
        if event == 'Open .bin file':
            bin_path = sg.popup_get_folder('Choose white noise ephys binary file.')
            print('Whitenoise directory: ' + str(bin_path))
        elif event in (None, 'Exit'):
            print('Exiting')
            break
        elif event == 'Run module':
            probe = values['-COMBO-']
            print('Probe: ' + str(probe))
            pwn = PrelimWhiteNoise(bin_path, probe)
            pwn.process()
    window.close()
    exit(0)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bin_path', type=str, default=None)
    parser.add_argument('--probe', type=str, default=None)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    if args.bin_path is None or args.probe is None:
        main()
    else:
        