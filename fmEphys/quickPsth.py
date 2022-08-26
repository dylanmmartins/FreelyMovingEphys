import argparse, os, json

import fmEphys

def main():


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default=None)
    parser.add_argument('--use', type=str, choices=['stim','ttl','loop_ttl'])
    args = parser.parse_args()

    fmEphys.utils.video.calc_dStim(vidpath, savepath)
