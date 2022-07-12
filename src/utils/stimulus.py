import os

from src.path import make_recording_name
from src.file import read_h5, write_h5


# ephys analysis that isn't stimulus-specific. just aligning everything.

def read_worldcam_h5()
def read_topcam_h5()
def read_IMU_h5()
def read_treadmill_h5()
def read_ephys_json()
def read_eyecam_h5()

def timing_alignment(ephysT0, eyeT, worldT,
                    topT=None, ballT=None, imuT=None):

    eyeT -= ephysT0
    worldT -= ephysT0

    # 8-hour offset for some data
    if eyeT[0] < -600:
        eyeT += 8*60*60
    if worldT[0] < -600:
        worldT += 8*60*60

    if topT is not None:
        topT = topT - ephysT0
    if ballT is not None:
        ballT = ballT - ephysT0
    if imuT is not None:
        imuT = imuT - ephysT0



def head_fixed(rpath):

    rname = make_recording_name(rpath)

    savepath = os.path.join(rpath, '{}_stim_responses.h5'.format(rname))

    # Sometimes the existing h5 cannot be overwritten, so it's better to delete it.
    if os.path.isfile(savepath):
        os.remove(savepath)

    # Paths of behavior data
    reye_path = os.path.join(rpath, '{}_Reye.h5'.format(rname))
    world_path = os.path.join(rpath, '{}_World.h5'.format(rname))
    treadmill_path = os.path.join(rpath, '{}_Treadmill.h5'.format(rname))

    # Paths of ephys data
    ephys_spike_path = os.path.join(rpath, '{}_ephys_merge.json'.format(rname))
    ephys_binary_path = os.path.join(rpath, '{}_Ephys.bin'.format(rname))
    
    # Open worldcam
