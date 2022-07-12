# preprocessing e.g. deinterlace, undistort, etc.

import cv2
import os, subprocess
import numpy as np
from tqdm import tqdm

def make_savepath(path, key):
    savedir = os.path.split(path)[0]
    savename = '.'.join((os.path.split(path)[1]).split('.')[:-1])
    savepath = os.path.join(savedir, ('{}_{}.avi'.format(savename, key)))
    return savepath

def deinterlace(path, savepath=None, rotate180=True, expected_fps_in=30):
    """
    `expected_fps` is the expected frame rate of aquisition. deinterlacing should doublet his
    """
    if savepath is None:
        savepath = make_savepath(path, 'deinter')
  
    # Open the video
    cap = cv2.VideoCapture(path)

    # Get total number of frames and the aquisition rate
    fs = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == expected_fps_in:
        if rotate180:
            vf_val = 'yadif=1:-1:0, vflip, hflip, scale=640:480'
        else:
            vf_val = 'yadif=1:-1:0, scale=640:480'
        subprocess.call(['ffmpeg', '-i', path, '-vf', vf_val, '-c:v', 'libx264',
                        '-preset', 'slow', '-crf', '19', '-c:a', 'aac', '-b:a',
                        '256k', '-y', savepath])

def rotate_video(path, savepath=None, h=False, v=False):
    if savepath is None:
        savepath = make_savepath(path, 'rotate')

    if h is True and v is True:
        # horizontal and vertial flip
        vf_val = 'vflip, hflip'
    elif h is True and v is False:
        # horizontal only
        vf_val = 'hflip'
    elif h is False and v is True:
        vf_val = 'vflip'

    # Only do the rotation is at least one axis is being flipped
    if h is True or v is True:
        subprocess.call(['ffmpeg', '-i', path, '-vf', vf_val, '-c:v',
                         'libx264', '-preset', 'slow', '-crf', '19',
                         '-c:a', 'aac', '-b:a', '256k', '-y', savepath])
 
def calc_distortion(path, savepath,
                    board_w=7, board_h=5):
    """
    path is the path to a video of the checkerboard moving (an .avi)
    savepath needs to be a .npz
    """

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # Termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points
    objp = np.zeros((board_h*board_w,3), np.float32)
    objp[:,:2] = np.mgrid[0:board_w,0:board_h].T.reshape(-1,2)

    # Read in video
    calib_vid = cv2.VideoCapture(path)

    nF = int(calib_vid.get(cv2.CAP_PROP_FRAME_COUNT)) # number of frames

    # Iterate through frames
    print('Finding checkerboard corners for {} frames'.format(nF))
    for _ in tqdm(range(nF)):

        # Open frame
        ret, img = calib_vid.read()

        # Make sure the frame is read in correctly
        if not ret:
            break

        # Convert to grayscale
        if img.shape[2] > 1:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # Find the checkerboard corners
        ret, corners = cv2.findChessboardCorners(gray, (board_w,board_h), None)
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    # Calculate the distortion
    print('Calculating distortion (slow)')

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Save the values out
    np.savez(savepath, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

def fix_distortion(path, proppath, savepath=None):

    if savepath is None:
        savepath = make_savepath(path, 'undist')
    
    # Load the camera properties
    camprops = np.load(proppath)

    # Unpack camera properties
    mtx = camprops['mtx']; dist = camprops['dist']
    rvecs = camprops['rvecs']; tvecs = camprops['tvecs']

    # Read in video
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    nF = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Setup the file writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    newvid = cv2.VideoWriter(savepath, fourcc, fps,
                              (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    # Iterate through frames
    for _ in tqdm(range(nF)):

        # Read frame, make sure it opens correctly
        ret, frame = cap.read()
        if not ret:
            break

        # Fix distortion
        f = cv2.undistort(frame, mtx, dist, None, mtx)

        # write the frame to the video
        newvid.write(f)

    newvid.release()

def fix_vid_contrast(path, savepath):

    if savepath is None:
        savepath = make_savepath(path, 'contrast')

    # Read in existing video, get some properties
    vid = cv2.VideoCapture(path)
    nF = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Get the new video set up
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    newvid = cv2.VideoWriter(savepath, fourcc, 60.0, (w, h))

    print('Increasing contrast for {} frames'.format(nF))

    for _ in tqdm(range(nF)):
        
        ret, f = vid.read()
        if not ret:
            break

        # Convert to greyscale
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)

        # compute gamma
        # gamma = log(mid*255)/log(mean)
        mid = 0.5
        mean = np.mean(gray)
        gamma = np.log(mid*255) / np.log(mean)

        # apply gamma correction to frame
        newF = np.power(f, gamma).clip(0, 255).astype(np.uint8)

        # Write frame
        newvid.write(newF)

    newvid.release()