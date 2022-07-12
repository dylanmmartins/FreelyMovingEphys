import xarray as xr
from scipy.io import savemat
import os, json
import cv2
import pandas as pd
import numpy as np
import PySimpleGUI as sg
import h5py

def read_ephys_binary(path, n_ch, probe_name=None, chmap_path=None):
    """ Read in ephys binary and remap channels.

    Parameters:
    if a probe name is given, the binary file will be remapped. otherwise, channels will be kept in the same order

    Returns:
    ephys (pd.DataFrame): ephys data with shape (time, channel)
    """
    # set up data types
    dtypes = np.dtype([('ch'+str(i),np.uint16) for i in range(0,n_ch)])
    # read in binary file
    ephys_arr = pd.DataFrame(np.fromfile(path, dtypes, -1, ''))
    if probe_name is not None:
        # open channel map file
        if chmap_path is None:
            chmap_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),'/config/channel_maps.json')
        with open(chmap_path, 'r') as fp:
            all_maps = json.load(fp)
        # get channel map for the current probe
        ch_map = all_maps[probe_name]
        # remap with known order of channels
        ephys_arr = ephys_arr.iloc[:,[i-1 for i in list(ch_map)]]
    return ephys_arr

    import xarray as xr

def nc2mat(f=None):
    """
    If an nc file path isn't given, a dialog box will open.
    """
    if f is None:
        f = sg.popup_get_file('Choose .nc file.')
    data = xr.open_dataset(f)
    data_dict = dict(zip(list(data.REYE_ellipse_params['ellipse_params'].values), [data.REYE_ellipse_params.sel(ellipse_params=p).values for p in list(data.REYE_ellipse_params['ellipse_params'].values)]))
    save_name = os.path.join(os.path.split(f)[0], os.path.splitext(os.path.split(f)[1])[0])+'.mat'
    print('saving {}'.format(save_name))
    savemat(save_name, data_dict)

def avi_to_arr(path, ds=0.25):
    vid = cv2.VideoCapture(path)
    # array to put video frames into
    # will have the shape: [frames, height, width] and be returned with dtype=int8
    arr = np.empty([int(vid.get(cv2.CAP_PROP_FRAME_COUNT)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)*ds),
                        int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)*ds)], dtype=np.uint8)
    # iterate through each frame
    for f in range(0,int(vid.get(cv2.CAP_PROP_FRAME_COUNT))):
        # read the frame in and make sure it is read in correctly
        ret, img = vid.read()
        if not ret:
            break
        # convert to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # downsample the frame by an amount specified in the config file
        img_s = cv2.resize(img, (0,0), fx=ds, fy=ds, interpolation=cv2.INTER_NEAREST)
        # add the downsampled frame to all_frames as int8
        arr[f,:,:] = img_s.astype(np.int8)
    return arr

def save_dict_to_group(h5file, path, dic):
    """ Recursively save dictionary contents to group.
    """
    if isinstance(dic,dict):
        iterator = dic.items()
    elif isinstance(dic,list):
        iterator = enumerate(dic)
    else:
        ValueError('Cannot save %s type' % type(dic))

    for key, item in iterator:
        if isinstance(dic,list):
            key = str(key)
        if isinstance(item, (np.ndarray, np.int64, np.float64, int, float, str, bytes)):
            h5file[path + key] = item
        elif isinstance(item, dict) or isinstance(item,list):
            save_dict_to_group(h5file, path + key + '/', item)
        else:
            raise ValueError('Cannot save %s type'%type(item))

def load_dict_from_group(h5file, path):
    """ Recurively load dictionary contents from group.
    """
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item[()]
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = load_dict_from_group(h5file, path + key + '/')
    return ans

def write_h5(filename, dic):
    """
    Saves a python dictionary or list, with items that are themselves either
    dictionaries or lists or (in the case of tree-leaves) numpy arrays
    or basic scalar types (int/float/str/bytes) in a recursive
    manner to an hdf5 file, with an intact hierarchy.
    """
    with h5py.File(filename, 'w') as h5file:
        save_dict_to_group(h5file, '/', dic)

def read_h5(filename, ASLIST=False):
    """
    Default: load a hdf5 file (saved with io_dict_to_hdf5.save function above) as a hierarchical
    python dictionary (as described in the doc_string of io_dict_to_hdf5.save).
    if ASLIST is True: then it loads as a list (on in the first layer) and gives error if key's are not convertible
    to integers. Unlike io_dict_to_hdf5.save, a mixed dictionary/list hierarchical version is not implemented currently
    for .load
    """
    with h5py.File(filename, 'r') as h5file:
        out = load_dict_from_group(h5file, '/')
        if ASLIST:
            outl = [None for l in range(len(out.keys()))]
            for key, item in out.items():
                outl[int(key)] = item
            out = outl
        return out

def read_dlc_positions(path, multianimal=False):

    pts = pd.read_hdf(path)

    if multianimal is False:
        # Organize columns
        pts.columns = [' '.join(col[:][1:3]).strip() for col in pts.columns.values]
        pts = pts.rename(columns={pts.columns[n]: pts.columns[n].replace(' ', '_') for n in range(len(pts.columns))})

    elif multianimal is True:
        pts.columns = ['_'.join(col[:][1:]).strip() for col in pts.columns.values]

    return pts