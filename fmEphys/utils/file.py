import xarray as xr
from scipy.io import savemat
import os, json
import cv2
import pandas as pd
import numpy as np
import PySimpleGUI as sg
import h5py

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

def read_dlc_positions(path, multianimal=False, split_xyl=False):

    pts = pd.read_hdf(path)

    if multianimal is False:
        # Organize columns
        pts.columns = [' '.join(col[:][1:3]).strip() for col in pts.columns.values]
        pts = pts.rename(columns={pts.columns[n]: pts.columns[n].replace(' ', '_') for n in range(len(pts.columns))})

    elif multianimal is True:
        pts.columns = ['_'.join(col[:][1:]).strip() for col in pts.columns.values]

    if not split_xyl:
        return pts

    elif split_xyl:
        x_pos = pd.Series([])
        y_pos = pd.Series([])
        likeli = pd.Series([])

        for col in pts.columns.values:
            if '_x' in col:
                x_pos = pd.concat([x_pos, pts.loc[col]])
            elif '_y' in col:
                y_pos = pd.concat([y_pos, pts.loc[col]])
            elif 'likeli' in col:
                likeli = pd.concat([likeli, pts.loc[col]])
        return x_pos, y_pos, likeli