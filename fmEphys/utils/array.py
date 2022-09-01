"""Functions for manipulating arrays.
"""

import numpy as np
import xarray as xr

def drop_nan_along(x, axis=1):
    """Drop all NaNs along one axis of an array.

    Parameters
    ----------
    x : np.array
        Array
    axis : int
        Axis to drop NaNs along.

    Returns
    x : np.array
        Same as input `x` but with all NaNs removed.
    
    """
    # axis=1 will drop along columns (i.e. any rows with NaNs will be dropped)
    x = x[~np.isnan(x).any(axis=axis)]
    return x

def add_jitter(center, size, scale=0.2):
    """Jitter values around a center point.
    
    Parameters
    ----------
    center : int or float
        Values
    size : int
        Number of points to create which are scattered around `center`
    scale : float
        Maximum distance that points can jitter relative to `center` in
        positive or negative direction.

    Returns
    -------
    jittered : np.array
        Jittered values
    """
    jittered = np.ones(size) + np.random.uniform(center-scale, center+scale, size)
    return jittered

def s2arr(s):
    """Collapse Series of lists to 2D array.

    This is for a pd.Series where each index contains a list of values. The list
    should be stored in each index as an object. The list in each index must be
    of equal length. This function will return a 2D array with a shape determined
    by the length of the Series and the length shared by all lists in each index.

    Parameters
    ----------
    s : pd.Series
        Series in which each index is a list as an object. Every index
        must have a list of the same length.

    Returns
    -------
    a : np.array
        2D array with axis 0 matching the number of indexes in the Series and
        axis 1 matching the length of the lists in the input Series.
    """
    # Only check the length of the list in index 0...
    # If the list lendths are not consistant, this will error
    a = np.zeros([np.size(s,0), len(s.iloc[0])])
    for i, vals in enumerate(s):
        a[i,:] = vals
    return a
    
def merge_uneven_xr(objs, dim_name='frame'):
    """Merge DataArrays of unequal lengths.

    For two or more xarray DataArrays with the same named dimension which
    is of nearly the same length. This will compare their lengths and
    shorten the length to match the shortest dimention length of the
    DataArrays that were compared.

    Do not use this if the lengths are off by significant size. This is
    best used for cases when the length is rarely different and in those
    cases is different by a small amount.

    Once the lengths are corrected, they will be merged into a
    single xr.Dataset

    Parameters
    ----------
    objs : list
        List of xr.DataArray
    dim_name : str
        Name of the DataArray dimension along which lengths should
        be compared and shortened.

    Returns
    -------
    mergered_objs : xr.Dataset
        All data mergered together into a single Dataset.
    """
    # Check lengths
    max_lens = []
    for obj in objs:
        max_lens.append(dict(obj.frame.sizes)[dim_name])
    
    # Use the smallest
    set_len = np.min(max_lens)

    # Shorten everything to `set_len`
    even_objs = []
    for obj in objs:
        obj_len = dict(obj.frame.sizes)[dim_name]

        if obj_len > set_len: # if this one is too long

            # Find how much it needs to be shortened by
            diff = obj_len - set_len
            good_inds = range(0, obj_len-diff)

            # Set the new end
            obj = obj.sel(frame=good_inds)

            even_objs.append(obj)

        # If it is the smallest length or all objects have the same length,
        # just append it to the list of objects to merge
        else:
            even_objs.append(obj)

    # Merge the xr which now have equal lengths
    merged_objs = xr.merge(even_objs)

    return merged_objs