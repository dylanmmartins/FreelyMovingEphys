import numpy as np
import xarray as xr

def drop_nan_along(x, axis=1):
    # axis=1 will drop along columns (i.e. any rows with NaNs will be dropped)
    x = x[~np.isnan(x).any(axis=axis)]
    return x

def add_jitter(center, size, scale=0.2):
    return np.ones(size) + np.random.uniform(center-scale, center+scale, size)

def s2arr(s):
    """ Collapse pd.Series of lists to two dimensions.

    Input should be a pd.Series in which each index in
    the column/series contains a list of values. All lists must be the same length.

    if flat is true
    Return a flattened 1D array of all values in a pandas Series. 
    Can be a Series where each value is a list.

    Parameters
    --------
    s : pd.Series
        Series in which each index is a list as an object. Every index
        must have a list of the same length.
    
    Returns
    --------
    a : np.array
        2D array with axis 0 matching the number of indexes in the Series and
        axis 1 matching the length of the object lists in the input Series.
    """
    a = np.zeros([np.size(s,0), len(s.iloc[0])])
    for i, vals in enumerate(s):
        a[i,:] = vals
    return a
    
def merge_uneven_xr(objs, dim_name='frame'):
        """ Merge a list of xr DataArrays even when their lengths do not match.
        """

        # Iterate through objects
        max_lens = []
        for obj in objs:

            # Get the sizes of the dim, dim_name
            max_lens.append(dict(obj.frame.sizes)[dim_name])

        # Get the smallest of the object's lengths
        set_len = np.min(max_lens)

        # Shorten everything to `set_len`
        even_objs = []
        for obj in objs:

            # Get the length of the current object
            obj_len = dict(obj.frame.sizes)[dim_name]

            # Obj is too long
            if obj_len > set_len:

                # Find how much it needs to be shortened by
                diff = obj_len - set_len
                good_inds = range(0, obj_len-diff)

                # Set the new end
                obj = obj.sel(frame=good_inds)

                # Add to merge list
                even_objs.append(obj)

            # If it is the smallest length or all objects have the same length
            else:

                # Just append it to the list of objects to merge
                even_objs.append(obj)

        # Merge the xr that now have even lengths
        merged_objs = xr.merge(even_objs)

        return merged_objs