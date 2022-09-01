"""Basic functions.
"""
import os
import yaml

import itertools
import numpy as np
from datetime import datetime

def find_index_in_list(a, subset):
    """Find the indexes of values in a list.

    Parameters
    --------
    a : list
        Values
    subset : list
        Values that may appear in `a`. This should have an equal or
        smaller number of values than `a`. It is okay for values in
        `subset` to not appear in `a` (these will not be given an index).
    
    Returns
    --------
    (idx, )
    """
    if not subset:
        return
    subset_len = len(subset)
    first_val = subset[0]
    for idx, item in enumerate(a):
        if item == first_val:
            if a[idx:idx+subset_len] == subset:
                yield tuple(range(idx, idx+subset_len))

def stderr(A):
    """Standard error.
    """
    return np.std(A) / np.sqrt(np.size(A))

def z_score(A):
    """Z-score.
    """
    return (np.max(np.abs(A))-np.mean(A)) / np.std(A)

def modind(a, b):
    """Modulation index.

    A positive modulation index indicates a preference
    for `a` over `b`. If one of the inputs is by definition
    preferred, it should be used as `a` and the non-preferred
    value should be `b`.
    """
    modind_val = (a - b) / (a + b)
    return modind_val

def nearest_ind(val, a):
    """Approximate the position of a value in an array.

    This is useful if `a` is an array of timestamps and you want the
    frame closest to the time `val`.
    
    Parameters
    ----------
    val : float
        Single value which falls somewhere between minimum and maximum
        values in `a`.
    a : np.array
        Array of values.
    
    Returns
    -------
    ind : int
        Position of a value in `a` which is the closest to `val` of all
        values in `a`.

    """
    ind = np.argmin(np.abs(a - val))
    return ind

def str_to_bool(value):
    """Parse str as bool type.

    Can be helpful when turning an argparse input into a bool.
    """
    if isinstance(value, bool):
        return value
    if value.lower() in {'False', 'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'True', 'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

def probe_to_ch(probe):
    """Get channel info from probe name.

    Parameters
    ----------
    probe : str
        Probe name. This must contain the number of channels/sites in the
        probe as a str (e.g. 64, 128).
    
    Returns
    -------
    numCh : int
        Number of channels/sites.
    chSpacing : int
        Vertical distance between sites in microns.
    """
    if '16' in probe:
        return 16, 25
    if '64' in probe:
        if probe=='DB_P64-8':
            return 64, 25/2
        else:
            return 64, 25
    if '128' in probe:
        return 128, 25

def get_all_cap_combs(s):
    """Get every unique pattern of character capitalization.

    Example
    -------
    For s='word', this would return [word, Word, wOrd, woRd, ... WORD] each
    as a string.
    
    """
    return map(''.join, itertools.product(*zip(s.upper(), s.lower())))

def str_today():
    """Format today's date and time.

    Returns
    -------
    str_date : str
        Current date
        e.g. Aug. 30 2022 -> 083022
    str_time : str
        Current hours and minutes
        e.g. 10:15am -> 10-15

    To do
    -----
      * Is the time in 24-hour time? If not it should be.
    """
    str_date = datetime.today().strftime('%m%d%y')
    str_time = datetime.today().strftime('%H-%M')

    return str_date, str_time