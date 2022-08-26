import os, yaml
import itertools
import numpy as np
from datetime import datetime

def find_index_in_list(a, subset):
    """
    Parameters
    --------
    a : list
        list of values
    subset : list
        list of values shorter than a, which may exist in a
    
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

def stderr(a):
    return np.std(a) / np.sqrt(np.size(a))

def z_score(A):
    return (np.max(np.abs(A))-np.mean(A)) / np.std(A)

def modind(a, b):
    """
    a should be pref, b should be the nonpref
    """

    modind_val = (a - b) / (a + b)
    return modind_val

def nearest_ind(val, a):
    """
    val is a single value, float or something

    a is an array of values
    """
    ind = np.argmin(np.abs(a - val))
    return ind

def str_to_bool(value):
    """ Parse strings to read argparse flag entries in as bool.
    
    Parameters
    --------
    value : str
        Input value.

    Returns
    --------
    bool
    """
    if isinstance(value, bool):
        return value
    if value.lower() in {'False', 'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'True', 'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

def probe_to_ch(probe):
    """
    returns number of channels and channel spacing (um)
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
    return map(''.join, itertools.product(*zip(s.upper(), s.lower())))

def fill_cfg(cfg, internals_path=None):
    if internals_path is None:
        utils_dir, _ = os.path.split(__file__)
        src_dir, _ = os.path.split(utils_dir)
        repo_dir, _ = os.path.split(src_dir)
        internals_path = os.path.join(repo_dir, 'config/internals.yml')

    with open(internals_path, 'r') as fp:
        internals = yaml.load(fp)
    
    # Fill in internal values
    missing = [k for k in internals.keys() if k not in cfg.keys()]

def str_today():
    str_date = datetime.today().strftime('%m%d%y')
    str_time = datetime.today().strftime('%H-%M')

    return str_date, str_time