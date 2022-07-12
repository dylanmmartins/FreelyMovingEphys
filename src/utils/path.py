import os
import fnmatch

def find(pattern, path):
    """ Glob for subdirectories.

    Parameters
    --------
    pattern : str
        str with * for missing sections of characters
    path : str
        path to search, including subdirectories
    
    Returns
    --------
    result : list
        list of files matching pattern.
    """
    result = []

    # Walk though the path directory and files
    for root, _, files in os.walk(path):

        # Walk to the file in the directory
        for name in files:

            # If the file matches the filetype, append it to the list
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root ,name))

    return result # return full list of file of a given type


def list_subdirs(rootdir, keep_parent=False):
    """ List subdirectories in a root directory.

    without keep_parent, the subdirectory itself is named
    with keep_parent, the subdirectory will be returned *including* its parent path
    """
    paths = []; names = []
    for item in os.scandir(rootdir):
        if os.path.isdir(item):
            if item.name[0]!='.':
                paths.append(item.path)
                names.append(item.name)

    if keep_parent:
        return paths
    elif not keep_parent:
        return names


def make_recording_name(path):
    """ Parse file names in recording path to build name of the recording.

    Parameters
    --------
    recording_path : str
        Path to the directory of one recording. Must be stimulus-specific.
        e.g. D:/path/to/animal/hf1_wn
    
    Returns
    recording_name : str
        Name of recording from a specific stimulus.
        e.g. 010101_animal_Rig2_control_hf1_wn
    """

    ignore = ['plot','IR','rep11','betafpv','side_gaze','._']

    fs = find('*.avi', path)
    filt = [f for f in fs if all(b not in f for b in ignore)][0]

    _, tail = os.path.split(filt)
    name_noext, _ = os.path.splitext(tail)
    split_name = name_noext.split('_')[:-1]
    name = '_'.join(split_name)
    
    return name

