import os, json

def read(path=None):

    if path is None:
        head, tail = os.path.split(os.path.abspath(__file__))
        path = os.path.join(head, 'pipeline.yml')

    with open(path, 'r') as fp:
        internals_dict = json.load(fp)

    return internals_dict

def make_cfg(cfg, intopts):
    """
    Read in intopts
    Merge internals options dictionary (intopts) into config dict (cfg)
    """
    cfg = {**cfg, **intopts}

    return cfg

def fill_cfg(cfg, internals_path=None):
    """
    
    """
    if internals_path is None:
        utils_dir, _ = os.path.split(__file__)
        src_dir, _ = os.path.split(utils_dir)
        repo_dir, _ = os.path.split(src_dir)
        internals_path = os.path.join(repo_dir, 'config/internals.yml')

    with open(internals_path, 'r') as fp:
        internals = yaml.load(fp)
    
    # Fill in internal values
    missing = [k for k in internals.keys() if k not in cfg.keys()]