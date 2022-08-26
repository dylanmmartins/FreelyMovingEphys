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