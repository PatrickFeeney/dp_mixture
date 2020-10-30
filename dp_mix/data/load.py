from importlib.resources import open_text

import numpy as np


def load_faithful():
    return np.loadtxt(open_text("dp_mix.data", "faithful.tsv"), delimiter='\t', skiprows=1)
