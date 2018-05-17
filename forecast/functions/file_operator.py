import numpy as np


def load_str_data(path):
    return np.loadtxt(path, delimiter=',', dtype=bytes).astype(str)
