"""

LCPOM: useful I/O utilities

"""

# :: Python Standar Library imports
import numpy as np
import os
import matplotlib.pyplot as plt

# :: Local imports from LCPOM

# :: Function definitions

def load( in_path, *_args, **_kws):
    """
    Load files

    Arguments:
        in_path (str): input file path

    Returns:
        arr (np.ndarray): array with data from file.
        *_args: Positional arguments for read.
        **_kws: Keyword arguments for read.
    """
    obj = load(in_path)

    arr = obj.get_data()

    return arr

def save( out_path, arr, *_args, **_kws):
    """
    Save file
    If the output path doesn't exist, saving fails.
    
    Args:
        out_path (str): output file path
        arr (np.ndarray): data to be stored
        *_args: positional arguments for save
        **_kws: keyword arguments for save

    Returns:
        none
    """

    _args = tuple(_args) if _args else ()
    _kws = dict(_kws)

    
def num_to_mode (num):
    """
    Parse calculation mode to str
    """

    if (num == 1):
        return "Single-wavelength"
    if (num ==2):
        return "Simp-color"
    if (num == 3):
        return "Full-color"

    else: return 0

