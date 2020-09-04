import pandas as pd
import numpy as np
import glob

def load_data_df(path):
    """
    Load data as a pandas data frame. This function merges all available csv files into
    a single data frame

    Args: Path to csv files

    Returns : Data frame with 21 columns and 'N' number of rows
    """
    path = path + "/*.csv"
    out = None

    for file in glob.glob(path):
        if out is None:
            out = pd.read_csv(file, header=None)
        else:
            df = pd.read_csv(file, header=None)
            out = pd.concat((out, df))
    # Set header
    out.columns = ["NumLanes", "LnChg", "OlrValid", "LnWidth", "DistL", "DistR",
                   "DistL", "DistR", "DistL", "DistR", "DistL", "DistR", "DistL", "DistR", "L2", "L1",
                   "R2", "R1", "LN1", "RN1", "Id"]
    return out


def load_data_numpy(path):
    """
    load data from csv files into a numpy array. This function merges all csv file data into a
    single array.

    Args: Path to csv files

    Returns : Numpy array of size [N * D]
    """
    path = path + "/*.csv"
    out = None

    for file in glob.glob(path):
        raw_data = open(file, 'r')
        data = np.loadtxt(raw_data, delimiter=',')
        if out is None:
            out = data
        else:
            out = np.vstack((out, data))
    return out[:, 0:20], out[:, 20]


def select_features(X, ks):
    """
    Select features that you want in the training data

    Args: Input data of size [N * D]

    Returns: numpy array of size [N * K]
    """
    out = None
    for k in ks:
        if out is None:
            out = X[:, k]
            out = out[:, np.newaxis]
        else:
            slice = X[:, k]
            slice = slice[:, np.newaxis]
            out = np.hstack((out, slice))
    return out




