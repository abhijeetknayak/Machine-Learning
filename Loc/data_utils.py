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
    print(out.shape)

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





