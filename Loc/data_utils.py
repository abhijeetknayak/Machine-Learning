import pandas as pd
import numpy as np
import glob

def load_data(path):
    path = path + "/*.csv"
    out = None

    for file in glob.glob(path):
        if out is None:
            out = pd.read_csv(file, header=None)
        else:
            df = pd.read_csv(file, header=None)
            out = pd.concat((out, df))
    out.columns = ["NumLanes", "LnChg", "OlrValid", "LnWidth", "DistL", "DistR",
                   "DistL", "DistR", "DistL", "DistR", "DistL", "DistR", "DistL", "DistR", "L2", "L1",
                   "R2", "R1", "LN1", "RN1", "Id"]
    print(out.shape)



