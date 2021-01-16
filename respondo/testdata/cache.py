import os
import pandas as pd
import numpy as np


cases = {
    "h2o_sto3g_adc1": 10,
    "h2o_sto3g_adc2": 65,
    "h2o_ccpvdz_adc1": 95,
    # "h2o_ccpvdz_adc2": 4655,
    "formaldehyde_sto3g_adc1": 32,
    "formaldehyde_sto3g_adc2": 560,
}


class MockExcitedStates:
    def __init__(self, dataframe):
        self.df = dataframe
        carts = ["_x", "_y", "_z"]
        for col in self.df.columns:
            if any(c in col for c in carts):
                fld = "_".join(col.split("_")[:-1])
                x = getattr(self.df, fld + "_x")
                y = getattr(self.df, fld + "_y")
                z = getattr(self.df, fld + "_z")
                assembled = np.vstack(
                    (x.values, y.values, z.values)
                ).T
                setattr(self, fld, assembled)
            else:
                setattr(self, col, getattr(self.df, col).values)


def read_full_diagonalization():
    ret = {}
    for case in cases:
        thisdir = os.path.dirname(__file__)
        json_file = os.path.join(thisdir, f"{case}.json")
        if not os.path.isfile(json_file):
            continue
        df = pd.read_json(json_file)
        ret[case] = MockExcitedStates(df)
    return ret


data_fulldiag = read_full_diagonalization()
