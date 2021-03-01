import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

"""This file contains some common financial functions that are used regularly"""

def log_returns(df):
    """Description: Calculates log returns
    Takes input of dataframe containing price column

    Outputs: dataframe column containing log returns
    """
    log_price = np.log(df)
    log_returns = log_price- log_price.shift(1)
    log_returns = log_returns.dropna()
    return log_returns


def resample_data(data, freq, resample_style):
    """Description: Data must be timestamped for function to work:
    Takes inputs of:
    - price data
    - desired frequency to resample to
    - resample method: sample first value on resmapling intervals or take sum inbetween resampling intervals.
    Outputs: resampled dataframe
    """

    if type(data.index) != pd.core.indexes.datetimes.DatetimeIndex:
        raise NotImplementedError("Data index must be in Pandas datetime format. Try converting using 'pd.to_datetime'.")
    else:
        if resample_style == "first":
            df = data
            df = df.resample(freq).first()
            df.dropna()

        elif resample_style == "sum":
            df = data
            df = df.resample(freq).sum()
            df = df[(df != 0).all(1)]

    return df