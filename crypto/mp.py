from multiprocessing import Pool
import pandas as pd

def offset_and_rolling_lag_corr(series1, series2, offset_lag, rolling_lag):
    """
    Inputs:
    - series1: series 1 who's current day values have their rolling correlation taken with series2's future lagged values - the hypothethical independent variable
    - series2: the series which has it's future values correlated with series1's past values - the hypothetical dependent variable
    Outputs:
    - lagged rolling correlation with series 1 current day values correlated with series 2 future lagged values
    """
    pair = pd.concat([series1, series2], axis=1)
    pair.iloc[:, 0] = pair.iloc[:, 0].shift(offset_lag)
    lagged_rcorr = pair.iloc[:, 0].rolling(rolling_lag).corr(pair.iloc[:, 1])
    lagged_rcorr = lagged_rcorr.dropna()
    return lagged_rcorr

if __name__ == '__main__':
    with Pool(5) as p:
        print(p.map(f, [1, 2, 3]))