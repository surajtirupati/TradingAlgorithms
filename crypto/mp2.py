from multiprocessing import Pool
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from yfinance import Ticker

def get_data(ticker):
    """
    Inputs:
    - ticker: string of ticker
    Outputs:
    - df: dataframe containing close price
    """
    df = Ticker(ticker).history(period="1y")
    return df

def get_close_data(ticker):
    """
    Inputs:
    - ticker: string of ticker
    Outputs:
    - df: dataframe containing close price
    """
    df = Ticker(ticker).history(period="1y")["Close"].rename(ticker)
    return df

def get_close_df(list_of_tickers):
    df_close = pd.DataFrame()

    for i in range(0, len(list_of_tickers)):
        df = get_close_data(list_of_tickers[i])
        df_close = pd.concat([df_close, df], axis=1)

    return df_close.dropna(axis=1)

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


def fit_sin(tt, yy):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c):  return A * np.sin(w*t + p) + c
    popt, pcov = curve_fit(sinfunc, tt, yy, p0=guess, maxfev=5000)
    A, w, p, c = popt
    f = w/(2.*np.pi)
    fitfunc = lambda t: A * np.sin(w*t + p) + c
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}


def fit_sinusoid_to_correlation(correlation):
    """
    Inputs:
    - correlation: lagged rolling correlation
    Outputs:
    - res: dictionary holding the result
    """

    tt = np.arange(0, len(correlation), 1)
    res = fit_sin(tt, correlation)

    return res

def fitted_sine_directional_acc(correlation, sine):
    """
    Inputs:
    - correlation: lagged rolling correlation dataframe
    - sine: fitted sinusoid
    Outputs:
    - sign_accuracy: directional accuracy % between fitted sine wave and correlation
    """

    corr_diff_sign = np.sign(correlation.diff().dropna())

    sine_diff_sign = np.sign(np.diff(sine))

    tmp = corr_diff_sign - sine_diff_sign

    sign_accuracy = len(tmp[tmp == 0]) / len(tmp)

    return sign_accuracy

def high_corr_perc(correlation, thresh):
    """
    Inputs:
    - correlation: lagged rolling correlation dataframe
    - thresh: absolute correlation threshold
    Outputs:
    - perc: percentage greater than input threshold
    """
    perc = len(correlation[abs(correlation) > thresh]) / len(correlation)

    return perc

def process(price_df, indep, dep, offset_lag, rolling_lag, thresh):

    # retrieve pandas series from price_df
    predictor = price_df[indep]
    predictee = price_df[dep]

    # obtain lagged rolling correlation
    correlation = offset_and_rolling_lag_corr(predictor, predictee, offset_lag, rolling_lag)

    # fit sinusoid and obtain result object
    fit_res = fit_sinusoid_to_correlation(correlation)

    # calculate period of fitted sinusoid
    period = int((2*np.pi)/fit_res["omega"])

    # obtaining the directional accuracy of the fit sinusoid
    tt = np.arange(0, len(correlation), 1)
    dir_acc = fitted_sine_directional_acc(correlation, fit_res["fitfunc"](tt))

    # obtaining percentage of correlations with large magnitude
    large_corr_percentage = high_corr_perc(correlation, thresh)

    return [period, dir_acc, large_corr_percentage]

