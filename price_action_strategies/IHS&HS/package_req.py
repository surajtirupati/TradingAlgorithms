from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from statsmodels.nonparametric.kernel_regression import KernelReg
import yfinance as yf
from datetime import datetime, timedelta
from backtester import *
from pandas.tseries.offsets import *
import operator
import warnings
warnings.filterwarnings("ignore")