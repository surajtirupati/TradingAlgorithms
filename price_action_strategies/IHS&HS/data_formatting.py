from find_extrema_file import *

def format_strategy_data(data, strategy, max_bars, signal_type):
    # initialising lists to store output variables
    detection_dates = []
    price_points = [[0, 1, 2, 3, 4]]
    entry_points = []
    entry_dates = []
    stoploss = []

    for i in range(0, len(data)):

        # indexing the prices of interest
        prices = data["Close"][i:i + max_bars]

        # finding the extrema
        extrema, _, _, _ = find_extrema(prices, bw=[0.85])

        # only run head and shoulders when there are 5 extrema
        if len(extrema) >= 5:
            # detecting the extrema of the input strategy
            e = strategy(extrema)

        # on the first appendment to the strategy detection variable lists
        if e is not None:

            check = np.array(price_points[-1]) - np.array(e)

            if np.count_nonzero(check == 0) == 0:  # ensures no duplicate patterns

                detection_dates.append(prices.index[-1])
                price_points.append(e)

                if signal_type == "bearish":
                    nxt_10_days = data.loc[prices.index[-1]:prices.index[-1] + timedelta(days=10)].Close.le(min(e))
                    entry_points.append(min(e))
                    stoploss.append(max(e))
                    op = operator.sub

                elif signal_type == "bullish":
                    nxt_10_days = data.loc[prices.index[-1]:prices.index[-1] + timedelta(days=10)].Close.ge(max(e))
                    entry_points.append(max(e))
                    stoploss.append(min(e))
                    op = operator.add

                if len(nxt_10_days[nxt_10_days == True]) != 0:
                    val = nxt_10_days[nxt_10_days == True].idxmin()
                    entry_dates.append(val)

                else:
                    entry_dates.append("No Entry")

    return detection_dates, price_points[1:], entry_points, entry_dates, stoploss, op


def create_strategy_df(data, detection_dates, entry_dates, stoploss, pos_target, op):
    # initialising strategy variables in dataframe
    data = data[["Open", "High", "Low", "Close", "Volume"]]
    data["PatternDetected"] = 0
    data["Entry"] = 0
    data["StopLoss"] = 0
    data["PosTarget"] = 0

    # creating values at relevant entry points

    for i in range(0, len(entry_dates)):

        if type(entry_dates[i]) != str:
            data.loc[detection_dates[i], "PatternDetected"] = 1
            data.loc[entry_dates[i], "Entry"] = 1
            data.loc[entry_dates[i], "StopLoss"] = stoploss[i]

            data.loc[entry_dates[i], "PosTarget"] = data.iloc[data.index.get_loc(entry_dates[i]) + 1]["Open"] * op(1, pos_target)

    return data