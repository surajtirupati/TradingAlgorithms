from package_req import *

def inverse_head_and_shoulders(extrema):
    """
    Input:
        extrema: extrema as pd.series with bar number as index
        max_bars: max bars for pattern to play out
    Returns:
        dates: list of dates containing the start and end bar of the pattern
    """

    # Need to start at five extrema for pattern generation
    for i in range(5, len(extrema)+1):
        window = extrema.iloc[i-5:i]

        # Using the notation from the paper to avoid mistakes
        e1 = window.iloc[0]
        e2 = window.iloc[1]
        e3 = window.iloc[2]
        e4 = window.iloc[3]
        e5 = window.iloc[4]

        # Inverse Head and Shoulders
        if (e1 < e2) and (e3 < e1) and (e3 < e5) and \
             (abs(e1 - e5) <= 0.03 * np.mean([e1, e5])) and \
             (abs(e2 - e4) <= 0.03 * np.mean([e1, e5])):

            return [e1, e2, e3, e4, e5]

    return