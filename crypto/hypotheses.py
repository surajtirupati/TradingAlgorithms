def high_corr_hypothesis(df):
    subset = df[abs(df["Correlation"]) > 0.7]
    return subset

def high_corr_while_incr(df):
    subset = df[(df["CorrDiff"] == 1) & (abs(df["Correlation"]) > 0.7)]
    return subset

def high_corr_while_decr(df):
    subset = df[(df["CorrDiff"] == -1) & (abs(df["Correlation"]) > 0.7)]
    return subset

def incr_corr(df):
    subset = df[df["CorrDiff"] == 1]
    return subset

def decr_corr(df):
    subset = df[df["CorrDiff"] == -1]
    return subset
