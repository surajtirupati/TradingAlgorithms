from correlation_analysis import get_data, offset_and_rolling_lag_corr
from scipy.stats import mannwhitneyu, levene, kruskal
import pandas as pd
import numpy as np


def mv_df(indep_name, dep_name, offset_lag, rolling_lag):
    """
    Inputs: independent variable, dependent variable, offset lag, and rolling lag

    Outputs: Dataframe with relevant market variables from each
    """

    # construct dataframes containing independent market variables
    df_indep = get_data(indep_name)
    df_indep["Returns"] = df_indep["Close"].pct_change()
    df_indep["1WkVol"] = df_indep["Returns"].rolling(7).std()

    # construct dataframes containing dependent market variables
    df_dep = get_data(dep_name)
    df_dep["Returns"] = df_dep["Close"].pct_change()
    df_dep["1WkVol"] = df_dep["Returns"].rolling(7).std()

    # obtaining only variables of interest
    df_indep = df_indep[["Close", "Returns", "1WkVol", "Volume"]]
    df_dep = df_dep[["Close", "Returns", "1WkVol", "Volume"]]

    # renaming columns
    df_indep.columns = ["Independent " + col for col in df_indep.columns]
    df_dep.columns = ["Dependent " + col for col in df_dep.columns]

    # combining the dataframes
    market_variable_df = pd.concat([df_indep, df_dep], axis=1)

    # obtaining index of interest
    idx_of_interest = market_variable_df.index[rolling_lag - 1:-offset_lag]

    # obtaining correlation with respective lags
    correlation = offset_and_rolling_lag_corr(df_indep["Independent Close"], df_dep["Dependent Close"], offset_lag,
                                              rolling_lag)
    correlation.index = idx_of_interest

    # appending, renaming, and calculating and appending absolute difference of correlation to the market variable dataframe
    market_variable_df = pd.concat([market_variable_df, correlation], axis=1)
    market_variable_df = market_variable_df.rename(columns={0: "Correlation"})
    market_variable_df["CorrDiff"] = np.sign(market_variable_df["Correlation"].diff())
    market_variable_df = market_variable_df.dropna()

    # returning the market_variable_df
    return market_variable_df


def gen_test_res(market_variables_df, test_df):
    """
    Inputs:
    - a market variable dataframe containing all the relevant market variables to conduct
    statistical tests on
    - a test_df which contains a subset of the values of the market variables based on a hypothesis
    condition.

    Outputs a dataframe containing each market variable and the results of each of the tests:
    - Mann Whitney U: Testing means
    - Levene: Testing variances
    - Kruskal Wallis: Testing medians

    """

    test_res = []

    for var_to_test in test_df.columns:

        if "Corr" not in var_to_test:

            if "Close" not in var_to_test:
                _, p_mwu = mannwhitneyu(test_df[var_to_test].values, market_variables_df[var_to_test])

                _, p_lev = levene(test_df[var_to_test].values, market_variables_df[var_to_test])

                _, p_kw = kruskal(test_df[var_to_test].values, market_variables_df[var_to_test])

                test_res.append([var_to_test, p_mwu, check_test_res(p_mwu), p_lev, check_test_res(p_lev), p_kw, check_test_res(p_kw)])

    res_df = pd.DataFrame(test_res)

    columns = ["Market Variable", "Mann Whitney P value", "Mann Whitney Res (Means)", "Levene P value", "Levene Res (Variances)", "Kruskal P value", "Kruskal Res (Medians)"]

    res_df.columns = columns

    return res_df

def check_test_res(p):
    """
    Checks the p value and returns the result - whether or not the NH can be rejected
    """

    if p < 0.05:
        return "Reject NH"

    else:
        return "Can't Reject NH"


def statistical_routine(indep_name, dep_name, offset_lag, rolling_lag, hypothesis_func):
    """
    Inputs:
    - independent variable name
    - dependent variable name
    - offset lag
    -rolling lag
    - hypothesis_func: a function that takes in the overall market variable dataframe and returns a subset based on a
    particular hypothesis

    Outputs:
    - dataframe containing the test results for each market variable and each of the input variables
    """

    mvdf = mv_df(indep_name, dep_name, offset_lag, rolling_lag)

    subset = hypothesis_func(mvdf)

    output = gen_test_res(subset, mvdf)

    output.insert(loc=0, column="Independent Name", value=indep_name)
    output.insert(loc=1, column="Dependent Name", value=dep_name)
    output.insert(loc=2, column="Offset Lag", value=offset_lag)
    output.insert(loc=3, column="Rolling Lag", value=rolling_lag)

    return output