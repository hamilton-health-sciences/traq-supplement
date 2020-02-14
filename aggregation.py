import pandas as pd
import numpy as np

from statistical_tests import digit_preference, digit_difference, ks_test, \
                              missingness_test, categorical_difference

MIN_SAMPLES = 5

def _compute_numeric_tests(s):
    centres = np.unique(s.index.get_level_values(0))
    varname = s.name
    series = []
    for c in centres:
        within = s.xs(c, level=0)
        other = s.drop(c, level=0)
        x_ = np.asarray(within, dtype=float)
        y_ = np.asarray(other, dtype=float)
        x = x_[~np.isnan(x_)]
        y = y_[~np.isnan(y_)]
        if len(x) > MIN_SAMPLES:
            p_dp = digit_preference(x)
            series.append([c, varname, 'digit_preference', p_dp])
            if len(y) > MIN_SAMPLES:
                p_dd = digit_difference(x, y)
                p_ks = ks_test(x, y)
                series.append([c, varname, 'digit_difference', p_dd])
                series.append([c, varname, 'ks', p_ks])
        if len(x_) > MIN_SAMPLES:
            p_missingness = missingness_test(x_, y_)
            series.append([c, varname, 'missingness', p_missingness])
    df = pd.DataFrame(series, columns=['centre', 'variable', 'test', 'pval'])
    return df

def _compute_categorical_tests(s):
    centres = np.unique(s.index.get_level_values(0))
    varname = s.name
    series = []
    for c in centres:
        within = s.xs(c, level=0)
        other = s.drop(c, level=0)
        x_ = np.asarray(within, dtype=float)
        y_ = np.asarray(other, dtype=float)
        x = x_[~np.isnan(x_)].astype(int)
        y = y_[~np.isnan(y_)].astype(int)
        if len(x) > MIN_SAMPLES and len(y) > MIN_SAMPLES:
            p_pcs = categorical_difference(x, y)
            series.append([c, varname, 'categorical_difference', p_pcs])
        if len(x_) > MIN_SAMPLES:
            p_missingness = missingness_test(x_, y_)
            series.append([c, varname, 'missingness', p_missingness])
    df = pd.DataFrame(series, columns=['centre', 'variable', 'test', 'pval'])
    return df

def compute_all_tests(df):
    '''Compute all available tests for the data.
    
    Parameters:
        schema : DataFrame
            A dataframe indexed by the variables in data with a column 'Type'.
        data : DataFrame
            The data the tests will be performed on, with missingness coded as nan.
    
    Returns:
        results : DataFrame
            A DataFrame consisting of 4 columns: centre, variable, type of test performed, and p-value.
    '''
    df_numeric = df.select_dtypes(include='number')
    df_categorical = df.select_dtypes(include='category')
    results_numeric = pd.concat(
        [_compute_numeric_tests(df_numeric.iloc[:, i]) for i in range(df_numeric.shape[1])],
        axis=0
    )
    results_categorical = pd.concat(
        [_compute_categorical_tests(df_categorical.iloc[:, i]) for i in range(df_categorical.shape[1])]
    )
    results = pd.concat([results_numeric, results_categorical], axis=0).set_index('centre')
    
    return results
