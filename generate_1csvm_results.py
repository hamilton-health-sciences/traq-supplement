import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from load_data import load_sas_study
from summary_statistics import output_predictive_model_stats

def extract_labelled_data(pvalues, mmd_features, anomalous_centres):
    def get_labelled(X):
        X_std = StandardScaler().fit_transform(X)
        X_std = pd.DataFrame(X_std, columns=X.columns)
        X_std.index = X.index
        y = X.index.to_series().apply(lambda x: x in anomalous_centres)
        return X_std, y

    pvalues['variable_test'] = pvalues['variable'] + '_' + pvalues['test']
    X_pvalues_ = pd.pivot_table(pvalues, index='centre',
                                columns='variable_test', values='pval')#.dropna(axis=1)
    X_pvalues_ = X_pvalues_.loc[:, ~X_pvalues_.columns.duplicated()]
    sample_missingness = (pd.isnull(X_pvalues_).sum(axis=1) / X_pvalues_.shape[1])
    #X_pvalues_ = X_pvalues_[sample_missingness < 0.5].dropna(axis=1)
    X_pvalues_ = X_pvalues_.dropna(axis=1)
    eps = np.finfo(np.float64).eps
    X_pvalues_ = np.log10(X_pvalues_ + eps)
    X_pvalues, y_pvalues = get_labelled(X_pvalues_)
    X_mmd, y_mmd = get_labelled(mmd_features)

    return X_pvalues, X_mmd, y_pvalues, y_mmd

def get_predictions(X, y, ktype):
    svm = OneClassSVM(kernel=ktype, gamma='auto').fit(X)
    y_pred = svm.predict(X)
    y_pred = ((1 - y_pred) / 2).astype(int)
    y_score = -svm.decision_function(X)
    result = pd.DataFrame({'centre': X.index, 'pred': y_pred, 'score': y_score,
                           'anomalous': y})

    return result

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--include-missingness', action='store_const',
                        default=False, const=True)
    parser.add_argument('--include-ks', action='store_const',
                        default=False, const=True)
    args = parser.parse_args()

    schema_hipattack, _, _ = load_sas_study('/dhi_work/share/data/fraud/hipattack')
    schema_poise, _, _ = load_sas_study('/dhi_work/share/data/fraud/poise')

    varnames_hipattack = schema_hipattack[
        schema_hipattack['Plate'].isin([1, 2, 3, 4, 5, 6, 7, 23, 24, 102, 106])
    ].index
    varnames_poise = schema_poise[schema_poise['Plate'].isin([1, 2, 3, 4, 5, 102])].index

    pvalues_hipattack = pd.read_csv('output/hipattack_p_values.csv').set_index('centre')
    pvalues_poise = pd.read_csv('output/poise_p_values.csv').set_index('centre')

    pvalues_hipattack = pvalues_hipattack[pvalues_hipattack['variable'].isin(varnames_hipattack)]
    pvalues_poise = pvalues_poise[pvalues_poise['variable'].isin(varnames_poise)]

    exclude = []
    if not args.include_missingness:
        exclude.append('missingness')
    if not args.include_ks:
        exclude.append('ks')

    pvalues_hipattack = pvalues_hipattack[~pvalues_hipattack['test'].isin(exclude)]
    pvalues_poise = pvalues_poise[~pvalues_poise['test'].isin(exclude)]

    mmd_features_hipattack = pd.read_csv('output/mmd_features_hipattack_pvals.csv').set_index('centre')
    mmd_features_poise = pd.read_csv('output/mmd_features_poise_pvals.csv').set_index('centre')
    #mmd_features_hipattack = pd.read_csv('/dhi_work/share/data/fraud/hipattack/mmd_features_subset_pvals.csv').set_index('centre')
    #mmd_features_poise = pd.read_csv('/dhi_work/share/data/fraud/poise/mmd_features_subset_pvals.csv').set_index('centre')

    X_hipattack, X_hipattack_mmd, y_hipattack, y_hipattack_mmd = extract_labelled_data(
        pvalues_hipattack, mmd_features_hipattack, [43, 216, 530]
    )
    print('HIPATTACK shape: {} x {}'.format(*X_hipattack.shape))
    X_poise, X_poise_mmd, y_poise, y_poise_mmd = extract_labelled_data(
        pvalues_poise, mmd_features_poise, [141, 142, 143, 144, 551, 552, 553, 554, 555, 556]
    )
    print('POISE shape: {} x {}'.format(*X_poise.shape))

    predictions_hipattack = get_predictions(X_hipattack, y_hipattack, 'poly')
    predictions_hipattack.to_csv('output/method_2_hipattack_results.csv')
    predictions_mmd_hipattack = get_predictions(X_hipattack_mmd, y_hipattack_mmd, 'rbf')
    predictions_poise = get_predictions(X_poise, y_poise, 'poly')
    predictions_poise.to_csv('output/method_2_poise_results.csv')
    predictions_mmd_poise = get_predictions(X_poise_mmd, y_poise_mmd, 'rbf')

    print('ONE-CLASS SVM RESULTS')
    print('=====================')
    print('HIP-ATTACK (p-values):')
    output_predictive_model_stats(predictions_hipattack)
    print('HIP-ATTACK (MMD):')
    output_predictive_model_stats(predictions_mmd_hipattack)
    print('POISE (p-values):')
    output_predictive_model_stats(predictions_poise)
    print('POISE (MMD):')
    output_predictive_model_stats(predictions_mmd_poise)

