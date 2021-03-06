import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, average_precision_score

from load_data import load_sas_study
from summary_statistics import output_predictive_model_stats

def get_proportions(results, thresh):
    '''Get proportion of p-values under a given threshold.

    Parameters:
        results : pd.DataFrame
            Data frame containing a column 'pval'.
        thresh : float
            p-value threshold.

    Returns:
        result : pd.DataFrame
            DataFrame with a single column 'pval_prop' indexed by centre.
    '''
    result = (results['pval'] < thresh).groupby('centre').mean().to_frame()
    result.columns = ['pval_prop']

    return result

def proportions_and_predictions(results, pval_thresh, prop_thresh,
                                anomalous_centres):
    '''Get labeled scores and predictions for the p-value approach.

    Parameters:
        results : pd.DataFrame
            Data frame containing the p-value results.
        pval_thresh : float
            The p-value threshold used to generate the proportions.
        prop_thresh : float
            The proportion threshold used as a decision boundary.
        anomalous_centres : list-like
            List of anomalous centre IDs.

    Returns:
        prop : pd.Dataframe
            Data frame containing the scores, the predictions, and the labels
            for the p-value approach.
    '''
    prop = get_proportions(results, pval_thresh)
    prop['pval_pred'] = (prop > prop_thresh)
    prop['anomalous'] = False
    prop.loc[np.intersect1d(prop.index, anomalous_centres), 'anomalous'] = True

    return prop

def output_statistics(df):
    '''Output summary statistics for the p-value model to the terminal.

    Parameters:
        df : pd.DataFrame
            Data frame containing the proportion scores and the labels.
    '''
    auroc = roc_auc_score(df['anomalous'], df['pval_prop'])
    aupr = average_precision_score(df['anomalous'], df['pval_prop'])
    centres = df.index
    num_centres = len(np.unique(centres))
    print('\tNumber of centers: {}'.format(num_centres))
    print('\tAUROC: {:.3f}'.format(auroc))
    print('\tAUPR: {:.3f}'.format(aupr))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--include-missingness', action='store_const',
                        default=False, const=True)
    parser.add_argument('--include-ks', action='store_const',
                        default=False, const=True)
    args = parser.parse_args()

    schema_hipattack, _, _ = load_sas_study(
        '/dhi_work/share/data/fraud/hipattack'
    )
    schema_poise, _, _ = load_sas_study(
        '/dhi_work/share/data/fraud/poise'
    )

    varnames_hipattack = schema_hipattack[
        schema_hipattack['Plate'].isin([1, 2, 3, 4, 5, 6, 7, 23, 24, 102, 106])
    ].index
    varnames_poise = schema_poise[
        schema_poise['Plate'].isin([1, 2, 3, 4, 5, 102])
    ].index

    pvalues_hipattack = pd.read_csv('output/hipattack_p_values.csv')\
                          .set_index('centre')
    pvalues_poise = pd.read_csv('output/poise_p_values.csv')\
                      .set_index('centre')

    pvalues_hipattack = pvalues_hipattack[
        pvalues_hipattack['variable'].isin(varnames_hipattack)
    ]
    pvalues_poise = pvalues_poise[
        pvalues_poise['variable'].isin(varnames_poise)
    ]

    exclude = []
    if not args.include_missingness:
        exclude.append('missingness')
    if not args.include_ks:
        exclude.append('ks')

    pvalues_hipattack = pvalues_hipattack[
        ~pvalues_hipattack['test'].isin(exclude)
    ]
    pvalues_poise = pvalues_poise[
        ~pvalues_poise['test'].isin(exclude)
    ]

    pvalues_hipattack = proportions_and_predictions(pvalues_hipattack,
                                                    0.05, 0.2,
                                                    [43, 216, 530])
    pvalues_poise = proportions_and_predictions(pvalues_poise, 0.05, 0.4,
                                                [141, 142, 143, 144,
                                                 551, 552, 553, 554, 555, 556])

    print('P-VALUE THRESHOLDING APPROACH')
    print('=============================')
    print('POISE results:')
    output_statistics(pvalues_poise)
    print('HIP-ATTACK results:')
    output_statistics(pvalues_hipattack)

