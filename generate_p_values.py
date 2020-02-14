from load_data import load_sas_study
from aggregation import compute_all_tests

def get_proportions(results, thresh):
    result = (kw_results['pval'] < thresh).groupby('centre').mean().to_frame()
    result.columns = ['pval_prop']

    return result

def proportions_and_predictions(results, pval_thresh, proportion_thresh,
                                anomalous_centres):
    prop = get_proportions(results, pval_thresh)
    prop['pval_pred'] = (prop > prop_thresh)
    prop['anomalous'] = False
    prop.loc[np.intersect1d(prop.index, anomalous_centres), 'anomalous'] = True

    return prop

if __name__ == '__main__':
    # Load the data
    hipattack_dataset = load_sas_study('/dhi_work/share/data/fraud/hipattack', impute=False)
    schema_hipattack, data_hipattack, anomalies_hipattack = hipattack_dataset
    poise_dataset = load_sas_study('/dhi_work/share/data/fraud/poise', impute=False)
    schema_poise, data_poise, anomalies_poise = poise_dataset

    # Compute all statistical tests
    statistical_tests_hipattack = compute_all_tests(data_hipattack)
    statistical_tests_poise = compute_all_tests(data_poise)

    statistical_tests_hipattack.to_csv('output/hipattack_p_values.csv')
    statistical_tests_poise.to_csv('output/poise_p_values.csv')

