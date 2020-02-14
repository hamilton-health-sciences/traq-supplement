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

