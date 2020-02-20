import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def output_predictive_model_stats(df):
    '''Output the statistics for a predictive model.'''
    auroc = roc_auc_score(df['anomalous'], df['score'])
    aupr = average_precision_score(df['anomalous'], df['score'])
    centers = df.index.get_level_values('centre')
    num_centers = len(np.unique(centers))
    print('\tNumber of centers: {}'.format(num_centers))
    print('\tAUROC: {:.3f}'.format(auroc))
    print('\tAUPR: {:.3f}'.format(aupr))
