import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def output_predictive_model_stats(df):
    auroc = roc_auc_score(df['anomalous'], df['score'])
    aupr = average_precision_score(df['anomalous'], df['score'])
    print('\tAUROC: {:.3f}'.format(auroc))
    print('\tAUPR: {:.3f}'.format(aupr))
