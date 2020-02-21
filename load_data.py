from glob import glob
import pandas as pd
from sas7bdat import SAS7BDAT
import numpy as np
from math import floor

from functools import reduce

INCLUSION_THRESHOLD = 5

def load_sas(fn, schema):
    '''Load a SAS file according to the given schema.

    Parameters:
        fn : str
            Path to the SAS file.
        schema : DataFrame
            Data frame containing the schema of the study.

    Returns:
        df_sub : DataFrame
            The data available in the SAS file, subset to the columns listed in
            the schema.
    '''
    with SAS7BDAT(fn) as f:
        df = f.to_data_frame()
        if 'id' in df.columns:
            df['id'] = df['id'].astype(int)
            df['centre'] = np.asarray(df['id']) // 1000
            df = df.set_index(['centre', 'id'])
        elif 'centre' in df.columns:
            print('WARNING: no id column found in df {}'.format(fn))
            df['id'] = -9
            df['centre'] = df['centre'].astype(int)
            df = df.set_index(['centre', 'id'])
        else:
            print('WARNING: no id column found in df {}'.format(fn))
            df['id'] = -9
            df['centre'] = -9
            df = df.set_index(['centre', 'id'])
    schema_cols = list(schema.index)
    cols = [c for c in df.columns if c in schema_cols]
    dtypes = schema.loc[cols].groupby('Name').first()['Type']
    df_sub = df[cols]
    for col in dtypes.index:
        if dtypes[col] == 'int':
            try:
                df_sub[col] = df_sub[col].astype(pd.Int64Dtype())
            except:
                msg = 'error converting int column {}, leaving as float'
                print(msg.format(col))
                try:
                    df_sub[col] = df_sub[col].astype(float)
                except:
                    print('not a valid float either! dropping {}'.format(col))
                    df_sub = df_sub.drop(col, axis=1)
        elif dtypes[col] == 'choice' or dtypes[col] == 'check' :
            try:
                df_sub[col] = pd.Categorical(df_sub[col].astype(pd.Int64Dtype()))
            except:
                msg = 'error converting categorical column {}, dropping it'
                print(msg.format(col))
                df_sub = df_sub.drop(col, axis=1)
        elif dtypes[col] == 'string' or dtypes[col] == 'date':
            df_sub = df_sub.drop(col, axis=1)

    return df_sub

def load_sas_study(root_path, impute=True):
    '''Load a set of mid-study data.
    
    Parameters:
        root_path : str
            The path to the root of the data directory. Expected files in this
            directory are 'anomalies.csv', 'schema.csv', and a directory
            plates, containing the SAS files containing the actual data.
        impute : bool (default : True)
            Whether or not to impute missing values. If True, missing values are
            imputed as -9e10.

    Returns:
        schema : DataFrame
            The DataFrame containing the full schema.
        plates : DataFrame
            The DataFrame containing the full data.
        anomalies : DataFrame
            The DataFrame containing the information on anomalies.
    '''
    anomalies_path = root_path + '/anomalies.csv'
    schema_path = root_path + '/schema.csv'
    plates_paths = glob(root_path + '/plates/*.sas7bdat')
    
    anomalies = pd.read_csv(anomalies_path)
    schema = pd.read_csv(schema_path).set_index('Name')
    plates = [load_sas(fn, schema) for fn in plates_paths]
    uniq_pids = np.unique(
        np.concatenate([np.asarray(plate.index.get_level_values(1))
                        for plate in plates if len(plate.columns) > 0])
    )
    nonanom_pids = np.setdiff1d(uniq_pids, anomalies['id'])
    anomalies = pd.concat([
        anomalies,
        pd.DataFrame({'id': nonanom_pids,
                      'centre': nonanom_pids // 1000,
                      'fraud': [0] * len(nonanom_pids)})
    ], axis=0)
    anomalies['centre'] = anomalies['id'] // 1000
    anomalies['anomalous'] = (anomalies['fraud'] == 1)
    anomalies = anomalies.drop('fraud', axis=1).set_index(['centre', 'id'])
    plates_joined = reduce(
        lambda x, y: x.merge(y, on=['centre', 'id'], how='outer'),
        [p for p in plates if len(p.columns) > 0]
    )
    if impute:
        plates_numeric = plates_joined.select_dtypes(include=['int', 'float'])\
                                      .astype(float)
        plates_cat = plates_joined.select_dtypes(include=['category'])
        for c in plates_cat.columns:
            try:
                plates_cat[c] = plates_cat[c].cat.add_categories(-9e10)\
                                             .fillna(-9e10)
            except:
                print('failed imputing on column {}, will drop'.format(c))
        plates_cat = plates_cat.dropna(axis=1)
        plates_numeric = plates_numeric.fillna(-9e10)
        plates_joined = pd.concat([plates_numeric, plates_cat], axis=1)

    # Remove centers with too few samples
    sample_size = plates_joined.reset_index().set_index('centre')\
                               .groupby('centre')['id'].nunique().to_frame()
    sample_size.columns = ['sample_size']
    centres = sample_size[sample_size['sample_size'] >= INCLUSION_THRESHOLD]\
                .index
    plates_joined = plates_joined[
        plates_joined.index.get_level_values(0).isin(centres)
    ]
    anomalies = anomalies[anomalies.index.get_level_values(0).isin(centres)]

    # Remove admin variables
    admin_variable_names = np.loadtxt('output/admin_variable_names.txt', dtype=str)
    plates_joined = plates_joined.drop(np.intersect1d(plates_joined.columns, admin_variable_names), axis=1)
    
    return schema, plates_joined, anomalies

