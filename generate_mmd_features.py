import numpy as np
import pandas as pd

from sklearn.manifold import MDS
from load_data import load_sas_study

def continuous_kernel(x):
    '''Computes a basic squared-exponential kernel, where the parameter is set
    according to the median heuristic.

    Parameters:
        x : np.ndarray
            A length N vector containing the observations of interest.
    
    Returns:
        K : np.ndarray
            An N x N matrix of kernel values.
    '''
    D = np.abs(np.subtract.outer(x, x))
    D[np.isnan(x), :] = np.nan
    D[:, np.isnan(x)] = np.nan
    D[np.isnan(D)] = np.nanmean(D)
    D[np.diag_indices(len(x))] = 0
    sig = np.median(D[np.triu_indices(len(x))])
    K = np.exp(-D**2 / (2 * sig**2))
    if np.isnan(K).sum() > 0:
        K = np.zeros(K.shape)
    return K

def categorical_kernel(alpha=1.0):
    '''Generates a categorical kernel function with deflation parameter alpha.
    Reference: doi:10.3233/978-1-61499-320-9-171 (defn 2.1 and 2.2).

    Parameters:
        alpha : float
            Deflation parameter of the kernel.

    Returns:
        k : function
            A function which accepts a vector of categorical values as a
            parameter, and returns the corresponding kernel matrix.
    '''
    # TODO: convert from functional to function
    def k(x):
        xu, xc = np.unique(x, return_counts=True)
        xc = xc[np.argsort(xu)]
        xu_new = np.arange(np.max(xu) + 1)
        for i in xu_new:
            if i not in xu:
                idx = np.min(np.where(xu > i)[0])
                xc = np.insert(xc, idx, 0)
                print(xu)
                print(xu_new)
                print(xc)
        prob = xc / len(x)
        K = np.tile(x, (len(x), 1))
        K = (1 - prob[K]**alpha)**(1 / alpha)
        K[np.not_equal.outer(x, x)] = 0
        return K
    return k

def compute_aggregate_kernel(df, plate_subset=None, schema=None, categorical_alpha=1.0):
    '''Compute the aggregate patient-wise kernel for a given dataset.

    Parameters:
        df : DataFrame
            The DataFrame containing the data of interest, containing N rows.
        plate_subset : list-like
            The list of plate numbers to subset to.
        schema : DataFrame
            The schema describing the data.
        categorical_alpha : float
            The alpha parameter to use for the categorical kernel.

    Returns:
        K_sum : np.ndarray
            The N x N kernel matrix which is simply the mean of the individual
            variable-level kernels.
    '''
    if plate_subset is not None:
        variables = np.asarray(schema[schema['Plate'].isin(plate_subset)].index)
        df = df[np.intersect1d(df.columns, variables)]
    df_numeric = df.select_dtypes(include='number')
    print('{} numeric features extracted'.format(df_numeric.shape[1]))
    df_categorical = df.select_dtypes(include='category')
    print('{} categorical features extracted'.format(df_categorical.shape[1]))
    K_sum = 0
    bad = 0
    for i in range(df_numeric.shape[1]):
        K_i = continuous_kernel(np.asarray(df_numeric.iloc[:, i]).astype(float))
        if K_i.sum() == 0:
            print('Bad {}'.format(i))
            bad += 1
        K_sum += K_i
    cK = categorical_kernel(categorical_alpha)
    for i in range(df_categorical.shape[1]):
        x = df_categorical.iloc[:, i].cat.add_categories(-9e5).fillna(-9e5)
        x = np.asarray(pd.Categorical(x).codes).astype(int)
        K_i = cK(x)
        if (np.isnan(K_i).sum() == 0):
            K_sum += K_i
        else:
            print('Bad {}'.format(i))
            bad += 1
    K_sum /= df_numeric.shape[1] + df_categorical.shape[1] - bad
    
    return K_sum

def compute_mmd(df, K):
    '''Compute the maximum mean discrepancy between centres and from centre to
    the global distribution.

    Parameters:
        df : DataFrame
            The DataFrame containing the data that was used to generate the
            kernel.
        K : np.ndarray
            A C x C matrix, where C is the number of centers included in `df`.

    Returns :
        mmd : np.ndarray
            A C x C matrix, containing the MMD statistics.
        pvals : np.ndarray
            A C x C matrix, containing the corresponding p-values.
        mmd_global : np.ndarray
            A length C array, containing the MMD between each centre and the
            global distribution.
        unique_centres : np.ndarray
            A list of the centres in the computations.
    '''
    centre = np.asarray(df.index.get_level_values('centre'))
    unique_centres = np.unique(centre)
    mmd2 = np.zeros((len(unique_centres), len(unique_centres)))
    pvals = np.zeros((len(unique_centres), len(unique_centres)))
    kmax = K.max()
    mmd2_global = np.zeros((len(unique_centres)))
    for i, ci in enumerate(unique_centres):
        sel_i = (centre == ci)
        mmd2_global[i] = np.mean(K[sel_i, :][:, sel_i]) - \
                         2 * np.mean(K[~sel_i, :][:, sel_i]) + \
                         np.mean(K[~sel_i, :][:, ~sel_i])
        for j, cj in enumerate(unique_centres):
            sel_j = (centre == cj)
            mmd2[i, j] = np.mean(K[sel_i, :][:, sel_i]) - \
                         2 * np.mean(K[sel_i, :][:, sel_j]) + \
                         np.mean(K[sel_j, :][:, sel_j])
            m = np.min([np.sum(sel_i), np.sum(sel_j)])
            pvals[i, j] = 1 / np.exp((np.sqrt(mmd2[i, j]) * \
                          np.sqrt(m / (2 * kmax)) - 1)**2 / 2)
    mmd2[mmd2 < 0] = 0
    mmd = np.sqrt(mmd2)
    mmd_global = np.sqrt(mmd2_global)
    
    return mmd, pvals, mmd_global, unique_centres


def centre_latent_mmd(df, K, k_mds=8, include_global=True, use_pvals=False):
    '''Compute the features used as inputs to the unsupervised models.

    Parameters:
        df : DataFrame
            The DataFrame containing the data that was used to generate the
            kernel.
        K : np.ndarray
            A C x C matrix, where C is the number of centers included in `df`.
        k_mds : int
            The number of dimensions to be used when running multi-dimensional
            scaling on the MMD distance matrix.
        include_global : boolean
            Whether or not to include the center-to-global MMD as a feature.
        use_pvals : boolean
            Whether to transform the MMD distances to p-values, which considers
            the size of the centre.
    Returns:
        result : DataFrame
            A DataFrame containing the engineered features.
    '''
    mmd, pvals, mmd_global, unique_centres = compute_mmd(df, K)
    mds = MDS(n_components=k_mds, dissimilarity='precomputed')
    if use_pvals:
        X_mmd = mds.fit_transform(-np.log10(pvals))
    else:
        X_mmd = mds.fit_transform(mmd)
    cols = ['centre'] + ['mmd_dim{}'.format(i) for i in range(k_mds)]
    if include_global:
        X_mmd = np.concatenate([X_mmd, mmd_global[:, np.newaxis]], axis=1)
        cols += ['mmd_global']
    result = pd.DataFrame(
        np.concatenate([unique_centres[:, np.newaxis], X_mmd], axis=1),
        columns=cols
    )
    result['centre'] = result['centre'].astype(int)
    result = result.set_index('centre')

    return result

if __name__ == '__main__':
    from os import path

    schema_hipattack, data_hipattack, anomalies_hipattack = load_sas_study(
        '/dhi_work/share/data/fraud/hipattack',
        impute=False
    )
    schema_poise, data_poise, anomalies_poise = load_sas_study(
        '/dhi_work/share/data/fraud/poise',
        impute=False
    )
    
    if path.exists('output/kernel_hipattack.npz'):
        K_hipattack = np.load('output/kernel_hipattack.npz')['arr_0']
    else:
        K_hipattack = compute_aggregate_kernel(
            data_hipattack,
            [1, 2, 3, 4, 5, 6, 7, 23, 24, 102, 106],
            schema_hipattack
        )
        np.savez('output/kernel_hipattack.npz', K_hipattack)

    if path.exists('output/kernel_poise.npz'):
        K_poise = np.load('output/kernel_poise.npz')['arr_0']
    else:
        K_poise = compute_aggregate_kernel(data_poise,
                                           [1, 2, 3, 4, 5, 102],
                                           schema_poise)
        np.savez('output/kernel_poise.npz', K_poise)

    mmd_features_hipattack = centre_latent_mmd(data_hipattack, K_hipattack)
    mmd_features_hipattack_pvals = centre_latent_mmd(data_hipattack,
                                                     K_hipattack,
                                                     use_pvals=True)
    mmd_features_poise = centre_latent_mmd(data_poise, K_poise)
    mmd_features_poise_pvals = centre_latent_mmd(data_poise,
                                                 K_poise,
                                                 use_pvals=True)

    mmd_features_hipattack.to_csv('output/mmd_features_hipattack.csv')
    mmd_features_hipattack_pvals.to_csv(
        'output/mmd_features_hipattack_pvals.csv'
    )
    mmd_features_poise.to_csv('output/mmd_features_poise.csv')
    mmd_features_poise_pvals.to_csv(
        'output/mmd_features_poise_pvals.csv'
    )

