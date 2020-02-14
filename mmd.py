from sklearn.manifold import MDS

def compute_mmd(df, K):
    centre = np.asarray(df.index.get_level_values('centre'))
    unique_centres = np.unique(centre)
    mmd2 = np.zeros((len(unique_centres), len(unique_centres)))
    pvals = np.zeros((len(unique_centres), len(unique_centres)))
    kmax = K.max()
    mmd2_global = np.zeros((len(unique_centres)))
    for i, ci in enumerate(unique_centres):
        sel_i = (centre == ci)
        mmd2_global[i] = np.mean(K[sel_i, :][:, sel_i]) - 2 * np.mean(K[~sel_i, :][:, sel_i]) + np.mean(K[~sel_i, :][:, ~sel_i])
        for j, cj in enumerate(unique_centres):
            sel_j = (centre == cj)
            mmd2[i, j] = np.mean(K[sel_i, :][:, sel_i]) - 2 * np.mean(K[sel_i, :][:, sel_j]) + np.mean(K[sel_j, :][:, sel_j])
            m = np.min([np.sum(sel_i), np.sum(sel_j)])
            pvals[i, j] = 1 / np.exp((np.sqrt(mmd2[i, j]) * np.sqrt(m / (2 * kmax)) - 1)**2 / 2)
    mmd2[mmd2 < 0] = 0
    mmd = np.sqrt(mmd2)
    mmd_global = np.sqrt(mmd2_global)
    
    return mmd, pvals, mmd_global, unique_centres


def centre_latent_mmd(df, K, k_mds=8, include_global=True, use_pvals=False):
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
    result = pd.DataFrame(np.concatenate([unique_centres[:, np.newaxis], X_mmd], axis=1), columns=cols)
    result['centre'] = result['centre'].astype(int)
    result = result.set_index('centre')
    return result
