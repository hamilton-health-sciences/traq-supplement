import numpy as np
import scipy as sp
import scipy.stats

def _chisq(x, p):
    '''Perform Pearson's chi-squared test.
    
    Parameters:
        x : np.ndarray
            The vector containing the observations.
        p : dict
            Dict mapping unique values of x to probabilities.
            
    Returns:
        pval : float
            The p-value from the test.
    '''
    u = list(p.keys())
    counts = np.asarray([np.sum(x == i) for i in u])
    prob = np.asarray([p[i] for i in u])
    expec = prob * len(x)
    _, pval = sp.stats.chisquare(counts, expec)
    return pval

def digit_preference(x):
    '''Performs a chi-squared test on Benford's law given a set of numbers.
    
    Parameters:
        x : np.ndarray
            The vector containing numbers (ints or floats).
    
    Returns:
        pval : float
            The p-value from the test.
    '''
    # Benford's law
    bp = {i: np.log10(1 + 1 / i) for i in range(1, 10)}
    leading = np.asarray([str(i)[0] for i in x], dtype=int)
    pval = _chisq(leading, bp)
    
    return pval

def digit_difference(x, y, leading=True):
    '''Performs a chi-squared test on the leading or tailing digits of x
    compared to the distribution of those of y.
    
    Parameters:
        x : np.ndarray
            The vector containing observed numbers to compare (ints or floats).
        y : np.ndarray
            The vector containing expected numbers to compare (ints or floats).
    
    Returns:
        pval : float
            The p-value from the test.
    '''
    if leading:
        x_ = np.asarray([str(i)[0] for i in x], dtype=int)
        y_ = np.asarray([str(i)[0] for i in y], dtype=int)
    else:
        x_ = np.asarray([str(np.floor(i).astype(int))[-1] for i in x], dtype=int)
        y_ = np.asarray([str(np.floor(i).astype(int))[-1] for i in y], dtype=int)
        print(x_)
    uy = np.unique(y_, return_counts=True)
    p = dict(zip(uy[0], uy[1] / len(y_)))
    pval = _chisq(x_, p)
    
    return pval

def categorical_difference(x, y):
    '''Performs a chi-squared test comparing x to the distribution of y.
    
    Parameters:
        x : np.ndarray
            The vector containing observations to compare.
        y : np.ndarray
            The vector containing expecteds to compare.
        
    Returns:
        pval : float
            The p-value from the test.
    '''
    uy = np.unique(y, return_counts=True)
    p = dict(zip(uy[0], uy[1] / len(y)))
    pval =_chisq(x, p)
    
    return pval

def missingness_test(x, y):
    '''Performs a chi-squared test comparing the missingness distribution of x
    to that of y.
    
    Parameters:
        x : np.ndarray
            The vector containing observations to compare.
        y : np.ndarray
            The vector containing expecteds to compare.
        
    Returns:
        pval : float
            The p-value from the test.
    '''
    x_ = np.isnan(x)
    y_ = np.isnan(y)
    pval = categorical_difference(x_, y_)
    
    return pval

def ks_test(x, y):
    '''Performs a univariate Kolmogorov-Smirnov test.'''
    _, pval = sp.stats.ks_2samp(x, y)
    
    return pval

def correlation_difference(X, Y):
    '''Performs a bootstrapped test of the difference in correlations.
    
    Parameters:
        X : np.ndarray
            The matrix containing observations to compare.
        Y : np.ndarray
            The matrix containing the observations from the base distribution to
            compare to.
        
    Returns:
        pval : float
            The p-value from the test.
    '''
    n = X.shape[0]
    N = Y.shape[0]
    r = np.corrcoef(X.T)
    R = np.corrcoef(Y.T)
    null_r = []
    nbootstrap = 1000
    for _ in range(nbootstrap):
        sel = np.random.choice(N, size=n)
        null_r.append(np.corrcoef(Y[sel, :].T))
    
    d = np.sum((r - R)**2)
    dnull = np.asarray([np.sum((r - null_ri)**2) for null_ri in null_r])
    pval = np.sum(dnull > d)
    
    return pval / nbootstrap

def multivariate_equality_of_ranks(X, Y):
    '''Perform a multivariate Kruskal-Wallis test on X and Y.
    Reference: doi:10.1080/03610926.2016.1146767
    
    Parameters:
        X : np.ndarray
            The matrix containing observations to compare
        Y : np.ndarray
            A matrix representing observations from the base distribution to
            compare to.
            
    Returns:
        pval : float
            The p-value from the test.
    '''
    n_x, n_y = X.shape[0], Y.shape[0]
    N = n_x + n_y
    
    XY = np.concatenate([X, Y], axis=0)
    rank_XY = np.asarray([np.argsort(XY[:, j]) for j in range(XY.shape[1])]).T + 1
    rank_X = rank_XY[:n_x, :]
    rank_Y = rank_XY[n_x:, :]
    
    U_X = rank_X.mean(axis=0) - (N + 1) / 2
    U_Y = rank_Y.mean(axis=0) - (N + 1) / 2
    V = 0
    for i in range(n_x):
        rxi = rank_X[i, :] - (N + 1) / 2
        V += np.outer(rxi, rxi)
    for i in range(n_y):
        ryi = rank_Y[i, :] - (N + 1) / 2
        V += np.outer(ryi, ryi)
    V /= (N - 1)
    
    Vinv = np.linalg.inv(V)
    W2 = n_x * np.matmul(U_X, np.matmul(Vinv, U_X)) + n_y * np.matmul(U_Y, np.matmul(Vinv, U_Y))
    
    df = X.shape[1]
    pval = 1 - sp.stats.chi2.cdf(W2, df)
    
    return pval

