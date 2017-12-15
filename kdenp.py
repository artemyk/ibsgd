import numpy as np
from scipy.misc import logsumexp
#from sselogsumexp import logsumexp

def get_dists(X):
    """numpy code to compute the pairwise distance matrix for a set of
    vectors specifie by the matrix X.
    """
    x2 = np.expand_dims(np.sum(np.square(X), axis=1), 1)
    dists = x2 + x2.T - 2*np.dot(X, X.T)
    return dists

def get_shape(x):
    dims = x.shape[1]
    N = x.shape[0]
    return dims, N

def kde_entropy_from_dists_loo(dists, x, var):
    dims, N = get_shape(x)
    dists2 = dists + np.eye(int(N)) * 10e20
    dists2 = dists2 / (2*var)
    normconst = (dims/2.0)*np.log(2*np.pi*var)
    lprobs  = logsumexp(-dists2, axis=1) - np.log(N-1) - normconst
    h = -np.mean(lprobs)
    return h

def entropy_estimator(x, var):
    dims, N = get_shape(x)
    dists = get_dists(x)
    dists2 = dists / (2*var)
    normconst = (dims/2.0)*np.log(2*np.pi*var)
    lprobs = logsumexp(-dists2, axis=1) - np.log(N) - normconst
    h = -np.mean(lprobs)
    return dims/2 + h
