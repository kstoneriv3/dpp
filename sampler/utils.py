import numpy as np
import scipy

import gzip
import os
from os import path

import sys
if sys.version_info.major < 3:
    import urllib
else:
    import urllib.request as request

def get_eig(L, flag_gpu=False):
    if flag_gpu:
        pass
    else:
        return scipy.linalg.eigh(L)

def get_sympoly(D, k, flag_gpu=False):
    N = D.shape[0]
    if flag_gpu:
        pass
    else:
        E = np.zeros((k+1, N+1))

    E[0] = 1.
    for l in range(1,k+1):
        E[l,1:] = np.copy(np.multiply(D, E[l-1,:N]))
        E[l] = np.cumsum(E[l], axis=0)

    return E


def gershgorin(A):
    radius = np.sum(np.absolute(A), axis=0)
    
    lambda_max = np.max(radius)
    lambda_min = np.min(2 * np.diag(A) - radius)

    return lambda_min, lambda_max


def kpp(X, k, flag_kernel=False):
    # if X is not kernel, rows of X are samples

    N = X.shape[0]
    rst = np.zeros(k, dtype=int)
    rst[0] = np.random.randint(N)

    if flag_kernel:
        # kernel kmeans++
        v = np.ones(N) * np.inf
        for i in range(1, k):
            Y = np.diag(X) + np.ones(N)*X[rst[i-1],rst[i-1]] - 2*X[rst[i-1]]
            v = np.minimum(v,Y)
            r = np.random.uniform()
            rst[i] = np.where(v.cumsum() / v.sum() >= r)[0][0]

    else:
        # normal kmeans++
        centers = [X[rst[0]]]
        for i in range(1, k):
            dist = np.array([min([np.linalg.norm(x-c)**2 for c in centers]) for x in X])
            r = np.random.uniform()
            ind = np.where(dist.cumsum() / dist.sum() >= r)[0][0]
            rst[i] = ind
            centers.append(X[ind])

    return rst

def kl_gaussian(mu0, sigsq0, mu1, sigsq1):
    return 1/2 * (np.log(sigsq1)-np.log(sigsq0) + sigsq0/sigsq1 - 1 + (mu0-mu1)**2/sigsq1)

def kl_gaussian_vec(vec):
    [mu0, sigsq0, mu1, sigsq1] = list(vec)
    return kl_gaussian(mu0, sigsq0, mu1, sigsq1)

def kl_multi_gaussian(mu0, S0, mu1, S1):
    
    assert(len(mu0)==len(mu1))
    assert(S0.shape==S1.shape)
    
    k = len(mu0)
    invS1 = np.linalg.inv(S1)
    tr = np.sum(invS1*S0)
    quad = ((mu1-mu0) @ invS1 @ (mu1-mu0))
    logdet = np.linalg.slogdet(S1)[1] - np.linalg.slogdet(S0)[1]
    
    return 1/2 * (tr + quad - k + logdet)