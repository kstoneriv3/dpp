import numpy as np
from scipy.linalg import inv, norm, det
from scipy.spatial.distance import pdist, squareform
import pandas as pd

# find optimal hyperparameter for Gaussian Process
# and return kernel matrix and so on
def full_GP(sigma2_0, nTrn, nTot, data, trnY, tstY):
    
    pairwise_dists = squareform(pdist(data[:,:-1], 'euclidean'))
    L_0 = np.exp(-pairwise_dists ** 2 / 100 ** 2)
    trnL_0 = L_0[:nTrn, :nTrn]
    inv_mat_0 = inv(trnL_0 + sigma2_0 * np.identity(nTrn))
    alpha_0 = inv_mat_0 @ trnY
    
    # correct the scaling factor
    # Newton like method for optimization of hyperparameter
    scale = trnY.var()/sigma2_0
    for i in range(100):
        dscale = - nTrn/(2*scale) + 1/(2*scale**2) * trnY @ inv_mat_0 @ trnY
        ddscale = (nTrn/2*scale**2) - 1/(scale**3) * trnY @ inv_mat_0 @ trnY
        scale -= 0.1*dscale/ddscale
    
    sigma2 = scale * sigma2_0
    L = scale * L_0
    trnL = scale * trnL_0
    
    inv_mat = inv(trnL + sigma2 * np.identity(nTrn))
    alpha = inv_mat @ trnY
    Y_hat_full = L[:,:nTrn]@ alpha
    Var_Y_hat_full = L - L[:,:nTrn] @ inv_mat @ L[:nTrn,:]
    MSE_full_trn = norm(Y_hat_full[:nTrn] - trnY)
    MSE_full_tst = norm(Y_hat_full[nTrn:] - tstY)
    
    return [sigma2, L, trnL, Y_hat_full, Var_Y_hat_full, MSE_full_trn, MSE_full_tst]


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

def compute_scores(smpl, trnY, tstY, trnL, L, Y_hat_full, Var_Y_hat_full, sigma2):
    
    scores = {}
    k = len(smpl)
    nTrn = trnL.shape[0]
    nTot = L.shape[0]
    C = trnL[np.ix_(range(nTrn), smpl)]
    W = C[np.ix_(smpl, range(k))]
    trnL_prime = C @ inv(W) @ C.T
    W_inv = inv(W)
    CC = C.T @ C
    A = CC @ W_inv
    B = inv(sigma2*W + CC)
    Y_hat = L[np.ix_(range(nTot), smpl)] @ W_inv @ (1/sigma2*(C.T @ trnY - CC @ B @ C.T @ trnY) )
    Var_Y_hat = L + \
                L[np.ix_(range(nTot), smpl)] @ (
                    - (1/sigma2*(W_inv @ A - A.T @ B @ A) )
                ) @ L[np.ix_(range(nTot), smpl)].T
    tmp = np.stack([
            Y_hat_full[nTrn:], np.diag(Var_Y_hat_full)[nTrn:] + sigma2, 
            Y_hat[nTrn:],      np.diag(Var_Y_hat)[nTrn:] + sigma2
        ]).T
    
    scores["logdet"] = np.linalg.slogdet(W)[1]
    scores["matrix error"] = norm(trnL_prime - trnL, 'fro')
    scores["in-sample MSE"] = norm(Y_hat[:nTrn] - trnY)
    scores["out-of-sample MSE"] = norm(Y_hat[nTrn:] - tstY)
    scores["in-sample KL"] = kl_multi_gaussian(Y_hat, Var_Y_hat, Y_hat_full, Var_Y_hat_full)
    scores["out-of-sample KL"] = np.apply_along_axis(arr=tmp,axis=1,func1d=kl_gaussian_vec).mean()
    return pd.Series(scores)