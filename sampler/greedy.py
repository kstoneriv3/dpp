import numpy as np
import time
from numpy.linalg import inv, LinAlgError

# DPP sampler
# input:
#   k: size of sampled subset

logdet = lambda A: np.linalg.slogdet(A)[1]

def compute_logL(smpl, L, trnY, nTrn, sigma2):
    
    n = L.shape[0]
    k = len(smpl)
    C = L[np.ix_(range(nTrn), smpl)]
    W = L[np.ix_(smpl, smpl)]
    CC = C.T @ C
    
    try:
        logdet_= logdet(W + 1/sigma2*CC) - logdet(W) + nTrn*np.log(sigma2)
        quad = 1/sigma2 * (trnY @ trnY - trnY @ C @ inv(sigma2*W + CC) @ C.T @trnY)
        logL = (1/2) * (- logdet_ - nTrn*np.log(2*np.pi) - quad)
        return logL
    
    except LinAlgError as e:
        if e.args[0] == "Singular matrix":
            return - np.infty
    else:
            raise(e)
            
    

def sample(k, batch_size, n_iter, L, trnY, nTrn, sigma2):
    
    # first nTrn indices of L needs to correspond to training data.
    # e.g. L[:nTrn, :nTrn] is a kernel of training data, 
    #      L[nTrn:, nTrn] is a kernel of test data (or any data which is not used in training)
    # results are sampled from whole indices (train and test combined)
    
    #initialize
    assert(L.shape[0]==L.shape[1])
    n = L.shape[0]
    perm = np.random.permutation(n)
    smpl = perm[:k]
    non_smpl = perm[k:]
    logL = compute_logL(smpl, L, trnY, nTrn, sigma2)
    
    #itaration
    for i in range(n_iter):
        perm_smpl = np.random.permutation(k)
        perm_non_smpl = np.random.permutation(n - k)
        smpl_tmp = np.concatenate([
                        smpl[perm_smpl[:k-batch_size]],
                        non_smpl[perm_non_smpl[:batch_size]]
        ])
        logL_tmp = compute_logL(smpl_tmp, L, trnY, nTrn, sigma2)
        if logL_tmp > logL:
            non_smpl = np.concatenate([
                        smpl[perm_smpl[k-batch_size:]],
                        non_smpl[perm_non_smpl[batch_size:]]
            ])
            smpl = smpl_tmp
            logL = logL_tmp
            
    return smpl
        