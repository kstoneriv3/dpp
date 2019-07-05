import numpy as np
import time
from scipy.linalg import sqrtm, inv

# DPP sampler
# input:
#   k: size of sampled subset

inv = lambda A: np.linalg.inv(A)
logdet = lambda A: np.linalg.slogdet(A)[1]

def compute_logL(smpl, trnL, trnY, nTrn, sigma2):
    
    k = len(smpl)
    C = trnL[np.ix_(range(nTrn), smpl)]
    W = C[np.ix_(smpl, range(k))]
    CC = C.T @ C
    logdet_= logdet(W + 1/sigma2*CC) - logdet(W) + nTrn*np.log(sigma2)
    quad = 1/sigma2 * (trnY @ trnY - trnY @ C @ inv(sigma2*W + CC) @ C.T @trnY)
    logL = (1/2) * (- logdet_ - nTrn*np.log(2*np.pi) - quad)
    return logL

def sample(k, batch_size, n_iter, trnL, trnY, nTrn, sigma2):

    #initialize
    perm = np.random.permutation(nTrn)
    smpl = perm[:k]
    non_smpl = perm[k:]
    logL = compute_logL(smpl, trnL, trnY, nTrn, sigma2)
    
    #itaration
    for i in range(n_iter):
        perm_smpl = np.random.permutation(k)
        perm_non_smpl = np.random.permutation(nTrn - k)
        smpl_tmp = np.concatenate([
                        smpl[perm_smpl[:k-batch_size]],
                        non_smpl[perm_non_smpl[:batch_size]]
        ])
        logL_tmp = compute_logL(smpl_tmp, trnL, trnY, nTrn, sigma2)
        if logL_tmp > logL:
            non_smpl = np.concatenate([
                        smpl[perm_smpl[k-batch_size:]],
                        non_smpl[perm_non_smpl[batch_size:]]
            ])
            smpl = smpl_tmp
            logL = logL_tmp
            
    return smpl
        