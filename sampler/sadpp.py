import numpy as np
import time
from . import quadrature
from . import utils

# Simulated Annealing DPP sampler
# input:
#   L: numpy 2d array, kernel for DPP
#   mix_step: number of mixing steps for Markov chain
#   k: size of sampled subset
#   init_rst: initialization
#   flag_gpu: use gpu acceleration

def sample(L, mix_step, k, init_rst=None, flag_gpu=False, silent=False, func_beta=lambda x:1):
    N = L.shape[0]
    rst = init_rst
    tic_len = mix_step // 5

    # k-dpp annealing
    if rst is None:
        rst = rst = np.random.permutation(N)[:k]
    rst_bar = np.setdiff1d(range(N), rst)

    A = np.copy(L[np.ix_(rst, rst)])

    for i in range(mix_step):
        if silent==False:
            if (i+1) % tic_len == 0:
                print('{}-th iteration.'.format(i+1))
        rem_ind = np.random.randint(k)
        add_ind = np.random.randint(N-k)
        v = rst[rem_ind]
        u = rst_bar[add_ind]

        tmp_rst = np.delete(np.copy(rst), rem_ind)
        tmp_rst_bar = np.delete(np.copy(rst_bar), add_ind)
        tmp_A = np.copy(A)
        tmp_A = np.delete(tmp_A, rem_ind, axis=0)
        tmp_A = np.delete(tmp_A, rem_ind, axis=1)
        bu = np.copy(L[np.ix_([u], tmp_rst)])
        bv = np.copy(L[np.ix_([v], tmp_rst)])

        lambda_min, lambda_max = utils.gershgorin(tmp_A)
        lambda_min = np.max([lambda_min, 1e-5])
        beta = func_beta(i)
        prob = np.random.uniform()**(1/beta)
        tar = prob * L[v,v] - L[u,u]

        flag = quadrature.gauss_kdpp_judge(tmp_A, bu[0], bv[0], prob, tar, lambda_min, lambda_max)

        if flag:
            rst = np.append(tmp_rst, [u])
            rst_bar = np.append(tmp_rst_bar, [v])
            A = np.r_[np.c_[tmp_A, bu.transpose()], np.c_[bu, L[u,u]]]

    return rst




