import math
import numpy as np


def owen_sam_shap(runner, runs=2, q_splits=100):
    n = runner.X.shape[0]
    mcs = {}
    s = []
    for _ in range(runs):
        for q_num in range(q_splits + 1):
            q = q_num / q_splits
            s.append(np.array(np.random.binomial(1, q, n)))
    s = np.array(s)

    eval_counts = 0

    for i in range(n):
        for j in range(s.shape[0]):
            cur_s = s[j].copy()
            cur_s[i] = 0
            X_cur = np.concatenate([runner.X[cur_s == 1], runner.init_X])
            y_cur = np.concatenate([runner.y[cur_s == 1], runner.init_y])
            runner.model.fit(X_cur, y_cur)
            v1 = runner.value()
            X_cur = np.concatenate([X_cur, [runner.X[i]]])
            y_cur = np.concatenate([y_cur, [runner.y[i]]])
            runner.model.fit(X_cur, y_cur)
            v2 = runner.value()
            if i not in mcs:
                mcs[i] = [v2 - v1]
            else:
                mcs[i].append(v2 - v1)
            eval_counts += 1
    return mcs, eval_counts


def owen_sam_shap_halved(runner, runs=2, q_splits=100):
    n = runner.X.shape[0]
    mcs = {}
    s = []
    for _ in range(runs):
        for q_num in range(q_splits // 2 + 1):
            q = q_num / q_splits
            b = np.array(np.random.binomial(1, q, n))
            s.append(b)
            if q != 0.5:
                s.append(1 - b)
    s = np.array(s)

    eval_counts = 0

    for i in range(n):
        for j in range(s.shape[0]):
            cur_s = s[j].copy()
            cur_s[i] = 0
            X_cur = np.concatenate([runner.X[cur_s == 1], runner.init_X])
            y_cur = np.concatenate([runner.y[cur_s == 1], runner.init_y])
            runner.model.fit(X_cur, y_cur)
            v1 = runner.value()
            X_cur = np.concatenate([X_cur, [runner.X[i]]])
            y_cur = np.concatenate([y_cur, [runner.y[i]]])
            runner.model.fit(X_cur, y_cur)
            v2 = runner.value()
            if i not in mcs:
                mcs[i] = [v2 - v1]
            else:
                mcs[i].append(v2 - v1)
            eval_counts += 1
    return mcs, eval_counts



def stratified_sampling(runner, num_samples):
    n = runner.X.shape[0]
    div = sum([(j+1)**(2/3) for j in range(n)])
    for i in range(n):
        quota = [0 for _ in range(n)]
        for k in range(n):
            quota[k] = math.floor(num_samples * (k+1)**(2/3) / div)
        left = num_samples - sum(quota)
        for j in range(left):
            quota[j] += 1
        for k in range(n):
            players = [r for r in range(n) if i != r]
            samples = [np.random.permutation(players) for _ in range(quota[k])]
            for i in range(len(samples)):
                samples[i] = np.insert(samples[i], i, k)



# Sobol sequence sampler helper functions

from scipy.optimize import root_scalar
from scipy.stats.qmc import Sobol # needs scipy version 1.7 or later

def f_last(psi):
    return 0.5 * np.pi

from scipy.special import beta
def f_mid(psi, j, d):
    assert 1 <= j < d-2
    return 1./beta(0.5*(d-j-1), 0.5) * np.power(np.sin(psi), d-j-2)

from scipy.integrate import quad
def cdf_F(psi, j, d):
    assert j <= d-2, "j must be <= d-2"

    if j == d - 2:
        return quad(f_last, 0, psi)
    else:
        return quad(f_mid, 0, psi, args=(j,d))

from math import ceil
def get_U_hat(d, norm_ord=1):
    U = np.zeros((d-1, d))

    for i in range(d-1):
        for j in range(d):
            for s in range(d-1):
                if i == s and j == i+1:
                    U[i,j] = -(s+1)
                elif i <= s and i >= j:
                    U[i,j] = 1   
      
    from numpy.linalg import norm
    return U / norm(U, axis=1, ord=norm_ord)[:,None]

def PolarToCartesian(r, psis):
    x = np.zeros(len(psis) + 1) # shape of x should be d -1, and since len(psis) = p-2, initialize x = np.zeros(len(psis)+1) 

    for i in range(len(x)):
        x[i] = r
        for j in range(i):
            x[i] *= np.sin(psis[j])
        
        if i < len(psis):
            x[i] *= np.cos(psis[i])

    return x

def SobolPermutations(num_samples, dimension, seed=3244, verbose=False):
    '''
    num_samples: the number of permutations to sample
    dimension: the number of players, i.e., the dimension of the permutation

    '''

    sampled_permutations = []
    U_hat = get_U_hat(dimension)

    sampler = Sobol(d=dimension-2, scramble=True, seed=seed)
    sobol_sequence = sampler.random_base2(m=ceil(np.log2(num_samples)))
    for sobol_point in sobol_sequence:
        psis = np.zeros(dimension-2)
        for j in range(dimension-2):

            target = sobol_point[j]        
            sol = root_scalar(lambda x, *args:cdf_F(x, *args)[0] - target, args=(j+1, dimension), bracket=(0, np.pi))
            psis[j] = sol.root

        y = PolarToCartesian(1, psis)        
        # print(f'shape of y is {y.shape}, shape of U_hat is {U_hat.shape}')
        z = U_hat.T @ y
        # print(f'shape of z is {z.shape}')
        sampled_permutations.append(np.argsort(z))

    if verbose and num_samples != len(sampled_permutations):
        print(f'requested num_samples is {num_samples}, number of sampled permutations is {len(sampled_permutations)}, returning the first {num_samples} sampled permutations.')
        print('It is advised to sample a number that is an exact power of 2, of permutations to enjoy the theoretical properties of Sobol sequence.')

    return sampled_permutations[:num_samples]



# Improved KernelSHAP sampler helper functions
def utility(S):
    # S is a binary bit indicating which player is in
    # e.g., S = '0b001' means on the last player is in the coalition
    # e.g., S = '0b1001' means the last player and the fourth last player are in the coalition

    # need to be careful with the ordering, can use str.zfill(n) to simplify logic
    # e.g., S = '0b001', S.replace('0b','').zfill(n) = '0000000001' for n = 10

    return eval(S) # it is a placeholder to be modified


def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)

def kernel_mu(s, n):
    # the kernel in Improving KernelSHAP. mu_sh()
    if s == 0 or s == n:
        return float('inf')
    return  (n-1.) / ( nCr(n, s)  * s* (n - s))


def _get_A_inv(n):
    '''
    n: number of players
    '''

    # filling in the non-diagonal entries which are all the same
    constA = 1. / (n *(n-1))
    numer = sum( [(k-1)/(n-k) for k in range(2, n)] )
    denom = sum( [1./(k*(n-k)) for k in range(1, n)] )

    A = np.ones( (n,n)) * constA * numer / denom

    # filling in the diagonal entries first
    np.fill_diagonal(A, 0.5)
    # print(A)
    return  np.linalg.inv(A)


def get_kernelSHAP_estimate(num_samples, n, seed=1234):
    '''
    num_samples: number of coalitions/permutations to sample
    n: numbe of players
    '''


    if seed:    
        import random
        random.seed(3244)

    probs = {}
    sum_probs = 0
    for bit in range(1, 2**n-1):
        prob = kernel_mu( bin(bit).count("1"), n)
        sum_probs += prob

        # for each bit that represents a coalition: 00100, the 3rd out of 5 players is in the coalition, i.e., a singleton coalition
        # we increment the expected_Z[i] by prob * 1
        # note the str for bit is enumerated in reverse since bin(bit) does not automatically produce fixed length,
        # so we need to iterate from right to left
        '''
        for i, present in enumerate(bin(bit).replace('0b','')[::-1]):
            expected_Z[i] += prob * (present=='1')
        '''
        probs[bin(bit)] = prob

    # for kernel_mu defined, the expected_Z takes form [0.5 for _ in range(n)]
    expected_Z = np.asarray([0.5 for _ in range(n)])
    A_inv = _get_A_inv(n)

    coalition_bits = random.choices(list(probs.keys()), weights=list(probs.values()), k=num_samples)

    b_bars = []
    b_bar = np.zeros(n)
    single_sample_SV_estimates = []

    for coalition_bit in coalition_bits:
        U = utility(coalition_bit)

        coalition_bit = coalition_bit.replace('0b','').zfill(n)
        coalition_bit_vector = np.array([int(bit) for bit in coalition_bit])

        b_bar += U * coalition_bit_vector

        # sample-one by one

        single_b_bar = U * coalition_bit_vector - expected_Z * utility(bin(0))
        b_bars.append(single_b_bar)


        numer = np.ones(n)@ A_inv @ single_b_bar - utility(bin(2**n-1)) + utility(bin(0))
        denom = (np.ones(n) @ A_inv @ np.ones(n))

        # multiply by A_inv with (single_b_bar - 1 * (numer/denom))
        single_sample_SV_estimate = A_inv @ (single_b_bar - np.ones(n)* numer / denom)

        single_sample_SV_estimates.append(single_sample_SV_estimate)



    #  joint sample estimate: calculate the numerator and denominator in Equ (9)
    b_bar /= num_samples
    b_bar -= expected_Z * utility(bin(0))

    numer = np.ones(n)@ A_inv @ b_bar - utility(bin(2**n-1)) + utility(bin(0))
    denom = (np.ones(n) @ A_inv @ np.ones(n))

    # multiply by A_inv with (b_bar - 1 * (numer/denom))
    SV_estimates = A_inv @ (b_bar - np.ones(n)* numer / denom)

    print(f"SV estimates from direct vectorisation:\n{SV_estimates}")

    # single sample estimate

    single_sample_SV_estimate_final = np.asarray(single_sample_SV_estimates).mean(axis=0)

    print(f"SV estimates from single sample and then mean computation:\n{single_sample_SV_estimate_final}")

    print(f"both calculations are equal: {np.allclose(SV_estimates, single_sample_SV_estimate_final)}")

    return SV_estimates