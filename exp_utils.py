import cProfile

import random
from collections import defaultdict
from copy import deepcopy
from functools import lru_cache

import numpy as np

def utility(S):
    return _utility(tuple(sorted(S)))


@lru_cache(maxsize=1024)
def _utility(S):
    # S is a tuple of the sorted list. 
    # Using a tuple so as to use lru_cache since a list is an unhashable type
    return 1.0/(len(S)+1) + np.random.normal()


# Samplers
def _owen(n, num_samples, **kwargs):
    runs = kwargs.get('runs', 2)
    q_splits = num_samples // n - 1
    samples = defaultdict(list)
    weights = defaultdict(list)
    for _ in range(runs):
        for q_num in range(q_splits // runs + 1):
            q = q_num / q_splits
            b = np.array(np.random.binomial(1, q, n))
            ids = list(np.nonzero(b)[0])
            rest = [j for j in range(n) if j not in ids]
            for i in ids:
                first = [j for j in ids if j != i]
                perm = first + [i] + rest

                samples[i].append(perm)
                weights[i].append(1.0)
            if q != 0.5:
                ids, rest = rest, ids
                for i in ids:
                    first = [j for j in ids if j != i]
                    perm = first + [i] + rest

                    samples[i].append(perm)
                    weights[i].append(1.0) 

    return samples, weights


import math
def _stratified(n, num_samples):
    num_samples = num_samples // n

    samples = defaultdict(list)
    weights = defaultdict(list)
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
            s = [np.random.permutation(players) for _ in range(quota[k])]
            for r in range(len(s)):
                samples[k].append(np.insert(s[r], i, k))
                weights[k].append(1.0)

    return samples, weights


def _MC(n, num_samples):
    samples = defaultdict(list)
    weights = defaultdict(list)
    for i in range(n):
        for _ in range(num_samples // n):
            perm = np.random.permutation(n)
            samples[i].append(perm)
            weights[i].append(1.0)

    return samples, weights
# samples, weights = _MC(n, num_samples)


from tools.other_estimators import SobolPermutations
def _Sobol(n, num_samples):

    samples = defaultdict(list)
    weights = defaultdict(list)
    for i in range(n):
        samples[i] = SobolPermutations(num_samples//n, n, seed=i)
        weights[i] = np.ones(len(samples[i]))

    return samples, weights
# samples, weights = _Sobol(n, num_samples)


from tools.perm_sampler import PermutationSampler
from tools.utils import get_afs, get_mc_sum_by_card

def _bootstrap(n, num_samples, utility, seed=1234):
    sampler = PermutationSampler(n, seed)
    samples, weights = sampler.sample_sequential(num_samples, method='random')
    return _get_mcs(utility, samples, weights)



def _get_mcs(utility, samples, weights=[]):
    if len(weights) == 0:
        weights = np.ones(len(samples))
    else:
        assert len(samples) == len(weights)

    mcs = defaultdict(list)

    for permutation, weight in zip(samples, weights):
        coalition = []
        curr_utility = utility(coalition)
        for pos, i in enumerate(permutation):
            new_utility = utility(coalition + [i])

            mc = weight * (new_utility - curr_utility)
            mcs[i].append((mc, len(coalition)))

            curr_utility = new_utility
            coalition.append(i)

    return mcs

from tools.other_estimators import _get_A_inv
from tools.utils import get_afs
def kernelSHAP(n, num_samples, utility, xi=1e-3, seed=1234, bootstrap_n=100):
    if seed: random.seed(seed)

    expected_Z = np.asarray([0.5 for _ in range(n)])
    A_inv = _get_A_inv(n)

    # setting up weights for sampling according to the kernel
    weights = np.arange(1, n)
    weights = 1 / (weights * (n - weights))
    weights = weights / np.sum(weights)

    S = np.zeros((num_samples, n), dtype=bool)
    num_included = np.random.choice(n - 1, size=num_samples, p=weights) + 1

    # updating the indices of players in the sampled permutations S of shape (num_samples, num_players)
    for row, num in zip(S, num_included):
        inds = np.random.choice(n, size=num, replace=False)
        row[inds] = 1


    GRAND, NULL = range(n), []
    b_bars, SV_estimate_list = [], []
    b_bar = np.zeros(n)
    mcs = _bootstrap(n, bootstrap_n, utility, seed=seed)


    min_afs = []

    # estimating SV or mcs according to the sampled permutations S
    for sample in S:

        coalition = [i for i, bit in enumerate(sample) if bit]
        coalition_bit_vector = np.asarray(sample, dtype=int)

        U = utility(coalition)

        single_b_bar = U * coalition_bit_vector - expected_Z * utility(NULL)
        b_bar += single_b_bar
        # sample-one by one
        b_bars.append(single_b_bar)

        numer = np.ones(n)@ A_inv @ single_b_bar - utility(GRAND) + utility(NULL)
        denom = (np.ones(n) @ A_inv @ np.ones(n))

        # multiply by A_inv with (single_b_bar - 1 * (numer/denom))
        SV_estimate = A_inv @ (single_b_bar - np.ones(n)* numer / denom)

        SV_estimate_list.append(SV_estimate)

        '''
        coalition_cardinality = len(coalition)
        for i in coalition: # coalition is a list of indices of players in this coalition
            mc = SV_estimate[i]
            cardinality = coalition_cardinality - 1

            mcs[i].append( (mc, cardinality) )
        '''
        for i in range(n):
            mcs[i].append((SV_estimate[i], None))

        afs = get_afs(mcs, xi)
        min_idx = np.argmin(afs)
        min_afs.append(afs[min_idx])

    # SV_estimates = np.asarray(SV_estimate_list).mean(axis=0) # a vector of dimension self.n_sources

    return mcs, afs, min_afs


def active(n, num_samples, utility, seed=1234, xi=1e-3, bootstrap_n=100, **kwargs):
    mcs = _bootstrap(n, bootstrap_n, utility, seed)

    sampler = PermutationSampler(n, seed)
    mc_by_card = get_mc_sum_by_card(mcs)
    sampler.estimate_parameters(mc_by_card, alpha=kwargs.get('alpha', 2))

    min_afs = []
    for _ in range(num_samples):
        afs = get_afs(mcs, xi)
        target_idx = np.argmin(afs)
        min_afs.append(afs[target_idx])

        single_sample, single_weight = sampler.sample_one_point(target_idx, 1)        
        single_sample, single_weight= single_sample[target_idx], single_weight[target_idx]

        cur_mcs = _get_mcs(utility, single_sample, single_weight)

        for i in cur_mcs.keys():
            if i in mcs.keys():
                mcs[i] += cur_mcs[i]

    return mcs, afs, min_afs


from tools.utils import get_afs
def generic_sampler(method, n, num_samples, utility, seed=1234, xi=1e-3, bootstrap_n=100):

    if method.lower() == 'owen':
        samples, weights = _owen(n, num_samples)

    elif method.lower() == 'stratified':
        samples, weights = _stratified(n, num_samples)

    elif method.lower() == 'mc':
        samples, weights = _MC(n, num_samples)

    elif method.lower() == 'sobol':
        samples, weights = _Sobol(n, num_samples)

    else:
        raise NotImplementedError
    
    if method not in ['owen', 'stratified', 'MC', 'Sobol']:
        # print("Do NOT ignore weights.")
        pass

    samples_ = deepcopy(samples)
    # print(f'For {method}, the sampled permutations counts are: {[len(value) for value in samples.values()]}')

    # mcs = defaultdict(list)
    mcs = _bootstrap(n, num_samples=bootstrap_n, utility=utility, seed=seed)

    min_afs = []
    while len(list(samples.keys())) > 0:
        picked_i = random.choice(list(samples.keys()))
        permutation = samples[picked_i].pop()
        
        if len(samples[picked_i]) == 0:
            samples.pop(picked_i, None)

        pos_i = list(permutation).index(picked_i)
        mc = utility(permutation[:pos_i+1]) - utility(permutation[:pos_i])
        mcs[picked_i].append( (mc, pos_i) )

        afs = get_afs(mcs, xi)
        min_idx = np.argmin(afs)
        min_afs.append(afs[min_idx])

    return mcs, afs, min_afs



from itertools import chain, combinations

def _powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    powerset = chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    return [list(subset) for subset in powerset]

from math import factorial as fac
def exact(n, utility):

    SVs = np.zeros(n)
    coeffs = [fac(c) * fac(n-c-1) / fac(n) for c in range(n) ] # the coeffcieint of a coalition of size c
    powerset = _powerset(range(n))

    for i in range(n):
        for coalition in powerset:
            if i in coalition:
                continue
            else:
                marginal_contribution = utility(coalition + [i]) -  utility(coalition)
                SVs[i] += marginal_contribution * coeffs[len(coalition)]
    return SVs

def _get_SV_estimates(n, mcs, normalize=True):
    '''
    mcs: a dict of lists
    <key, value>: <i, list of mcs>
    eash list of mcs = [[mc, card], ..., [mc, card]]

    '''
    SV_estimates = np.zeros(n)
    for i in range(n):
        numeric_mcs = [mc for mc, _ in mcs[i]]
        SV_estimates[i] = np.mean(numeric_mcs)

    if normalize:
        return SV_estimates / np.mean(SV_estimates)
    else:
        return SV_estimates

if __name__ == '__main__':

    n = 10
    num_samples = 1000 # total number of samples across all n players

    # cProfile.run('kernelSHAP(n, num_samples, utility)')
    # cProfile.run('mcs, afs, min_afs = active(n, num_samples, utility)')
    # cProfile.run('generic_sampler(\'MC\', n, num_samples, utility)')

    # cProfile.run('exact(n, utility)')

    print(exact(n, utility))
    mcs, afs, min_afs = kernelSHAP(n, num_samples, utility)
    mcs, afs, min_afs = active(n, num_samples, utility)
    mcs, afs, min_afs = generic_sampler('Sobol', n, num_samples, utility)
    mcs, afs, min_afs = generic_sampler('owen', n, num_samples, utility)
    mcs, afs, min_afs = generic_sampler('stratified', n, num_samples, utility)
    mcs, afs, min_afs = generic_sampler('MC', n, num_samples, utility)



