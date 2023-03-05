import random
from functools import lru_cache
from collections import defaultdict

import numpy as np
import torch
import argparse


from itertools import permutations
from math import factorial

from vol_utils.data_utils import get_datasets
# from vol_utils.volume import replicate

from exp_utils import exact, generic_sampler, kernelSHAP, active, _get_SV_estimates

# Reproducebility
seed = 1234
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# ---------- DATA PREPARATION ----------


parser = argparse.ArgumentParser(description='Configure the experiment.')


parser.add_argument('-d', '--dataset', help='Dataset name', type=str, required=True, choices=['linear', 'friedman', 'hartmann', 'used_car', 'uber_lyft', 'credit_card', 'hotel_reviews'])
parser.add_argument('-n', '--n', help='The number of players', type=int, default=5)
parser.add_argument('-m', '--num_samples', help='The number of samples (i.e., permutations for SV estimation).', type=int, default=1000)
parser.add_argument('-M', '--train_size', help='Size of the dataset of each player.', type=int, default=200, choices=[30, 50, 100, 200])
parser.add_argument('-t', '--trials', help='Number of independent random trials to run.', type=int, default=5)

# parser.add_argument('-size', '--size', help='Making the dataset sizes differ (increasing trend).', type=bool, default=False)
# parser.add_argument('-superset', '--superset', help='Making the support of datasets being superset of previous one.', type=bool, default=False)
# parser.add_argument('-disjoint', '--disjoint', help='Making the support of datasets disjoint.', type=bool, default=False)

args = parser.parse_args()
print(args)

# -----------------------------
train_sizes = [args.train_size for _ in range(args.n)]

feature_datasets, feature_datasets_test, labels, test_labels, D =  get_datasets(args.dataset, args.n, train_sizes)

test_sizes = [len(test_dataset) for test_dataset in feature_datasets_test]

# ---------- DATA VALUATIONS ----------
res = {}

"""
Direct Volume-based values

"""
from vol_utils.volume import compute_volumes, compute_pinvs, compute_X_tilde_and_counts, compute_robust_volumes

# train_features = [dataset.data for dataset in train_datasets]
volumes, vol_all = compute_volumes(feature_datasets, D)

volumes_all = np.asarray(list(volumes) + [vol_all])
print('-------Volume Statistics ------')
print("Original volumes: ", volumes, "volume all:", vol_all)
res['vol'] = volumes


from scipy.stats import sem

"""
Discretized Robust Volume-based Shapley values
"""

@lru_cache(maxsize=min(2**args.n, 1024))
def _utility(S, omega):
    if len(S) == 0:
        return 0
    else:
        curr_train_X = torch.cat([dataset for j, dataset in enumerate(feature_datasets) if j in S]).reshape(-1, D)
        # curr_vol = torch.sqrt(torch.linalg.det(curr_train_X.T @ curr_train_X) + 1e-8)
        X_tilde, cubes = compute_X_tilde_and_counts(curr_train_X, omega)
        robust_vol = compute_robust_volumes([X_tilde], [cubes])[0]

        return robust_vol

def RV_utility(S, omega=0.1): 
    return _utility(tuple(sorted(S)), omega)


SV_trials = defaultdict(list)
statistics_trials = defaultdict(list)


for t in range(args.trials):

    exact_SV = exact(args.n, RV_utility)

    SV_trials['Exact'].append(exact_SV / np.mean(exact_SV))

    # kernelSHAP
    mcs, afs, min_afs = kernelSHAP(args.n, args.num_samples, RV_utility, seed=t, bootstrap_n=500)
    s_estimates = _get_SV_estimates(args.n, mcs)
    SV_trials['KernelSHAP'].append(s_estimates)
    statistics_trials['KernelSHAP'].append((mcs, afs, min_afs))

    for alpha in [0, 2, 5, 100]:

        # active: 2-FAE
        mcs, afs, min_afs = active(args.n, args.num_samples, RV_utility, seed=t, bootstrap_n=500, alpha=alpha)
        s_estimates = _get_SV_estimates(args.n, mcs)
        SV_trials['2-FAE-a'+str(alpha)].append(s_estimates)
        statistics_trials['2-FAE-a'+str(alpha)].append((mcs, afs, min_afs))

    # other samplers
    for method in ['Sobol', 'Stratified', 'Owen', 'MC']:
        # print(f'Executing {method} now.')
        mcs, afs, min_afs = generic_sampler(method, args.n, args.num_samples, RV_utility, seed=t, bootstrap_n=500)
        s_estimates = _get_SV_estimates(args.n, mcs)
        SV_trials[method].append(s_estimates)
        statistics_trials[method].append((mcs, afs, min_afs))


print('------Robust Volume Shapley value Statistics ------')
for method, SV_over_trials in SV_trials.items():
    SV_over_trials = np.stack(SV_over_trials)
    mean, std_err = np.mean(SV_over_trials, 0), sem(SV_over_trials, axis=0)
    print(f'RVSV {method}: mean {mean}, std-err {std_err}.')

    res[method] = mean
    res[method+'SV_over_trials'] = SV_over_trials
    res[method +'statistics'] = statistics_trials[method]
print('-------------------------------------')

# suffix = '_rep' if rep else '' + '_superset' if superset else '' + '_train_test_diff_distri' if train_test_diff_distr else '' + '_size' if size else '' + '_disjoint' if disjoint else ''

from os.path import join as oj
from vol_utils.utils import cwd
with cwd(oj('local_results-unif', 'vol')):
    np.savez(f'res_{args.dataset} {D}D_{args.n}n_{args.trials}t_{args.num_samples}samples.npz',
        res=res, n=args.n, t=args.trials, D=D, num_samples=args.num_samples, train_sizes=train_sizes, test_sizes=test_sizes, dataset=args.dataset, seed=seed)

exit()

