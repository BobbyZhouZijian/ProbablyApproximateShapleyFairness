import random
from functools import lru_cache
from collections import defaultdict

import numpy as np
import torch
import copy
import argparse

from itertools import permutations
from math import factorial

from vol_utils.data_utils import get_datasets

from exp_utils import exact, generic_sampler, kernelSHAP, active, _get_SV_estimates


# Reproducebility
seed = 1234
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
DATA_PATH = "./experiment_data"

# ---------- DATA PREPARATION ----------


parser = argparse.ArgumentParser(description='Configure the experiment.')


parser.add_argument('-d', '--dataset', help='Dataset name', type=str, required=True, choices=['linear', 'friedman', 'hartmann', 'used_car', 'uber_lyft', 'credit_card', 'hotel_reviews'])
parser.add_argument('-n', '--n', help='The number of players', type=int, default=5)
parser.add_argument('-m', '--num_samples', help='The number of samples (i.e., permutations for SV estimation).', type=int, default=1000)
parser.add_argument('-M', '--train_size', help='Size of the dataset of each player.', type=int, default=200, choices=[50, 100, 200])
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
Information theoretic data valuation

"""
from scipy.stats import sem
have_gp = True
if have_gp:
    from vol_utils.gpytorch_ig import compute_IG, fit_model

    exact_SV_trials = []
    s_values_kernelSHAP_trials = []
    s_values_active_trials = []

    SV_trials = defaultdict(list)
    statistics_trials = defaultdict(list)

    for t in range(args.trials):
        all_train_X = torch.cat(feature_datasets)
        all_train_y = torch.cat(labels).reshape(-1 ,1).squeeze()
        joint_model, joint_likelihood = fit_model(all_train_X, all_train_y)

        def IG_utility(S): 
            return _utility(tuple(sorted(S)))

        @lru_cache(maxsize=min(2**args.n, 1024))
        def _utility(S):
            if len(S) == 0:return 0
            
            else:
                curr_train_X = torch.cat([dataset for j, dataset in enumerate(feature_datasets)  if j in S]).reshape(-1, D)
                return compute_IG(curr_train_X, joint_model, joint_likelihood)

        exact_SV = exact(args.n, IG_utility)
        SV_trials['Exact'].append(exact_SV / np.mean(exact_SV))

        # kernelSHAP
        mcs, afs, min_afs = kernelSHAP(args.n, args.num_samples, IG_utility, seed=t)
        s_estimates = _get_SV_estimates(args.n, mcs)
        SV_trials['KernelSHAP'].append(s_estimates)
        statistics_trials['KernelSHAP'].append((mcs, afs, min_afs))


        for alpha in [0, 2, 5, 100]:

            # active: 2-FAE
            mcs, afs, min_afs = active(args.n, args.num_samples, IG_utility, seed=t, bootstrap_n=500, alpha=alpha)
            s_estimates = _get_SV_estimates(args.n, mcs)
            SV_trials['2-FAE-a'+str(alpha)].append(s_estimates)
            statistics_trials['2-FAE-a'+str(alpha)].append((mcs, afs, min_afs))


        # other samplers
        for method in ['Sobol', 'Stratified', 'Owen', 'MC']:
            # print(f'Executing {method} now.')
            mcs, afs, min_afs = generic_sampler(method, args.n, args.num_samples, IG_utility, seed=t)
            s_estimates = _get_SV_estimates(args.n, mcs)
            SV_trials[method].append(s_estimates)
            statistics_trials[method].append((mcs, afs, min_afs))


    print('------Information Gain Shapley value Statistics ------')
    for method, SV_over_trials in SV_trials.items():
        SV_over_trials = np.stack(SV_over_trials)
        mean, std_err = np.mean(SV_over_trials, 0), sem(SV_over_trials, axis=0)
        print(f'IGSV {method}: mean {mean}, std-err {std_err}.')

        res[method] = mean
        res[method+'SV_over_trials'] = SV_over_trials
        res[method +'statistics'] = statistics_trials[method]
    print('-------------------------------------')
    

from os.path import join as oj
from vol_utils.utils import cwd

with cwd(oj(DATA_PATH, 'IG')):
    np.savez(f'res_{args.dataset} {D}D_{args.n}n_{args.trials}t_{args.num_samples}samples.npz',
        res=res, n=args.n, t=args.trials, D=D, num_samples=args.num_samples, train_sizes=train_sizes, test_sizes=test_sizes, dataset=args.dataset, seed=seed)


exit()


# have_spgp = False
# if have_spgp:
#     from vol_utils.gpytorch_ig import compute_IG, fit_model

#     trials = 5

#     s_values_IG_trials = []
#     mc_s_values_IG_trials = []

#     for t in range(trials):

#         inducing_ratio = 0.25
#         inducing_count = int(torch.sum(torch.tensor(train_sizes)) * inducing_ratio * args.n)

#         end, begin = 1, 0
#         # uniform distribution of inducing
#         inducing_points = torch.rand((inducing_count, D)) * (end - begin) + begin 

#         all_train_X = torch.cat(feature_datasets)
#         all_train_y = torch.cat(labels).reshape(-1 ,1).squeeze()
#         joint_model, joint_likelihood = fit_model(all_train_X, all_train_y, inducing_points=inducing_points)


#         s_values_IG = torch.zeros(args.n)
#         monte_carlo_s_values_IG = torch.zeros(args.n)

#         orderings = list(permutations(range(args.n)))
#         # Monte-carlo : shuffling the ordering and taking the first K orderings
#         random.shuffle(orderings)
#         K = 4 # number of permutations to sample

#         for ordering_count, ordering in enumerate(orderings):

#             prefix_IGs = []
#             for position, i in enumerate(ordering):

#                 curr_indices = set(ordering[:position+1])

#                 curr_train_X = torch.cat([dataset for j, dataset in enumerate(feature_datasets)  if j in curr_indices ]).reshape(-1, D)
#                 # curr_train_y = torch.cat([label for j, label in enumerate(labels)  if j in curr_indices ]).reshape(-1, 1)
#                 # curr_train_y = curr_train_y.squeeze()

#                 # model, likelihood = fit_model(curr_train_X, curr_train_y, inducing_points=inducing_points)
#                 curr_IG = compute_IG(curr_train_X, joint_model, joint_likelihood)

#                 if position == 0: # first in the ordering
#                     marginal = curr_IG  - 0
#                 else:
#                     marginal = curr_IG - prefix_IGs[-1] 
#                 s_values_IG[i] += marginal
#                 prefix_IGs.append(curr_IG)

#                 if ordering_count < K:
#                     monte_carlo_s_values_IG[i] += marginal

#         s_values_IG /= factorial(args.n)
#         monte_carlo_s_values_IG /= K

#         s_values_IG_trials.append(s_values_IG)
#         mc_s_values_IG_trials.append(monte_carlo_s_values_IG)


#     s_values_IG_trials = torch.stack(s_values_IG_trials)
#     mc_s_values_IG_trials = torch.stack(mc_s_values_IG_trials)

#     print('------Information Gain SPGP Shapley value Statistics ------')
#     print("SPGP IG-based Shapley values: mean {}, sem {}".format(torch.mean(s_values_IG_trials, 0), sem(s_values_IG_trials, axis=0)))
#     print("SPGP IG-based MC-Shapley values: mean {}, sem {}".format(torch.mean(mc_s_values_IG_trials, 0), sem(mc_s_values_IG_trials, axis=0)))
#     print('-------------------------------------')

#     res['spgp_ig_sv'], res['spgp_ig_mc_sv'] = torch.mean(s_values_IG_trials, 0), torch.mean(mc_s_values_IG_trials, 0)


# suffix = '_rep' if rep else '' + '_superset' if superset else '' + '_train_test_diff_distri' if train_test_diff_distr else '' + '_size' if size else '' + '_disjoint' if disjoint else ''


