import sklearn
import shap
from functools import lru_cache

import sys
sys.path.append('./tools')
import logging
logging.basicConfig(stream = sys.stdout,
                    format = "%(levelname)s %(asctime)s - %(message)s", 
                    level = logging.INFO)
logger = logging.getLogger()

import random
from collections import defaultdict

import numpy as np
import copy
import argparse


from itertools import permutations
from math import factorial

from vol_utils.data_utils import get_datasets
# from vol_utils.volume import replicate

from exp_utils import exact, generic_sampler, kernelSHAP, active, _get_SV_estimates


from multiprocessing.pool import ThreadPool as Pool
import multiprocessing
# from multiprocessing import Pool

from scipy.stats import sem

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA


# Reproducebility
seed = 1234
random.seed(seed)
np.random.seed(seed)
DATA_PATH = 'experiment_data/'


# ---------- DATA PREPARATION ----------


parser = argparse.ArgumentParser(description='Configure the experiment.')

parser.add_argument('-d', '--dataset', help='Dataset name', type=str, required=True, choices=['iris', 'wine', 'adult', 'covertype'])
parser.add_argument('-m', '--num_samples', help='The number of samples (i.e., permutations for SV estimation).', type=int, default=1000)
parser.add_argument('-t', '--trials', help='Number of independent random trials to run.', type=int, default=5)
parser.add_argument('-n', '--num_components', help='Number of components for PCA.', type=int, default=-1)

args = parser.parse_args()
logger.info(args)

num_samples = args.num_samples
n_trials = args.trials

dataset = args.dataset
logger.info(f"load dataset: {dataset}.")

# initialize PCA
use_pca = args.num_components > 0
if use_pca:
    pca = PCA(n_components=args.num_components, random_state=seed)

if dataset == 'iris':
    # train a kNN classifier
    X_train,X_test,Y_train,Y_test = train_test_split(*shap.datasets.iris(), test_size=0.2, random_state=0)
    knn = KNeighborsClassifier()
    knn.fit(X_train, Y_train)
    clf = knn

elif dataset == 'adult':    
    X, y = shap.datasets.adult()
    if use_pca:
        X = pca.fit_transform(X)
    X = X[:2000]
    y = y[:2000]
    # train a SVM classifier
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=7)
    svm = SVC(kernel='rbf', probability=True)
    svm.fit(X_train, Y_train)
    clf = svm

elif dataset == 'covertype':
    X, y = sklearn.datasets.fetch_covtype(data_home='data', return_X_y=True)
    if use_pca:
        X = pca.fit_transform(X)
    y = ((y == 1) + 0.0).astype(int)
    X = X[:2000]
    y = y[:2000]
    # train a MLP
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    mlp = MLPClassifier(random_state=1, max_iter=300).fit(X_train, Y_train)
    clf = mlp

elif dataset == 'wine':
    X, y = sklearn.datasets.load_wine(return_X_y=True)
    if use_pca:
        X = pca.fit_transform(X)
    X = X[:2000]
    y = y[:2000]
    # train a Random Forest classifier
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=7)
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X_train, Y_train.ravel())
    clf = rf

N = n_features = X_train.shape[1]
logger.info(f"processing {N} features.")
    
if isinstance(X_test, np.ndarray):
    data_X = X_test
else:
    data_X = X_test.values
    
def feature_utility(S):
    return _utility(tuple(sorted(S)))[1]

@lru_cache(maxsize=2**n_features)
def _utility(S, delta = 0.1):
    n_cores = multiprocessing.cpu_count() // 2
    with Pool(n_cores) as pool:
        input_arguments = [(data_X, x, S, delta)  for x in data_X]
        output = pool.starmap(_get_predict_proba_S, input_arguments)
    res = np.mean(output)
    return S, res

def _get_predict_proba_S(data_X, x, S, delta):
    mask = np.ones(len(data_X), dtype=bool)
    for col_i in S:
        mask &= (x[col_i]* (1 - delta) <= data_X[:,col_i]) & (data_X[:,col_i] <= x[col_i]*(1+delta))
    
    if sum(mask) == 0:
        return 0
    else:
        # only take the probablity of being 1
        return np.mean(clf.predict_proba(data_X[mask])[:, 1]).squeeze()


exact_SV_trials = []
s_values_kernelSHAP_trials = []
s_values_active_trials = []

SV_trials = defaultdict(list)
statistics_trials = defaultdict(list)

res = {}

for n_trial in range(n_trials):
    logger.info(f'trial {n_trial+1} of {n_trials}')
    if N >= 10:
        mcs, afs, min_afs = generic_sampler('MC', N, num_samples, feature_utility, seed=n_trial)
        sv_estimates = _get_SV_estimates(N, mcs, normalize=False)    
        sv_estimates = sv_estimates / sv_estimates[0]
        exact_SV = sv_estimates
    else:
        exact_SV = exact(N, feature_utility)
        exact_SV =  exact_SV/ (exact_SV[0])
    SV_trials['Exact'].append(exact_SV)
    logger.info("Exact Shapley calculated.")

    mcs, afs, min_afs = kernelSHAP(N, num_samples, feature_utility, seed=n_trial)
    sv_estimates = _get_SV_estimates(N, mcs, normalize=False)
    sv_estimates = sv_estimates / (sv_estimates[0])
    SV_trials['KernelSHAP'].append(sv_estimates)
    statistics_trials['KernelSHAP'].append((mcs, afs, min_afs))


    explainer = shap.KernelExplainer(clf.predict_proba, X_test, link="logit")
    shap_values = explainer.shap_values(X_test, nsamples=num_samples)
    sv_estimates = shap_values[0].mean(axis=0)
    sv_estimates = sv_estimates / (sv_estimates[0])
    SV_trials['KernelSHAP-github'].append(sv_estimates)
    statistics_trials['KernelSHAP-github'].append((mcs, afs, min_afs))


    mcs, afs, min_afs = active(N, num_samples, feature_utility, alpha=0, seed=n_trial)
    sv_estimates = _get_SV_estimates(N, mcs, normalize=False)
    sv_estimates = sv_estimates / (sv_estimates[0])
    SV_trials['2-FAE-0'].append(sv_estimates)
    statistics_trials['2-FAE-0'].append((mcs, afs, min_afs))

    mcs, afs, min_afs = active(N, num_samples, feature_utility, alpha=2.0, seed=n_trial)
    sv_estimates = _get_SV_estimates(N, mcs, normalize=False)
    sv_estimates = sv_estimates / (sv_estimates[0])
    SV_trials['2-FAE-2'].append(sv_estimates)
    statistics_trials['2-FAE-2'].append((mcs, afs, min_afs))

    mcs, afs, min_afs = active(N, num_samples, feature_utility, alpha=5.0, seed=n_trial)
    sv_estimates = _get_SV_estimates(N, mcs, normalize=False)
    sv_estimates = sv_estimates / (sv_estimates[0])
    SV_trials['2-FAE-5'].append(sv_estimates)
    statistics_trials['2-FAE-5'].append((mcs, afs, min_afs))

    mcs, afs, min_afs = active(N, num_samples, feature_utility, alpha=100, seed=n_trial)
    sv_estimates = _get_SV_estimates(N, mcs, normalize=False)
    sv_estimates = sv_estimates / (sv_estimates[0])
    SV_trials['2-FAE-100'].append(sv_estimates)
    statistics_trials['2-FAE-100'].append((mcs, afs, min_afs))

    for method in ['Sobol', 'Stratified', 'Owen', 'MC']:
        logger.info(f'Executing {method} now.')
        mcs, afs, min_afs = generic_sampler(method, N, num_samples, feature_utility, seed=n_trial)
        sv_estimates = _get_SV_estimates(N, mcs, normalize=False)    
        sv_estimates = sv_estimates / (sv_estimates[0])

        SV_trials[method].append(sv_estimates)
        statistics_trials[method].append((mcs, afs, min_afs))

# logger.info('------Feature importance Shapley value Statistics ------')
for method, SV_over_trials in SV_trials.items():
    SV_over_trials = np.stack(SV_over_trials)
    mean, std_err = np.mean(SV_over_trials, 0), sem(SV_over_trials, axis=0)
    # logger.info(f'FeatureSV {method}: mean {mean}, std-err {std_err}.')

    res[method] = mean
    res[method+'SV_over_trials'] = SV_over_trials
    res[method +'statistics'] = statistics_trials[method]
# logger.info('-------')


# suffix = '_rep' if rep else '' + '_superset' if superset else '' + '_train_test_diff_distri' if train_test_diff_distr else '' + '_size' if size else '' + '_disjoint' if disjoint else ''

logger.info('Finished running. Now saving results...')
from os.path import join as oj
from vol_utils.utils import cwd
with cwd(oj(DATA_PATH, 'feature')):
    np.savez(f'res_{args.dataset}_{N}n_{args.trials}t_{args.num_samples}samples.npz',
        res=res, n=N, t=args.trials, num_samples=args.num_samples, dataset=args.dataset, seed=seed)
logger.info('All done.')
exit()

