import os
import numpy as np
import multiprocessing as mp
from scipy.stats import sem

import sys
sys.path.append('./tools')
from data_loader import load_data
from shap_runner import ShapRunner
from experiments import execute_shap_runner_active_valuation, execute_shap_runner_all_points
from experiments import execute_shap_runner_active_valuation_beta_shap, execute_shap_runner_all_points_beta_shap

import argparse
parser = argparse.ArgumentParser(description='Configure the experiment.')
parser.add_argument('-d', '--dataset', help='Dataset name', type=str, required=True)
parser.add_argument('-n', '--n', help='The number of data points', type=int, default=50)
parser.add_argument('-m', '--num_samples', help='The number of samples (i.e., permutations for SV estimation).', type=int, default=7500)
parser.add_argument('-b', '--num_bootstrap', help='The number of bootstrap samples.', type=int, default=25)
parser.add_argument('-t', '--trials', help='Number of independent random trials to run.', type=int, default=10)
parser.add_argument('-M', '--methods-list', nargs='+', default=['random', 'stratified', 'owen', 'Sobol', 'kernel', 'active-0', 'active-2', 'active-5', 'active-100'])
parser.add_argument('-l', '--learner', help='The learner for the task.', type=str, default='logistic')
args = parser.parse_args()

# Global variables
task = 'classification'
dataset_name = args.dataset
num_datapoints = args.n
dargs = {'n_data_to_be_valued': num_datapoints + 2, 'n_val': 100, 'n_test': 1000, 'seed': 2020}
model_name = args.learner
metric = 'accuracy'
seed = 2022
repeat_num = args.trials
num_samples = args.num_samples
num_bootstrap = args.num_bootstrap
xi = 1e-3
methods = args.methods_list

num_processes = 10


(X, y), (X_val, y_val), (X_test, y_test), flipped_index = load_data(task, dataset_name, is_noisy=True, **dargs)
is_flipped = np.zeros(len(X), dtype=int)
is_flipped[flipped_index] = 1
num_classes = len(set(y_val))
print(f"number of classes: {num_classes}")

# shift a class 0 and a class 1 to front; avoid flipped indices
need_class = None
for start in range(len(X)):
    if is_flipped[start] == 0:
        # switch with position 0
        X[start], X[0] = X[0], X[start]
        y[start], y[0] = y[0], y[start]
        is_flipped[0], is_flipped[start] = is_flipped[start], is_flipped[0]
        need_class = 1 - y[0]
        break
    if start == len(X) - 1:
        raise ValueError("all data points are flipped")

for i in range(1, len(X)):
    if y[i] == need_class and is_flipped[i] == 0:
        X[1], X[i] = X[i], X[1]
        y[1], y[i] = y[i], y[1]
        is_flipped[i], is_flipped[1] = is_flipped[1], is_flipped[i]
        break
    if i == len(X) - 1:
        raise ValueError("no data points in other class is not flipped.")

# set the first 2 as init values
X_init = X[:2]
y_init = y[:2]
X = X[2:]
y = y[2:]

# recover flipped indices
flipped_index = np.where(is_flipped == 1)[0]
flipped_index -= 2

runner = ShapRunner(X, y, X_val, y_val, model_name=model_name, 
                    metric=metric, seed=seed, n_jobs=1, min_cardinality=0, init_data=[X_init, y_init])
print(f"number of data points: {runner.n_sources}")


load = False
save = True

path = './experiment_data/noisy_label_detection'
if save and not os.path.exists(path):
    os.makedirs(path)

"""
Test out all sampling methods for data shapley
"""

res_data_shap = []
seeds = [seed + i for i in range(repeat_num)]
for method in methods:
    print(f"{method} started running")
    name = f"{path}/noisy_label_data_shap_{method}_{dataset_name}_{model_name}_{num_datapoints}_{num_samples}_{num_bootstrap}.npy"
    if load and os.path.exists(name):
        print("results exist. Skipping this method...")
        all_mcs = np.load(f"{path}/noisy_label_all_mcs_data_shap_{method}_{dataset_name}_{model_name}_{num_datapoints}_{num_samples}_{num_bootstrap}.npy", allow_pickle=True)
        all_afs = np.load(f"{path}/noisy_label_all_afs_data_shap_{method}_{dataset_name}_{model_name}_{num_datapoints}_{num_samples}_{num_bootstrap}.npy", allow_pickle=True)
        all_min_afs = np.load(f"{path}/noisy_label_all_min_afs_data_shap_{method}_{dataset_name}_{model_name}_{num_datapoints}_{num_samples}_{num_bootstrap}.npy", allow_pickle=True)
        res_data_shap.append((method, all_mcs, all_afs, all_min_afs)) 
        continue
    if method.startswith('active'):
        alpha = int(method.split('-')[-1])
        print(f"using alpha: {alpha}")
        with mp.Pool(processes=num_processes) as p:
            results = p.starmap(execute_shap_runner_active_valuation, [(runner, num_samples, num_bootstrap, xi, alpha, True, seeds[i])
                                                for i in range(repeat_num)])
    else:
        with mp.Pool(processes=num_processes) as p:
            results = p.starmap(execute_shap_runner_all_points, [(runner, num_samples, num_bootstrap, xi, method, seeds[i])
                                                    for i in range(repeat_num)])           
    all_mcs = [result[0] for result in results]
    all_afs = [result[1] for result in results]
    all_min_afs = [result[2] for result in results]
    res_data_shap.append((method, all_mcs, all_afs, all_min_afs))
    print(f"{method} finished running")
print("Data Shapley finished running")

for i, item in enumerate(res_data_shap):
    method, all_mcs, all_afs, all_min_afs = item
    all_min_afs = np.asarray(all_min_afs)
    all_min_afs_mean = np.mean(all_min_afs, axis=0)
    all_min_afs_sem = sem(all_min_afs,axis=0)
    res_data_shap[i] = (method, all_mcs, all_afs, all_min_afs, all_min_afs_mean, all_min_afs_sem)

if save:
    for item in res_data_shap:
        method, all_mcs, all_afs, all_min_afs, _, _ = item
        np.save(f"{path}/noisy_label_all_mcs_data_shap_{method}_{dataset_name}_{model_name}_{num_datapoints}_{num_samples}_{num_bootstrap}.npy", all_mcs)
        np.save(f"{path}/noisy_label_all_afs_data_shap_{method}_{dataset_name}_{model_name}_{num_datapoints}_{num_samples}_{num_bootstrap}.npy", all_afs)
        np.save(f"{path}/noisy_label_all_min_afs_data_shap_{method}_{dataset_name}_{model_name}_{num_datapoints}_{num_samples}_{num_bootstrap}.npy", all_min_afs)
        np.save(f"{path}/noisy_label_flipped_index_data_shap_{method}_{dataset_name}_{model_name}_{num_datapoints}_{num_samples}_{num_bootstrap}.npy", np.asarray(flipped_index))
    print("Data Shapley finished saving")

"""
Test out all sampling methods for beta shapley
"""
res_beta_shap = []
seeds = [seed + i for i in range(repeat_num)]
for method in methods:
    print(f"{method} started running")
    name = f"{path}/noisy_label_all_mcs_beta_shap_{method}_{dataset_name}_{model_name}_{num_datapoints}_{num_samples}_{num_bootstrap}.npy"
    if load and os.path.exists(name):
        print("results exist. Skipping this method...")
        all_mcs = np.load(f"{path}/noisy_label_all_mcs_beta_shap_{method}_{dataset_name}_{model_name}_{num_datapoints}_{num_samples}_{num_bootstrap}.npy", allow_pickle=True)
        all_afs = np.load(f"{path}/noisy_label_all_afs_beta_shap_{method}_{dataset_name}_{model_name}_{num_datapoints}_{num_samples}_{num_bootstrap}.npy", allow_pickle=True)
        all_min_afs = np.load(f"{path}/noisy_label_all_min_afs_beta_shap_{method}_{dataset_name}_{model_name}_{num_datapoints}_{num_samples}_{num_bootstrap}.npy", allow_pickle=True)
        res_beta_shap.append((method, all_mcs, all_afs, all_min_afs)) 
        continue
    if method.startswith('active'):
        alpha = int(method.split('-')[-1])
        print(f"using alpha: {alpha}")
        with mp.Pool(processes=num_processes) as p:
            results = p.starmap(execute_shap_runner_active_valuation_beta_shap, [(runner, num_samples, num_bootstrap, xi, alpha, True, seeds[i])
                                                for i in range(repeat_num)])
    else:
        with mp.Pool(processes=num_processes) as p:
            results = p.starmap(execute_shap_runner_all_points_beta_shap, [(runner, num_samples, num_bootstrap, xi, method, seeds[i])
                                                    for i in range(repeat_num)])           
    all_mcs = [result[0] for result in results]
    all_afs = [result[1] for result in results]
    all_min_afs = [result[2] for result in results]
    res_beta_shap.append((method, all_mcs, all_afs, all_min_afs))
    print(f"{method} finished running")
print("Beta Shapley finished running")

for i, item in enumerate(res_beta_shap):
    method, all_mcs, all_afs, all_min_afs = item
    all_min_afs = np.asarray(all_min_afs)
    all_min_afs_mean = np.mean(all_min_afs, axis=0)
    all_min_afs_std = np.std(all_min_afs, axis=0)
    all_min_afs_sem = sem(all_min_afs, axis=0)
    res_beta_shap[i] = (method, all_mcs, all_afs, all_min_afs, all_min_afs_mean, all_min_afs_sem)

if save:
    for item in res_beta_shap:
        method, all_mcs, all_afs, all_min_afs, _, _ = item
        np.save(f"{path}/noisy_label_all_mcs_beta_shap_{method}_{dataset_name}_{model_name}_{num_datapoints}_{num_samples}_{num_bootstrap}.npy", all_mcs)
        np.save(f"{path}/noisy_label_all_afs_beta_shap_{method}_{dataset_name}_{model_name}_{num_datapoints}_{num_samples}_{num_bootstrap}.npy", all_afs)
        np.save(f"{path}/noisy_label_all_min_afs_beta_shap_{method}_{dataset_name}_{model_name}_{num_datapoints}_{num_samples}_{num_bootstrap}.npy", all_min_afs)
        np.save(f"{path}/noisy_label_flipped_index_beta_shap_{method}_{dataset_name}_{model_name}_{num_datapoints}_{num_samples}_{num_bootstrap}.npy", np.asarray(flipped_index))
    print("Beta Shapley finished saving")

print("all done. exitting now...")
exit()
