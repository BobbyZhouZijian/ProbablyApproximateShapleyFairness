import numpy as np
import multiprocessing as mp

import os
import sys
sys.path.append('./tools')
from data_loader import load_data
from shap_runner import ShapRunner
from experiments import execute_shap_runner, execute_shap_runner_active_valuation, execute_shap_runner_all_points

import argparse
parser = argparse.ArgumentParser(description='Configure the experiment.')
parser.add_argument('-d', '--dataset', help='Dataset name', type=str, required=True)
parser.add_argument('-n', '--n', help='The number of data points', type=int, default=50)
parser.add_argument('-m', '--num_samples', help='The number of samples (i.e., permutations for SV estimation).', type=int, default=7500)
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
seed = 1234
repeat_num = args.trials
num_samples = args.num_samples
xi = 1e-3
methods = args.methods_list

(X, y), (X_val, y_val), (X_test, y_test) = load_data(task, dataset_name, **dargs)
num_classes = len(set(y_val))
print(f"number of classes: {num_classes}")

# shift a class 0 and a class 1 to front
need_class = 1 - y[0]
for i in range(1, len(X)):
    if y[i] == need_class:
        X[1], X[i] = X[i], X[1]
        y[1], y[i] = y[i], y[1]
        break
# set the first 2 as init values
X_init = X[:2]
y_init = y[:2]
X = X[2:]
y = y[2:]

runner = ShapRunner(X, y, X_val, y_val, model_name=model_name, 
                    metric=metric, seed=seed, n_jobs=1, min_cardinality=0, init_data=[X_init, y_init])
print(f"number of data points: {runner.n_sources}")

load = False
save = True

path = './experiment_data/nullity'
if save and not os.path.exists(path):
    os.makedirs(path)
seeds = [seed + i for i in range(repeat_num)]

print("Starting running true mcs")
# true_mcs = compute_exact_shap(runner)
with mp.Pool(processes=10) as p:
    true_mcs_list = p.starmap(execute_shap_runner, [(runner, True, num_samples, 'random', seeds[i])
                for i in range(repeat_num)])

print("Finished running true mcs. Now run est mcs")
res_shap = []
with mp.Pool(processes=10) as p:
    for k, method in enumerate(methods):
        k += 2
        num_bootstrap = int(num_samples * 0.2)
        if method.startswith('active'):
            alpha = int(method.split('-')[-1])
            print(f"using alpha: {alpha}")
            results = p.starmap(execute_shap_runner_active_valuation, [(runner, (num_samples - num_bootstrap) * num_datapoints, num_bootstrap, xi, alpha, True, k * seeds[i])
                                                    for i in range(repeat_num)])
        else:
            results = p.starmap(execute_shap_runner_all_points, [(runner, (num_samples - num_bootstrap) * num_datapoints, num_bootstrap, xi, method, k * seeds[i])
                                                    for i in range(repeat_num)])      
        all_mcs = [result[0] for result in results]
        all_afs = [result[1] for result in results]
        all_min_afs = [result[2] for result in results]
        res_shap.append((method, all_mcs, all_afs, all_min_afs))
print("Finished running est mcs. Now save...")

if save:
    np.save(f"{path}/nullity_exact_{dataset_name}_{model_name}_{num_samples}.npy", true_mcs_list)
    for method, all_mcs, all_afs, all_min_afs in res_shap:
        np.save(f"{path}/nullity_mcs_{method}_{dataset_name}_{model_name}_{num_samples}.npy", all_mcs)
        np.save(f"{path}/nullity_afs_{method}_{dataset_name}_{model_name}_{num_samples}.npy", all_afs)

print("all done. exitting now...")
exit()