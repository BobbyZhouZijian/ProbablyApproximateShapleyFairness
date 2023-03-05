import numpy as np
import multiprocessing as mp

import os
import sys
sys.path.append('./tools')
from data_loader import load_data
from shap_runner import ShapRunner
from experiments import execute_shap_runner, compute_exact_shap, execute_shap_runner_active_valuation, execute_shap_runner_all_points

import argparse
parser = argparse.ArgumentParser(description='Configure the experiment.')
parser.add_argument('-d', '--dataset', help='Dataset name', type=str, required=True)
parser.add_argument('-n', '--n', help='The number of data points', type=int, default=50)
parser.add_argument('-m', '--num_samples', help='The number of samples (i.e., permutations for SV estimation).', type=int, default=7500)
parser.add_argument('-t', '--trials', help='Number of independent random trials to run.', type=int, default=10)
parser.add_argument('-M', '--methods-list', nargs='+', default=['random', 'stratified', 'owen', 'Sobol', 'kernel', 'active-0', 'active-2', 'active-5', 'active-100'])
parser.add_argument('-l', '--learner', help='The learner for the task.', type=str, default='logistic')
parser.add_argument('-x', '--xi', help='xi value of the task', type=float, default=1e-3)
args = parser.parse_args()

# Global variables
task = 'regression'
dataset_name = args.dataset
num_datapoints = args.n
dargs = {'n_data_to_be_valued': num_datapoints + 2, 'n_val': 100, 'n_test': 1000, 'seed': 2020}
model_name = args.learner
metric = 'negative_mse'
seed = 2022
repeat_num = args.trials
num_samples = args.num_samples
xi = args.xi
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

runner = ShapRunner(X, y, X_val, y_val, model_name=model_name, task=task,
                    metric=metric, seed=seed, n_jobs=1, min_cardinality=0, init_data=[X_init, y_init])
print(f"number of data points: {runner.n_sources}")

load = False
save = True

path = './experiment_data/xi'
if save and not os.path.exists(path):
    os.makedirs(path)

print("Running est mcs...")
seeds = [seed + i for i in range(repeat_num)]
with mp.Pool(processes=10) as p:
    est_mcs_list = p.starmap(execute_shap_runner, [(runner, True, num_samples, 'random', seeds[i])
        for i in range(repeat_num)])

print("Finished running est mcs. Now save...")
if save:
    np.save(f"{path}/xi_est_{dataset_name}_{model_name}_{num_samples}_{xi}.npy", est_mcs_list)

print("all done. exitting now...")
exit()