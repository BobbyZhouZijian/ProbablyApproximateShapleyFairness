from utils import get_shaps, compute_weight_list
import numpy as np


def execute_shap_runner(runner, is_sequential, num_samples, sample_method, seed, **kwargs):
    return runner.run(is_sequential=is_sequential, num_samples=num_samples,
               sample_method=sample_method, seed=seed, **kwargs)

def execute_shap_runner_until_converge(runner, is_sequential, num_samples, sample_method,
                                       thres=0.05, max_iter=10, **kwargs):
    last_shaps = []
    marginal_contribs = []
    count = 0
    while True:
        count += 1
        runner.run(is_sequential=is_sequential, num_samples=num_samples, sample_method=sample_method, **kwargs)
        if len(last_shaps) == 0:
            marginal_contribs = runner.marginal_contribs
            shaps = np.asarray(get_shaps(runner.marginal_contribs))
            last_shaps = shaps
            continue
        else:
            n = len(marginal_contribs)
            for i in range(n):
                marginal_contribs[i] += runner.marginal_contribs[i]
            shaps = np.asarray(get_shaps(marginal_contribs))
            diff = np.abs(shaps - last_shaps)
            abs_shaps = np.abs(shaps)
            test_statistic = (diff / abs_shaps).mean()
            print(f"test statistic: {test_statistic}")
            if test_statistic <= thres:
                print(f"converged! number of iterations {count}.")
                return marginal_contribs
            last_shaps = shaps
        if count >= max_iter:
            print(f"did not manage to converge. max iter {count}")
            return marginal_contribs


def execute_shap_runner_active_valuation(runner, num_samples, bootstrap, xi, alpha, active, seed):
    mcs, afs, min_afs = runner.run_active_valuation(num_samples=num_samples,
                                                    bootstrap=bootstrap,
                                                    xi=xi,
                                                    alpha=alpha,
                                                    active=active,
                                                    seed=seed
                                                    )
    return mcs, afs, min_afs

def execute_shap_runner_all_points(runner, num_samples, bootstrap, xi, method, seed):
    if method == 'kernel':
        mcs, afs, min_afs = runner.run_kernel_shap(num_samples=num_samples,
                                                    bootstrap=bootstrap,
                                                    seed=seed, 
                                                    xi=xi
                                                    )
    else:
        mcs, afs, min_afs = runner.run_all_points(num_samples=num_samples,
                                                        bootstrap=bootstrap,
                                                        xi=xi,
                                                        method=method,
                                                        seed=seed
                                                        )
    return mcs, afs, min_afs

def execute_shap_runner_active_valuation_beta_shap(runner, num_samples, bootstrap, xi, alpha, active, seed):
    weight_list = compute_weight_list(runner.n_sources, 16, 1)
    mcs, afs, min_afs = runner.run_active_valuation(num_samples=num_samples,
                                                    bootstrap=bootstrap,
                                                    xi=xi,
                                                    alpha=alpha,
                                                    weight_list=weight_list,
                                                    active=active,
                                                    seed=seed
                                                    )
    return mcs, afs, min_afs

def execute_shap_runner_all_points_beta_shap(runner, num_samples, bootstrap, xi, method, seed):
    weight_list = compute_weight_list(runner.n_sources, 16, 1)
    if method == 'kernel':
        mcs, afs, min_afs = runner.run_kernel_shap(num_samples=num_samples,
                                                    bootstrap=bootstrap,
                                                    seed=seed, 
                                                    weight_list=weight_list,
                                                    xi=xi
                                                    )
    else:
        mcs, afs, min_afs = runner.run_all_points(num_samples=num_samples,
                                                        bootstrap=bootstrap,
                                                        xi=xi,
                                                        method=method,
                                                        weight_list=weight_list,
                                                        seed=seed
                                                        )
    return mcs, afs, min_afs

def compute_exact_shap(runner):
    return runner.run(is_sequential=True, sample_method='exact', )