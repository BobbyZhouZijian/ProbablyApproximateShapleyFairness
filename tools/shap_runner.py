import random
from functools import lru_cache
from collections import defaultdict
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from model_builder import return_model
from perm_sampler import PermutationSampler
from utils import get_afs, get_mc_sum_by_card, get_weight_star


class ShapRunner:
    def __init__(self,
                 X,
                 y,
                 X_val,
                 y_val,
                 task='classification',
                 model_name='logistic',
                 metric='accuracy',
                 min_cardinality=1,
                 seed=None,
                 sources=None,
                 init_data=None,
                 **kwargs
                 ):
        self.X, self.y, self.X_val, self.y_val = X, y, X_val, y_val
        self.task = task
        self.model_name = model_name
        self.metric = metric
        self.min_cardinality = min_cardinality
        self.seed = seed
        if seed:
            np.random.seed(seed)
        self.sources = sources

        self._initialize(init_data=init_data)
        self.model = return_model(self.model_name, **kwargs)
        self.random_score = self.init_score()

    def _initialize(self, init_data=None):
        if self.task == 'regression':
            assert (self.min_cardinality is not None), 'Check min_cardinality'
            self.is_regression = True
            if self.metric not in ['r2', 'negative_mse']:
                assert False, 'Invalid metric for regression!'
            self.init_X = init_data[0]
            self.init_y = init_data[1]
        elif self.task == 'classification':
            self.is_regression = False
            self.num_classes = len(set(self.y_val))
            if self.num_classes > 2:
                assert self.metric != 'f1', 'Invalid metric for multiclass!'
                assert self.metric != 'auc', 'Invalid metric for multiclass!'
            if self.metric not in ['accuracy', 'f1', 'auc', 'likelihood']:
                assert False, 'Invalid metric for classification!'

            if not init_data:
                # select init samples so that each class has at least 1 sample
                visited_cls = {}
                for i in range(len(self.y)):
                    if self.y[i] not in visited_cls:
                        visited_cls[self.y[i]] = i
                        if len(set(visited_cls.keys())) == self.num_classes:
                            break
                if len(set(visited_cls)) < self.num_classes:
                    raise ValueError("Missing samples for one or more classes in training dataset")
                self.init_X = np.zeros((0,) + tuple(self.X.shape[1:]))
                self.init_y = np.zeros(0, int)

                to_delete = []
                for cls in visited_cls:
                    idx = visited_cls[cls]
                    self.init_X = np.concatenate([self.init_X, self.X[np.array([idx])]])
                    self.init_y = np.concatenate([self.init_y, self.y[np.array([idx])]])
                    to_delete.append(idx)

                self.X = np.delete(self.X, to_delete, axis=0)
                self.y = np.delete(self.y, to_delete, axis=0)
                print(f"removed points with idx {to_delete} from training data.")
            else:
                self.init_X = init_data[0]
                self.init_y = init_data[1]
                print("initialize model using given init data.")
        else:
            raise NotImplementedError("Check problem")

        if self.seed:
            print(f"np seeded using seed: {self.seed}")
            np.random.seed(self.seed)

        self.n_points = len(self.X)
        if self.sources:
            print("Source is initialized. A unit of sample is given")
            for i in self.sources.keys():
                print(f'class {i}: {len(self.sources[i])} samples')
        else:
            print("No source provided. A unit of sample is 1 data point")
            self.sources = {i: np.array([i]) for i in range(self.n_points)}
        self.n_sources = len(self.sources)

    def init_score(self):
        if self.task == 'regression':
            if self.metric == 'r2':
                return 0
            elif self.metric == 'negative_mse':
                return -np.mean((self.y_val - np.mean(self.y_val)) ** 2)
            else:
                raise NotImplementedError
        elif self.task == 'classification':
            if self.metric == 'accuracy':
                hist = np.bincount(self.y_val) / len(self.y_val)
                return np.max(hist)
            elif self.metric == 'f1':
                rnd_f1s = []
                for _ in range(1000):
                    rnd_y = np.random.permutation(self.y_val)
                    rnd_f1s.append(f1_score(self.y_val, rnd_y))
                return np.mean(rnd_f1s)
            elif self.metric == 'auc':
                return 0.5
            elif self.metric == 'likelihood':
                hist = np.bincount(self.y_val) / len(self.y_val)
                return np.sum(hist * np.log(hist + 1e-9))
            else:
                raise NotImplementedError

    def value(self, X=None, y=None):
        if X is None:
            X = self.X_val
        if y is None:
            y = self.y_val

        if self.task == 'regression':
            if self.metric == 'r2':
                return self.model.score(X, y)
            elif self.metric == 'negative_mse':
                y_pred = self.model.predict(X)
                return -np.mean((y - y_pred) ** 2)
            elif self.metric == 'prediction':
                return self.model.predict(X)
            else:
                raise NotImplementedError
        elif self.task == 'classification':
            if self.metric == 'accuracy':
                return self.model.score(X, y)
            elif self.metric == 'f1':
                return f1_score(y, self.model.predict(X))
            elif self.metric == 'auc':
                probs = self.model.predict_proba(X)
                return roc_auc_score(y, probs[:, -1])
            elif self.metric == 'likelihood':
                probs = self.model.predict_proba(X)
                true_probs = probs[np.arange(len(y)), y]
                return np.mean(np.log(true_probs))
            elif self.metric == 'prediction':
                probs = self.model.predict_proba(X)
                true_probs = probs[np.arange(len(y)), 1]
                return true_probs
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def run(self, is_sequential=True, num_samples=100, sample_method='random', seed=None, **kwargs):
        sampler = PermutationSampler(self.n_sources, seed)
        if not is_sequential:
            sample_weight_pairs = sampler.sample_per_point(num_samples, method=sample_method, **kwargs)
        else:
            sample_weight_pairs = sampler.sample_sequential(num_samples, method=sample_method, **kwargs)
        return self._calc_marginal_contributions(sample_weight_pair=sample_weight_pairs)

    def run_active_valuation(self,
                             num_samples=100,
                             bootstrap=100,
                             xi=1e-3,
                             alpha=1,
                             weight_list=None,
                             active=True,
                             seed=None
                             ):
        mcs = self._run_bootstrap(bootstrap, seed, weight_list=weight_list)
        sampler = PermutationSampler(self.n_sources, seed)
        mc_by_card = get_weight_star(mcs)
        sampler.estimate_parameters(mc_by_card, alpha=alpha)
        min_afs = []
        afs = [0 for _ in range(self.n_sources)]
        for sample in range(num_samples):
            if (sample + 1) % 1000 == 0:
                print(f"processed {sample+1} samples.")
            afs = get_afs(mcs, xi)
            min_idx = np.argmin(afs)
            if active:
                target_idx = min_idx
            else:
                target_idx = np.random.choice([i for i in range(self.n_sources)], 1)[0]
            min_afs.append(afs[min_idx])
            sampled_weight_pairs = sampler.sample_one_point(target_idx, 1)
            cur_mcs = self._calc_marginal_contributions(sampled_weight_pairs, weight_list=weight_list)
            for i in cur_mcs.keys():
                if i in mcs.keys():
                    mcs[i] += cur_mcs[i]
        return mcs, afs, min_afs

    def run_all_points(self,
                             num_samples=100,
                             bootstrap=100,
                             xi=1e-3,
                             method='random',
                             weight_list=None,
                             seed=None
                             ):
        mcs = self._run_bootstrap(bootstrap, seed, weight_list=weight_list)
        sampler = PermutationSampler(self.n_sources, seed)
        if method == 'dir':
            mc_by_card = get_mc_sum_by_card(mcs)
            sampler.estimate_parameters(mc_by_card, method='dir')
        min_afs = []
        afs = [0 for _ in range(self.n_sources)]
        all_samples, all_weights = sampler.sample_per_point(num_samples, method)
        flattened = []
        for key in all_samples.keys():
            for j in range(len(all_samples[key])):
                flattened.append(({key:[all_samples[key][j]]}, {key: [all_weights[key][j]]}))
        # shuffle flattened to get meaningful k values
        random.seed(seed if seed else 2022)
        random.shuffle(flattened)
        # if len(flattened) != num_samples:
        #     raise ValueError(f"length of unpacked samples is not the same as num_samples. \
        #         found {len(flattened)}; need {num_samples}")
        for sample in range(len(flattened)):
            if (sample + 1) % 1000 == 0:
                print(f"processed {sample+1} samples.")
            afs = get_afs(mcs, xi)
            min_idx = np.argmin(afs)
            min_afs.append(afs[min_idx])
            sampled_weight_pairs = flattened[sample]
            cur_mcs = self._calc_marginal_contributions(sampled_weight_pairs, weight_list=weight_list)
            for i in cur_mcs.keys():
                if i in mcs.keys():
                    mcs[i] += cur_mcs[i]
        return mcs, afs, min_afs

    def _run_bootstrap(self, num_samples, seed, weight_list=None):
        sampler = PermutationSampler(self.n_sources, seed)
        sample_weight_pairs = sampler.sample_sequential(num_samples, method='random')
        return self._calc_marginal_contributions(sample_weight_pair=sample_weight_pairs, weight_list=weight_list)

    def _calc_marginal_contributions(self, sample_weight_pair, weight_list=None):
        samples, weights = sample_weight_pair
        self.marginal_contribs = {}

        if isinstance(samples, dict):
            # marginal contribution per point
            for idx in range(self.n_sources):
                if idx not in samples.keys():
                    continue
                self.marginal_contribs[idx] = []

                samples_idx = samples[idx]
                weights_idx = weights[idx]

                for i in range(len(samples_idx)):
                    sample = samples_idx[i]
                    weight = weights_idx[i]

                    X_batch = np.zeros((0,) + tuple(self.X.shape[1:]))
                    if not self.is_regression:
                        y_batch = np.zeros(0, int)
                    else:
                        y_batch = np.zeros((0,) + tuple(self.y.shape[1:]))

                    X_batch = np.concatenate([X_batch, self.init_X])
                    y_batch = np.concatenate([y_batch, self.init_y])

                    n = 0
                    for cardinality, ii in enumerate(sample):
                        if ii == idx:
                            n = cardinality
                            break
                        X_batch = np.concatenate([X_batch, self.X[self.sources[ii]]])
                        y_batch = np.concatenate([y_batch, self.y[self.sources[ii]]])

                    if n < self.min_cardinality:
                        continue

                    try:
                        self.model.fit(X_batch, y_batch)
                        old_score = self.value()
                    except Exception as e:
                        print(f"Error {e}; old score calculated using init score")
                        old_score = self.random_score

                    X_batch = np.concatenate([X_batch, self.X[self.sources[idx]]])
                    y_batch = np.concatenate([y_batch, self.y[self.sources[idx]]])
                    try:
                        self.model.fit(X_batch, y_batch)
                        new_score = self.value()
                    except Exception as e:
                        print(f"Error {e}; new score calculated using init score")
                        new_score = self.random_score

                    marginal_contrib = (new_score - old_score) * weight
                    if weight_list is not None:
                        marginal_contrib *= weight_list[n]
                    self.marginal_contribs[idx].append((marginal_contrib, n))
        elif isinstance(samples, list):
            # sequential marginal contribution
            for i, idxs in enumerate(samples):
                weight = weights[i]
                X_batch = np.zeros((0,) + tuple(self.X.shape[1:]))
                if not self.is_regression:
                    y_batch = np.zeros(0, int)
                else:
                    y_batch = np.zeros((0,) + tuple(self.y.shape[1:]))

                X_batch = np.concatenate([X_batch, self.init_X])
                y_batch = np.concatenate([y_batch, self.init_y])

                try:
                    self.model.fit(X_batch, y_batch)
                    old_score = self.value()
                except Exception as e:
                    print(f"Error {e}; old score calculated using random score")
                    old_score = self.random_score

                for n, idx in enumerate(idxs):
                    if idx not in self.marginal_contribs:
                        self.marginal_contribs[idx] = []

                    X_batch = np.concatenate([X_batch, self.X[self.sources[idx]]])
                    y_batch = np.concatenate([y_batch, self.y[self.sources[idx]]])

                    try:
                        self.model.fit(X_batch, y_batch)
                        new_score = self.value()
                    except Exception as e:
                        print(f"Error {e}; new score calculated using random score")
                        new_score = self.random_score

                    marginal_contrib = (new_score - old_score) * weight
                    old_score = new_score
                    if weight_list is not None:
                        marginal_contrib *= weight_list[n]
                    self.marginal_contribs[idx].append((marginal_contrib, n))
        return self.marginal_contribs


    def run_kernel_shap(self,
                        num_samples=100,
                        bootstrap=100,
                        xi=1e-3,
                        weight_list=None,
                        seed=None
                        ):

        if seed: random.seed(seed)

        n = self.n_sources
        num_samples //= n
        # num_samples = int(num_samples / (n/2)) # each coalition has n/2 data points on average

        expected_Z = np.asarray([0.5 for _ in range(n)])
        A_inv = self._get_A_inv()


        # setting up weights for sampling according to the kernel
        weights = np.arange(1, n)
        weights = 1 / (weights * (n - weights))
        weights = weights / np.sum(weights)

        S = np.zeros((num_samples, n), dtype=bool)
        num_included = np.random.choice(n - 1, size=num_samples, p=weights) + 1
        # total_samples_used = sum(num_included)
        # left = total_samples_to_use - total_samples_used
        # sign = 1 if left > 0 else -1
        # while left != 0:
        #     for i in range(len(num_included)):
        #         if 0 < num_included[i] + sign <= n:
        #             num_included[i] += sign
        #             left -= sign
        #         if left == 0:
        #             break
        # print(f"sum of evalutions updated: {sum(num_included)}")

        # updating the indices of players in the sampled permutations S of shape (num_samples, num_players)
        for row, num in zip(S, num_included):
            inds = np.random.choice(n, size=num, replace=False)
            row[inds] = 1


        GRAND, NULL = range(n), []
        b_bars, SV_estimate_list = [], []
        b_bar = np.zeros(n)
        mcs = self._run_bootstrap(bootstrap, seed, weight_list=weight_list)
        min_afs = []

        # estimating SV or mcs according to the sampled permutations S
        for count, sample in enumerate(S): # sample is a vector of True/False of length = n
            if (count + 1) % 100 == 0:
                print(f"processed {count+1} samples for kernelSHAP.")
            coalition = [i for i, bit in enumerate(sample) if bit]
            coalition_bit_vector = np.asarray(sample, dtype=int)

            U = self._kernel_shap_utility(coalition)

            # sample-one by one
            single_b_bar = U * coalition_bit_vector - expected_Z * self._kernel_shap_utility(NULL)
            b_bar += single_b_bar
            b_bars.append(single_b_bar)

            numer = np.ones(n)@ A_inv @ single_b_bar - self._kernel_shap_utility(GRAND) + self._kernel_shap_utility(NULL)
            denom = (np.ones(n) @ A_inv @ np.ones(n))

            # multiply by A_inv with (single_b_bar - 1 * (numer/denom))
            SV_estimate = A_inv @ (single_b_bar - np.ones(n)* numer / denom)
            SV_estimate_list.append(SV_estimate)

            # coalition_cardinality = len(coalition)
            # for i in coalition: # coalition is a list of indices of players in this coalition
            #     mc = SV_estimate[i]
            #     cardinality = coalition_cardinality - 1
            #     if weight_list is not None:
            #         mc = mc * weight_list[cardinality]
            #     mcs[i].append((mc, cardinality))
            for i in range(self.n_sources):
                mc = SV_estimate[i]
                cardinality = -1 # not used
                mcs[i].append((mc, cardinality))
                afs = get_afs(mcs, xi)
                min_idx = np.argmin(afs)
                min_afs.append(afs[min_idx])
        # SV_estimates = np.asarray(SV_estimate_list).mean(axis=0) # a vector of dimension self.n_sources

        return mcs, afs, min_afs

    @lru_cache()
    def _get_A_inv(self):
        n = self.n_sources
        # filling in the non-diagonal entries which are all the same
        constA = 1. / (n *(n-1))
        numer = sum( [(k-1)/(n-k) for k in range(2, n)] )
        denom = sum( [1./(k*(n-k)) for k in range(1, n)] )

        A = np.ones( (n,n)) * constA * numer / denom
        # filling in the diagonal entries first
        np.fill_diagonal(A, 0.5)
        return np.linalg.inv(A)


    def _kernel_shap_utility(self, coalition):
        # wraps coalition to make it hashable
        return self._utility(tuple(coalition))

    @lru_cache()
    def _utility(self, hashable_coalition):
        coalition = np.asarray(hashable_coalition)
        X_batch = np.zeros((0,) + tuple(self.X.shape[1:]))
        if not self.is_regression:
            y_batch = np.zeros(0, int)
        else:
            y_batch = np.zeros((0,) + tuple(self.y.shape[1:]))
        X_batch = np.concatenate([X_batch, self.init_X])
        y_batch = np.concatenate([y_batch, self.init_y])
        for idx in coalition:
            X_batch = np.concatenate([X_batch, self.X[self.sources[idx]]])
            y_batch = np.concatenate([y_batch, self.y[self.sources[idx]]])
        self.model.fit(X_batch, y_batch)
        return self.value()
