import math
from re import S
import numpy as np
from scipy.stats import beta, dirichlet
from scipy.optimize import root_scalar
from scipy.stats.qmc import Sobol # needs scipy version 1.7 or later
from other_estimators import get_U_hat, PolarToCartesian, cdf_F


class PermutationSampler:
    def __init__(self, n, seed=2022):
        self.n = n
        if seed:
            np.random.seed(seed)
        self.params = None

    def estimate_parameters(self, data, alpha=2.0):
        data = np.asarray(data)
        data = data * self.n / sum(data)
        prior = np.asarray([alpha for _ in range(self.n)])
        posterior = prior + data
        posterior /= sum(posterior)
        self.params = posterior

    def sample_one_point(self, idx, num_samples, **kwargs):
        samples = {}
        weights = {}
        samples[idx] = []
        weights[idx] = []
        for _ in range(num_samples):
            di = np.random.multinomial(1, self.params, size=1)[0]
            m = np.argmax(di)
            other_players = [k for k in range(self.n) if k != idx]
            perm = np.random.permutation(other_players)
            perm = np.insert(perm, m, idx)
            samples[idx].append(perm)
            weights[idx].append(1.0 / (self.params[m] * self.n))
        return samples, weights

    def sample_per_point(self, num_samples, method='random', **kwargs):
        samples = {}
        weights = {}
        if method == 'random':
            for i in range(self.n):
                samples[i] = []
                weights[i] = []
                for _ in range(num_samples // self.n):
                    perm = np.random.permutation(self.n)
                    samples[i].append(perm)
                    weights[i].append(1.0)
        elif method == 'Owen' or method.lower() == 'owen':
            runs = kwargs.get('runs', 2)
            q_splits = num_samples // self.n - 1
            for _ in range(runs):
                for q_num in range(q_splits // runs + 1):
                    q = q_num / q_splits
                    b = np.array(np.random.binomial(1, q, self.n))
                    ids = list(np.nonzero(b)[0])
                    rest = [j for j in range(self.n) if j not in ids]
                    for i in ids:
                        first = [j for j in ids if j != i]
                        perm = first + [i] + rest
                        if i not in samples:
                            samples[i] = [perm]
                            weights[i] = [1.0]
                        else:
                            samples[i].append(perm)
                            weights[i].append(1.0)
                    if q != 0.5:
                        ids, rest = rest, ids
                        for i in ids:
                            first = [j for j in ids if j != i]
                            perm = first + [i] + rest
                            if i not in samples:
                                samples[i] = [perm]
                                weights[i] = [1.0]
                            else:
                                samples[i].append(perm)
                                weights[i].append(1.0) 
        elif method == 'stratified':
            div = sum([(j+1)**(2/3) for j in range(self.n)])
            for i in range(self.n):
                quota = [0 for _ in range(self.n)]
                for k in range(self.n):
                    quota[k] = math.floor(num_samples * (k+1)**(2/3) / div)
                left = num_samples - sum(quota)
                for j in range(left):
                    quota[j] += 1
                for k in range(self.n):
                    players = [r for r in range(self.n) if i != r]
                    s = [np.random.permutation(players) for _ in range(quota[k])]
                    samples[k] = []
                    weights[k] = []
                    for r in range(len(s)):
                        samples[k].append(np.insert(s[r], i, k))
                        weights[k].append(1.0)

        elif method == 'Sobol' or method.lower() == 'sobol':
            dimension = self.n # number of players or lengthe of the permutation
            U_hat = get_U_hat(dimension)

            num_samples //= self.n
            sampled_permutations = []

            sampler = Sobol(d=dimension-2, scramble=True) # fixing seed would make all sampler the same
            sobol_sequence = sampler.random_base2(m=math.ceil(np.log2(num_samples)))
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

            if num_samples != len(sampled_permutations):
                print(f'requested num_samples is {num_samples}, number of sampled permutations is {len(sampled_permutations)}, returning the first {num_samples} sampled permutations.')
                print('It is advised to sample a number that is an exact power of 2, of permutations to enjoy the theoretical properties of Sobol sequence.')
                sampled_permutations = sampled_permutations[:num_samples]

            for perm in sampled_permutations:
                for i in perm:
                    if i not in samples:
                        samples[i] = [perm]
                        weights[i] = [1.0]
                    else:
                        samples[i].append(perm)
                        weights[i].append(1.0)
        else:
            raise NotImplementedError

        return samples, weights

    def sample_sequential(self, num_samples, method='random', **kwargs):
        samples = []
        weights = []
        if method == 'random':
            for _ in range(num_samples):
                perm = np.random.permutation(self.n)
                samples.append(perm)
                weights.append(1.0)
        elif method == 'exact':
            print("Compute Exact Shapley value. Disard num_samples.")
            for bits in range(2**self.n):
                coalition_bits = bin(bits).replace('0b', '').zfill(self.n)
                coalition = [i for i, b in enumerate(coalition_bits) if b == '1']
                rest = [i for i in range(self.n) if i not in coalition]
                for i in coalition:
                    first = [j for j in coalition if j != i]
                    samples.append(first + [i] + rest)
                    weights.append(1.0)
        else:
            raise NotImplementedError
        return samples, weights
