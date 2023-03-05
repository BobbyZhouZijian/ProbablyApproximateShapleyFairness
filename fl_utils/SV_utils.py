from itertools import permutations
from copy import deepcopy
from math import factorial

import torch

from .utils import add_update_to_model, evaluate


def exact(model, gradients, valid_loader, device='cuda'):

	N = len(gradients)
	# if N >=5 :
		# return exact_parallel(model ,gradients, valid_loader, device)

	SVs = torch.zeros(N, device=device)

	backup_model = deepcopy(model)

	perms = list(permutations(range(N)))
	with torch.no_grad():
		for perm in perms:

			for index in perm:
				gradient = gradients[index]

				_, acc_before = evaluate(model, valid_loader, loss_fn=None, device=device)

				model = add_update_to_model(model, gradient, device=device)

				_, acc_after = evaluate(model, valid_loader, loss_fn=None, device=device)

				marginal = acc_after - acc_before
				SVs[index] += marginal

			model.load_state_dict(backup_model.state_dict())

		SVs = torch.divide(SVs, factorial(N))

	return SVs

import random
def MC(model, gradients, valid_loader, device='cuda', sampling_count=150):
	
	'''
	np.log(2/0.1)/ (2*0.01) = 150,  for epsilon=delta=0.1, and the test accuracy is bounded by 1
	We use default sampling count 150.
	'''

	N = len(gradients)

	SVs = torch.zeros(N, device=device)

	backup_model = deepcopy(model)

	perms = list(permutations(range(N)))
	random.shuffle(perms)

	with torch.no_grad():
		for perm in perms[:sampling_count]:

			for index in perm:
				gradient = gradients[index]

				_, acc_before = evaluate(model, valid_loader, loss_fn=None, device=device)

				model = add_update_to_model(model, gradient, device=device)

				_, acc_after = evaluate(model, valid_loader, loss_fn=None, device=device)

				marginal = acc_after - acc_before
				SVs[index] += marginal

			model.load_state_dict(backup_model.state_dict())

		SVs = torch.divide(SVs, factorial(N))

	return SVs



def get_SV_by_permutations(model, gradients, valid_loader, perms, device='device'):
	N = len(gradients)

	SVs = torch.zeros(N, device=device)

	backup_model = deepcopy(model)

	with torch.no_grad():
		for perm in perms:

			for index in perm:
				gradient = gradients[index]

				_, acc_before = evaluate(model, valid_loader, loss_fn=None, device=device)

				model = add_update_to_model(model, gradient, device=device)

				_, acc_after = evaluate(model, valid_loader, loss_fn=None, device=device)

				marginal = acc_after - acc_before
				SVs[index] += marginal

			model.load_state_dict(backup_model.state_dict())

		SVs = torch.divide(SVs, factorial(N))

	return SVs

def exact_parallel(model, gradients, valid_loader, device='cuda'):

	import numpy as np
	import multiprocessing
	from multiprocessing.pool import ThreadPool

	n_cores = multiprocessing.cpu_count() - 2
	N = len(gradients)
	perms = list(permutations(range(N)))
	sub_perms = np.array_split(perms, 5)
	
	with ThreadPool(n_cores) as pool:
		input_arguments = [(model, gradients, valid_loader, sub_perm, device) for sub_perm in sub_perms]
		output = pool.starmap(get_SV_by_permutations, input_arguments)

	N = len(gradients)
	SVs_overall = torch.zeros(N)
	for sub_SVs in output:
		SVs_overall += sub_SVs

	return SVs_overall