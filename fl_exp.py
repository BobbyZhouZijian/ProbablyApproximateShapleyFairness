from os.path import join as oj
import copy
from collections import defaultdict
import pickle
from functools import lru_cache
from scipy.stats import sem

import numpy as np
import pandas as pd
import argparse

import torch
from torch import nn, optim
from torch.linalg import norm
import torch.nn.functional as F
from torchtext.data import Batch


from fl_utils.Data_Prepper import Data_Prepper
from fl_utils.arguments import mnist_args, cifar_cnn_args, mr_args, sst_args
from fl_utils.utils import compute_grad_update, add_update_to_model, add_gradient_updates,\
    flatten, unflatten, train_model, evaluate, cosine_similarity, mask_grad_update_by_order
from fl_utils.utils import cwd, set_up_plotting, init_deterministic

from exp_utils import exact, generic_sampler, kernelSHAP, active, _get_SV_estimates


parser = argparse.ArgumentParser(description='Process which dataset to run')
parser.add_argument('-d', '--dataset', help='Dataset name', type=str, required=True)
parser.add_argument('-N', '--participants', help='The number of participants', type=int, default=5)

parser.add_argument('-m', '--num_samples', help='The number of samples (i.e., permutations for SV estimation).', type=int, default=1000)
parser.add_argument('-t', '--trials', help='Number of independent random trials to run.', type=int, default=5)
parser.add_argument('-p', '--p', help='Strength of dirichlet prior.', type=int, default=2)


parser.add_argument('-cuda', dest='cuda', help='Use cuda if available.', action='store_true', default=True)
# parser.add_argument('-nocuda', dest='cuda', help='Not to use cuda even if available.', action='store_false')


cmd_args = parser.parse_args()
print(cmd_args)

N = cmd_args.participants
DATA_PATH = "./experiment_data"


if torch.cuda.is_available() and cmd_args.cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


if cmd_args.dataset == 'mnist':
    args = copy.deepcopy(mnist_args)

    if N > 0:
        participant_rounds = [[N, N*600]]
    else:
        participant_rounds = [[5, 3000], [10, 6000], [20, 12000]]
    # splits = ['powerlaw'] 
    splits = ['uniform', 'classimbalance', 'powerlaw'] 
    args['rounds'] = 60
    args['E'] = 1
    args['n_participants'] = 10
    args['sample_size_cap'] = 6000

elif cmd_args.dataset == 'cifar10':
    args = copy.deepcopy(cifar_cnn_args)    
    participant_rounds = [[10, 20000]]
    splits = ['classimbalance', 'powerlaw', 'uniform']
    args['rounds'] = 100
    args['E'] = 3
    args['n_participants'] = 10
    args['sample_size_cap'] = 20000


# only run with N=5
elif cmd_args.dataset == 'sst':
    args = copy.deepcopy(sst_args)  
    participant_rounds = [[5, 8000]]
    splits = ['powerlaw']
    args['rounds'] = 200
    args['E'] = 3
    args['n_participants'] = 5
    args['sample_size_cap'] = 8000

# only run with N=5 
elif cmd_args.dataset == 'mr':
    args = copy.deepcopy(mr_args)   
    participant_rounds = [[5, 8000]]
    splits = ['powerlaw']
    args['rounds'] = 200
    args['E'] = 3
    args['n_participants'] = 5
    args['sample_size_cap'] = 8000

E = args['E']


# ----------------- Guarantee deterministic behavior -------------------
seed =1234
init_deterministic(seed)


for split in splits:
    args['split'] = split #powerlaw ,  classimbalance

    for n_trial in range(cmd_args.trials):

        # for n_participants, sample_size_cap in participant_rounds:
            # args['n_participants'] = n_participants
            # args['sample_size_cap'] = sample_size_cap
        optimizer_fn = args['optimizer_fn']
        loss_fn = args['loss_fn']

        print(args)
        print("Data Split information for honest participants:")
        data_prepper = Data_Prepper(
            args['dataset'], train_batch_size=args['batch_size'], n_participants=args['n_participants'], sample_size_cap=args['sample_size_cap'], 
            train_val_split_ratio=args['train_val_split_ratio'], device=device, args_dict=args)

        valid_loader = data_prepper.get_valid_loader()
        test_loader = data_prepper.get_test_loader()

        train_loaders = data_prepper.get_train_loaders(args['n_participants'], args['split'])
        shard_sizes = data_prepper.shard_sizes


        shard_sizes = torch.tensor(shard_sizes).float()
        relative_shard_sizes = torch.div(shard_sizes, torch.sum(shard_sizes))           
        print("Shard sizes are: ", shard_sizes.tolist())

        if args['dataset'] in ['mr', 'sst']:
            server_model = args['model_fn'](args=data_prepper.args).to(device)
        else:
            server_model = args['model_fn']().to(device)

        D = sum([p.numel() for p in server_model.parameters()])

        # ---- init honest participants ----
        participant_models, participant_optimizers, participant_schedulers = [], [], []

        for i in range(N):
            model = copy.deepcopy(server_model)
            optimizer = optimizer_fn(model.parameters(), lr=args['lr'])
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = args['lr_decay'])

            participant_models.append(model)
            participant_optimizers.append(optimizer)
            participant_schedulers.append(scheduler)

        valid_perfs = defaultdict(list) # validation performance results
        local_perfs = defaultdict(list) # local training dataset performance results

        overall_SVs = defaultdict(list)
        overall_statistics = defaultdict(list)

        # ---- FL begins ---- 
        for _round in range(args['rounds']):

            gradients = []

            # ---- Honest participants ---- 
            for i in range(N):
                loader = train_loaders[i]
                model = participant_models[i]
                optimizer = participant_optimizers[i]
                scheduler = participant_schedulers[i]

                model.train()
                model = model.to(device)
                backup = copy.deepcopy(model)
                model = train_model(model, loader, loss_fn, optimizer, device=device, E=E, scheduler=scheduler)
                gradient = compute_grad_update(old_model=backup, new_model=model, device=device)
                gradients.append(gradient)

            fedavg_gradients = []
            for gradient, weight in zip(gradients, relative_shard_sizes):
                gradient = [torch.multiply(layer_grad, weight) for layer_grad in gradient ]
                fedavg_gradients.append(gradient)


            # ---- Calculating/Approximating SVs ----
            def FL_utility(S): 
                return _utility(tuple(sorted(S)))

            @lru_cache(maxsize=min(2**N, 1024))
            def _utility(S):
                if len(S) == 0:return 0
                
                else:
                    S_model = copy.deepcopy(server_model)
                    for i in S:
                        S_model = add_update_to_model(S_model, fedavg_gradients[i])                     
                    _, acc = evaluate(S_model, test_loader, loss_fn=None, device=device)
                    return acc.item()

            epoch_SVs = exact(N, FL_utility)
            overall_SVs['Exact'].append(epoch_SVs / np.mean(epoch_SVs))

            # kernelSHAP
            mcs, afs, min_afs = kernelSHAP(N, cmd_args.num_samples, FL_utility, seed=n_trial, bootstrap_n=200)
            s_estimates = _get_SV_estimates(N, mcs)
            overall_SVs['KernelSHAP'].append(s_estimates)
            overall_statistics['KernelSHAP'].append(([], afs, min_afs))

            # active: 2-FAE
            for alpha in [0, 2, 5, 100]:
                mcs, afs, min_afs = active(N, cmd_args.num_samples, FL_utility, seed=n_trial, bootstrap_n=200, alpha=alpha)
                s_estimates = _get_SV_estimates(N, mcs)
                overall_SVs['2-FAE-a'+str(alpha)].append(s_estimates)
                overall_statistics['2-FAE-a'+str(alpha)].append((mcs, afs, min_afs))


            # other samplers
            for method in ['Sobol', 'Stratified', 'Owen', 'MC']:
                # print(f'Executing {method} now.')
                mcs, afs, min_afs = generic_sampler(method, N, cmd_args.num_samples, FL_utility, seed=n_trial, bootstrap_n=200)
                s_estimates = _get_SV_estimates(N, mcs)
                overall_SVs[method].append(s_estimates)
                overall_statistics[method].append(([], afs, min_afs))


            # ---- Server Aggregate ----
            for gradient in fedavg_gradients:
                server_model = add_update_to_model(server_model, gradient)

            # update the weights of each local model
            for i in range(N):
                participant_models[i].load_state_dict(server_model.state_dict())

            round_accs = []
            for i, model in enumerate(participant_models):
                loss, accuracy = evaluate(model, valid_loader, loss_fn=loss_fn, device=device)

                round_accs.append(accuracy.item())

                valid_perfs[str(i)+'_loss'].append(loss.item())
                valid_perfs[str(i)+'_accu'].append(accuracy.item())

                fed_loss, fed_accu = 0, 0
                for j, train_loader in enumerate(train_loaders):
                    loss, accuracy = evaluate(model, train_loader, loss_fn=loss_fn, device=device)

                    if j == i:
                        local_perfs[str(i)+'_loss'].append(loss.item())
                        local_perfs[str(i)+'_accu'].append(accuracy.item())
            # print("round mean accuarcy {} max accuracy {}".format(np.mean(round_accs), max(round_accs)))

        # ---- Results saving ---- 
        suffix = f'{args["split"][:3].upper()}-n{N}-num_samples{cmd_args.num_samples}-trials-{n_trial+1}of{cmd_args.trials}'

        with cwd(oj(DATA_PATH, 'fl', args['dataset'], suffix)):

            for method, method_overall_SVs in overall_SVs.items():
                method_SVs = np.stack(method_overall_SVs)               
                overall_SVs_df = pd.DataFrame(method_SVs)
                overall_SVs_df.to_csv(f'{method}-SVs.csv', index=False)
                mean, std_err = np.mean(method_SVs, 0), sem(method_SVs, axis=0)

                print(f'FLSV over epochs: {method}: mean {mean}, std-err {std_err}.')

                np.savez(f'{N}n_{cmd_args.trials}t_{cmd_args.num_samples}samples.npz',
                    overall_statistics=overall_statistics, overall_SVs=overall_SVs, 
                    n=N, t=cmd_args.trials, num_samples=cmd_args.num_samples, dataset=cmd_args.dataset, seed=1234)


            df = pd.DataFrame(valid_perfs)
            # print("Validation performance:")
            df.to_csv('valid.csv', index=False)
            
            df = pd.DataFrame(local_perfs)
            df.to_csv('local.csv', index=False)

            with open('settings_dict.txt', 'w') as file:
                [file.write(key + ' : ' + str(value) + '\n') for key, value in args.items()]

            with open('settings_dict.pickle', 'wb') as f: 
                pickle.dump(args, f)
             
            # pickle - loading 
            # with open(oj(folder,'settings_dict.pickle'), 'rb') as f: 
                # args = pickle.load(f) 

