import copy
import math
import random
import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from sklearn.preprocessing import StandardScaler

def init_deterministic():
    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

use_cuda = True
cuda_available = torch.cuda.is_available()


def update_gpu(args):
    if 'cuda' in str(args['device']):
        args['device'] = torch.device('cuda:{}'.format(args['gpu']))
    if torch.cuda.device_count() > 0:
        args['device_ids'] = [device_id for device_id in range(torch.cuda.device_count())]
    else:
        args['device_ids'] = []



import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

# for MNIST 32*32
class CNN_Net(nn.Module):

    def __init__(self, device=None):
        super(CNN_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, 1)
        self.conv2 = nn.Conv2d(64, 16, 7, 1)
        self.fc1 = nn.Linear(4 * 4 * 16, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):
        x = x.view(-1, 1, 32, 32)
        x = torch.tanh(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = torch.tanh(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 16)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def get_train_valid_indices(n_samples, train_val_split_ratio, sample_size_cap=None):
    indices = list(range(n_samples))
    random.seed(1111)
    random.shuffle(indices)
    split_point = int(n_samples * train_val_split_ratio)
    train_indices, valid_indices = indices[:split_point], indices[split_point:]
    if sample_size_cap is not None:
        train_indices = indices[:min(split_point, sample_size_cap)]

    return  train_indices, valid_indices 

def powerlaw(sample_indices, n_participants, alpha=1.65911332899, shuffle=False):
    # the smaller the alpha, the more extreme the division
    if shuffle:
        random.seed(1234)
        random.shuffle(sample_indices)

    from scipy.stats import powerlaw
    import math
    party_size = int(len(sample_indices) / n_participants)
    b = np.linspace(powerlaw.ppf(0.01, alpha), powerlaw.ppf(0.99, alpha), n_participants)
    shard_sizes = list(map(math.ceil, b/sum(b)*party_size*n_participants))
    indices_list = []
    accessed = 0
    for participant_id in range(n_participants):
        indices_list.append(sample_indices[accessed:accessed + shard_sizes[participant_id]])
        accessed += shard_sizes[participant_id]
    return indices_list

def scale_normal(datasets, datasets_test):
    """
        Scale both the training and test set to standard normal distribution. The training set is used to fit.
        Args:
            datasets (list): list of datasets of length M
            datasets_test (list): list of test datasets of length M
        
        Returns:
            two lists containing the standarized training and test dataset
    """
    
    scaler = StandardScaler()
    scaler.fit(torch.vstack(datasets))
    return [torch.from_numpy(scaler.transform(dataset)).float() for dataset in datasets], [torch.from_numpy(scaler.transform(dataset)).float() for dataset in datasets_test]

def get_synthetic_datasets(n_participants, d=1, sizes=[], s=50, ranges=None):
    """
        Args:
            n_participants (int): number of data subsets to generate
            d (int): dimension
            sizes (list of int): number of data samples for each participant, if supplied
            s (int): number of data samples for each participant (equal), if supplied
            ranges (list of list): the lower and upper bound of the input domain for each participant, if supplied

        Returns:
            list containing the generated synthetic datasets for all participants
    """

    if 0 == len(sizes): 
        sizes = torch.ones(n_participants, dtype=int) * s

    datasets = []
    for i, size in enumerate(sizes):
        if ranges != None:
            dataset = torch.rand((size, d)) * (ranges[i][1] - ranges[i][0]) + ranges[i][0]
        else:
            dataset = torch.rand((size, d)) * (1 - 0) + 0 
            # dataset = np.random.uniform(0, 1, (size, d))
            # dataset = np.random.normal(0, 1, (size,d))
        datasets.append(dataset.reshape(-1, d))
    return datasets

def generate_linear_labels(datasets, d=1, weights=None, bias=None):

    # generate random true weights and the bias
    if weights is None:
        weights = torch.normal(0, 1, size=(d,))
    if bias is None:
        bias = torch.normal(0, 1, size=(1,))


    labels = []
    w_b = torch.cat((weights, bias))
    for X in datasets:
        one_padded_X = torch.cat((X, torch.ones((len(X), 1))), axis=1)
        y = (one_padded_X @ w_b).reshape(-1, 1)
        labels.append(y)
    return labels, weights, bias


def friedman_function(X, noise_std=0):
    """
    Create noisy friedman values: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_friedman1.html

    """
    assert X.shape[1] >= 5, "The input features must have at least 5 dimensions."
    M = len(X)
    y = 10 * torch.sin(math.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2 + 10 * X[:, 3] + 5 * X[:, 4] 
    return y.reshape(M, 1)

import torch
from botorch.test_functions.synthetic import Hartmann

def hartmann_function(X, noise_std=0.05):
    """
    Create noisy Hartmann values: https://www.sfu.ca/~ssurjano/hart4.html

    """

    (M, dim) = X.shape
    assert dim in (3, 4, 6), "Hartmann function of dimensions: (3,4,6) is implemented."

    neg_hartmann = Hartmann(dim=dim, negate=True)
    y = neg_hartmann(X)
    return y.reshape(M, 1)



def get_datasets(n_participants, split='powerlaw', device=None):

    train = FastMNIST('datasets/MNIST', train=True, download=True)
    test = FastMNIST('datasets/MNIST', train=False, download=True)

    train_indices, valid_indices = get_train_valid_indices(len(train), 0.8, n_participants * 600)
    del train, test

    if split == 'classimbalance':
        n_classes = 10          
        data_indices = [torch.nonzero(train_set.targets == class_id).view(-1).tolist() for class_id in range(n_classes)]
        class_sizes = np.linspace(1, n_classes, n_participants, dtype='int')
        print("class_sizes for each party", class_sizes)
        party_mean = 600

        from collections import defaultdict
        party_indices = defaultdict(list)
        for party_id, class_sz in enumerate(class_sizes):   
            classes = range(class_sz) # can customize classes for each party rather than just listing
            each_class_id_size = party_mean // class_sz
            # print("party each class size:", party_id, each_class_id_size)
            for i, class_id in enumerate(classes):
                # randomly pick from each class a certain number of samples, with replacement 
                selected_indices = random.choices(data_indices[class_id], k=each_class_id_size)

                # randomly pick from each class a certain number of samples, without replacement 
                '''
                NEED TO MAKE SURE THAT EACH CLASS HAS MORE THAN each_class_id_size for no replacement sampling
                selected_indices = random.sample(data_indices[class_id],k=each_class_id_size)
                '''
                party_indices[party_id].extend(selected_indices)

                # top up to make sure all parties have the same number of samples
                if i == len(classes) - 1 and len(party_indices[party_id]) < party_mean:
                    extra_needed = party_mean - len(party_indices[party_id])
                    party_indices[party_id].extend(data_indices[class_id][:extra_needed])
                    data_indices[class_id] = data_indices[class_id][extra_needed:]

        indices_list = [party_index_list for party_id, party_index_list in party_indices.items()] 

    elif split == 'powerlaw':   
        indices_list = powerlaw(list(range(len(train_set))), n_participants)

    train_datasets = [   ]

    for i, indices in enumerate(indices_list):
        train_datasets.append(Custom_Dataset(train.data[train_indices], train.targets[train_indices], device=device))

    validation_set = Custom_Dataset(train.data[valid_indices],train.targets[valid_indices] , device=device)
    test_set = Custom_Dataset(test.data, test.targets, device=device)

    return  train_datasets, validation_set, test_set


def get_train_loaders(n_participants, split='powerlaw', batch_size=32, device=None):

    train = FastMNIST('datasets/MNIST', train=True, download=True)
    test = FastMNIST('datasets/MNIST', train=False, download=True)

    train_indices, valid_indices = get_train_valid_indices(len(train), 0.8, n_participants * 600)
    
    train_set = Custom_Dataset(train.data[train_indices], train.targets[train_indices], device=device)
    validation_set = Custom_Dataset(train.data[valid_indices],train.targets[valid_indices] , device=device)
    test_set = Custom_Dataset(test.data, test.targets, device=device)

    del train, test

    if split == 'classimbalance':
        n_classes = 10          
        data_indices = [torch.nonzero(train_set.targets == class_id).view(-1).tolist() for class_id in range(n_classes)]
        class_sizes = np.linspace(1, n_classes, n_participants, dtype='int')
        print("class_sizes for each party", class_sizes)
        party_mean = 600

        from collections import defaultdict
        party_indices = defaultdict(list)
        for party_id, class_sz in enumerate(class_sizes):   
            classes = range(class_sz) # can customize classes for each party rather than just listing
            each_class_id_size = party_mean // class_sz
            # print("party each class size:", party_id, each_class_id_size)
            for i, class_id in enumerate(classes):
                # randomly pick from each class a certain number of samples, with replacement 
                selected_indices = random.choices(data_indices[class_id], k=each_class_id_size)

                # randomly pick from each class a certain number of samples, without replacement 
                '''
                NEED TO MAKE SURE THAT EACH CLASS HAS MORE THAN each_class_id_size for no replacement sampling
                selected_indices = random.sample(data_indices[class_id],k=each_class_id_size)
                '''
                party_indices[party_id].extend(selected_indices)

                # top up to make sure all parties have the same number of samples
                if i == len(classes) - 1 and len(party_indices[party_id]) < party_mean:
                    extra_needed = party_mean - len(party_indices[party_id])
                    party_indices[party_id].extend(data_indices[class_id][:extra_needed])
                    data_indices[class_id] = data_indices[class_id][extra_needed:]

        indices_list = [party_index_list for party_id, party_index_list in party_indices.items()] 

    elif split == 'powerlaw':   
        indices_list = powerlaw(list(range(len(train_set))), n_participants)

    participant_train_loaders = [DataLoader(train_set, batch_size=batch_size, sampler=SubsetRandomSampler(indices)) for indices in indices_list]
    valid_loader = DataLoader(validation_set, batch_size=128)
    test_loader = DataLoader(test_set, batch_size=128)
    return participant_train_loaders, test_loader 


from torch.utils.data import Dataset

class Custom_Dataset(Dataset):

    def __init__(self, X, y, device=None, transform=None):
        self.data = X.to(device)
        self.targets = y.to(device)
        self.count = len(X)
        self.device = device
        self.transform = transform

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        if self.transform:
            return self.transform(self.data[idx]), self.targets[idx]

        return self.data[idx], self.targets[idx]


from torchvision.datasets import MNIST
class FastMNIST(MNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)       
        
        self.data = self.data.unsqueeze(1).float().div(255)
        from torch.nn import ZeroPad2d
        pad = ZeroPad2d(2)
        self.data = torch.stack([pad(sample.data) for sample in self.data])

        self.targets = self.targets.long()

        self.data = self.data.sub_(self.data.mean()).div_(self.data.std())
        # self.data = self.data.sub_(0.1307).div_(0.3081)
        # Put both data and targets on GPU in advance
        print('MNIST data shape {}, targets shape {}'.format(self.data.shape, self.targets.shape))
        if 'device' in kwargs:
            self.data, self.targets = self.data.to(kwargs['device']), self.targets.to(kwargs['device'])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target


def compute_grad_update(old_model, new_model, device=None):
    # maybe later to implement on selected layers/parameters
    if device:
        old_model, new_model = old_model.to(device), new_model.to(device)
    return [(new_param.data - old_param.data) for old_param, new_param in zip(old_model.parameters(), new_model.parameters())]

def add_update_to_model(model, update, weight=1.0, device=None):
    if not update: return model
    if device:
        model = model.to(device)
        update = [param.to(device) for param in update]
            
    for param_model, param_update in zip(model.parameters(), update):
        param_model.data += weight * param_update.data
    return model

def add_gradient_updates(grad_update_1, grad_update_2, weight = 1.0):
    assert len(grad_update_1) == len(
        grad_update_2), "Lengths of the two grad_updates not equal"
    
    for param_1, param_2 in zip(grad_update_1, grad_update_2):
        param_1.data += param_2.data * weight

def flatten(grad_update):
    return torch.cat([update.data.view(-1) for update in grad_update])

def unflatten(flattened, normal_shape):
    grad_update = []
    for param in normal_shape:
        n_params = len(param.view(-1))
        grad_update.append(torch.as_tensor(flattened[:n_params]).reshape(param.size())  )
        flattened = flattened[n_params:]

    return grad_update

def mask_grad_update_by_order(grad_update, mask_percentile):
    grad_update = copy.deepcopy(grad_update)

    mask_percentile = max(0, mask_percentile)
    for i, layer in enumerate(grad_update):
        layer_mod = layer.data.view(-1).abs()
        if mask_percentile is not None:
            mask_order = math.ceil(len(layer_mod) * mask_percentile)

        if mask_order == 0:
            grad_update[i].data = torch.zeros(layer.data.shape, device=layer.device)
        else:
            topk, indices = torch.topk(layer_mod, min(mask_order, len(layer_mod)-1))
            grad_update[i].data[layer.data.abs() < topk[-1]] = 0
    return grad_update