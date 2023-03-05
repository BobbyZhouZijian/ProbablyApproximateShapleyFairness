import numpy as np
from sklearn.datasets import fetch_covtype, fetch_openml, load_diabetes, load_breast_cancer


def make_balance_sample(data, target, target_label=1, seed=2017):
    np.random.seed(seed)
    p = np.mean(target)
    if p < 0.5:
        # target label is the minor class
        index_minor_class = np.where(target == target_label)[0]
    else:
        # target label is the major class
        index_minor_class = np.where(target != target_label)[0]

    n_minor_class = len(index_minor_class)
    n_major_class = len(target) - n_minor_class
    new_minor = np.random.choice(index_minor_class, size=n_major_class - n_minor_class, replace=True)

    data = np.concatenate([data, data[new_minor]])
    target = np.concatenate([target, target[new_minor]])
    return data, target


def load_classification_dataset(n_data_to_be_valued=200, n_val=100, n_test=1000, dataset='gaussian', clf_path='../data'
                                , seed=2022):
    np.random.seed(seed)
    if dataset == 'gaussian':
        print('-' * 50)
        print('Gaussian Classification Dataset')
        print('-' * 50)
        n, input_dim = 50000, 5
        data = np.random.normal(size=(n, input_dim))
        beta_true = np.array([2.0, 1.0, 0.0, 0.0, 0.0]).reshape(input_dim, 1)
        p_true = np.exp(data.dot(beta_true)) / (1.0 + np.exp(data.dot(beta_true)))
        target = np.random.binomial(n=1, p=p_true).reshape(-1)
    elif dataset == 'covertype':
        print('-' * 50)
        print('Covertype Dataset')
        print('-' * 50)
        data, target = fetch_covtype(data_home=clf_path, return_X_y=True)
        target = ((target == 1) + 0.0).astype(int)
        data, target = make_balance_sample(data, target)
    elif dataset == 'breast_cancer':
        print('-' * 50)
        print('Breast cancer Dataset')
        print('-' * 50)
        data, target = load_breast_cancer(return_X_y=True)
        target = ((target == 1) + 0.0).astype(int)
        data, target = make_balance_sample(data, target)
    elif dataset == 'mnist':
        print('-' * 50)
        print('MNIST 784 Dataset')
        print('-' * 50)
        data, target = fetch_openml('mnist_784', data_home=clf_path, return_X_y=True)
        # convert data from DataFrame to np array
        data = np.asarray(data)
        # data = data.to_numpy()
        # change to 2 cls
        target = ((target == '1') + 0.0).astype(int)
        data, target = make_balance_sample(data, target)
    else:
        raise NotImplementedError

    random_idxs = list(range(len(data)))
    np.random.shuffle(random_idxs)
    data = data[random_idxs]
    target = target[random_idxs]
    X = data[:n_data_to_be_valued]
    y = target[:n_data_to_be_valued]
    X_val = data[n_data_to_be_valued:(n_data_to_be_valued + n_val)]
    y_val = target[n_data_to_be_valued:(n_data_to_be_valued + n_val)]
    X_test = data[(n_data_to_be_valued + n_val):(n_data_to_be_valued + n_val + n_test)]
    y_test = target[(n_data_to_be_valued + n_val):(n_data_to_be_valued + n_val + n_test)]

    print(f'number of samples: {len(X)}')
    X_mean, X_std = np.mean(X, 0), np.std(X, 0)
    normalizer_fn = lambda x: (x - X_mean) / np.clip(X_std, 1e-12, None)
    X, X_val, X_test = normalizer_fn(X), normalizer_fn(X_val), normalizer_fn(X_test)

    return (X, y), (X_val, y_val), (X_test, y_test)


def load_regression_dataset(n_data_to_be_valued=200, n_val=100, n_test=1000, dataset='gaussian', clf_path='../data'
                            , seed=2022):
    if dataset == 'diabetes':
        print('-' * 48)
        print('Diabetes dataset')
        print('-' * 48)
        data, target = load_diabetes(return_X_y=True)
    else:
        raise NotImplementedError

    random_idxs = list(range(len(data)))
    np.random.seed(seed)
    np.random.shuffle(random_idxs)
    data = data[random_idxs]
    target = target[random_idxs]
    X = data[:n_data_to_be_valued]
    y = target[:n_data_to_be_valued]
    X_val = data[n_data_to_be_valued:(n_data_to_be_valued + n_val)]
    y_val = target[n_data_to_be_valued:(n_data_to_be_valued + n_val)]
    X_test = data[(n_data_to_be_valued + n_val):(n_data_to_be_valued + n_val + n_test)]
    y_test = target[(n_data_to_be_valued + n_val):(n_data_to_be_valued + n_val + n_test)]

    print(f'number of samples: {len(X)}')
    X_mean, X_std = np.mean(X, 0), np.std(X, 0)
    normalizer_fn = lambda x: (x - X_mean) / np.clip(X_std, 1e-12, None)
    X, X_val, X_test = normalizer_fn(X), normalizer_fn(X_val), normalizer_fn(X_test)

    return (X, y), (X_val, y_val), (X_test, y_test)


def load_data(task, dataset_name, is_noisy=False, **dargs):
    if task == 'classification':
        n_data_to_be_valued = dargs.get('n_data_to_be_valued', 200)
        n_val = dargs.get('n_val', 200)
        n_test = dargs.get('n_test', 1000)
        rid = dargs.get('rid', 1)  # this is used as an identifier for cifar100_test dataset
        (X, y), (X_val, y_val), (X_test, y_test) = load_classification_dataset(n_data_to_be_valued=n_data_to_be_valued,
                                                                               n_val=n_val,
                                                                               n_test=n_test,
                                                                               dataset=dataset_name)

    elif task == 'regression':
        n_data_to_be_valued = dargs.get('n_data_to_be_valued', 200)
        n_val = dargs.get('n_val', 200)
        n_test = dargs.get('n_test', 1000)
        (X, y), (X_val, y_val), (X_test, y_test) = load_regression_dataset(n_data_to_be_valued=n_data_to_be_valued,
                                                                               n_val=n_val,
                                                                               n_test=n_test,
                                                                               dataset=dataset_name)
    else:
        raise NotImplementedError('Check problem')

    if is_noisy:
        # training is flipped
        flipped_index = np.random.choice(np.arange(n_data_to_be_valued), n_data_to_be_valued // 10, replace=False)
        y[flipped_index] = (1 - y[flipped_index])

        # validation is also flipped
        flipped_val_index = np.random.choice(np.arange(n_val), n_val // 10, replace=False)
        y_val[flipped_val_index] = (1 - y_val[flipped_val_index])
        return (X, y), (X_val, y_val), (X_test, y_test), flipped_index
    else:
        return (X, y), (X_val, y_val), (X_test, y_test)
