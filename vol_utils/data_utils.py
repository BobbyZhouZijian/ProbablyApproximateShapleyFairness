import os
from os.path import join as oj
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import torch

# from kaggle.api.kaggle_api_extended import KaggleApi
# api = KaggleApi()
# api.authenticate()


# def get_house_sales_datasets(n_participants, d=None, sizes=[], s=50):

# 	df = pd.read_csv(".data/House_sales/kc_house_data.csv")

# 	perc_05 = df['price'].quantile(0.05)
# 	perc_95 = df['price'].quantile(0.9)
# 	df = df[( (perc_05 <= df['price']) & (df['price'] <= perc_95) )]

# 	cols_to_remove = ['id', 'date', 'lat', 'long', 'waterfront', 'zipcode']
# 	df = df.drop(columns = cols_to_remove)

# 	df['age'] = 2021 - df['yr_built']
# 	df['rennovated'] = (df['yr_renovated'] != 0).astype(int)
# 	cols_to_remove = ['yr_built', 'yr_renovated']
# 	df = df.drop(columns=cols_to_remove)


# 	from sklearn import preprocessing

# 	y = df['price']/ 1e7
# 	X = df.drop(columns=['price'])
# 	names = X.columns
# 	min_max_scaler = preprocessing.MinMaxScaler()
# 	x_scaled = min_max_scaler.fit_transform(X.values)
# 	df = pd.DataFrame(x_scaled, columns=names)

# 	from sklearn.model_selection import train_test_split
# 	X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)


# 	datasets, labels  = [], []
# 	if len(sizes) == 0:
# 		sizes = [s for i in range(n_participants)]


# 	for size in sizes:
# 		indices = np.random.choice(np.arange(len(X_train)), size)
		
# 		datasets.append(torch.from_numpy(X_train[indices]))
# 		labels.append(torch.from_numpy(y_train[indices]))

# 	return datasets, labels, torch.from_numpy(X_test), torch.from_numpy(y_test)


def download_dataset(dataset_name='House_sales', data_folder_dir='.data'):
	'''
	Args:
		dataset_name (str): dataset name, from ['House_sales', 'California_housing', 'Used_car', 'COVID_hospital', 'Uber_lyft', 'Hotel_review']
		data_folder_dir (str): to save the downloaded datasets
	'''
	dataset_names = ['House_sales', 'California_housing', 'Used_car', 'COVID_hospital', 'Uber_lyft', 'Hotel_review']
	kaggle_names = {
		'House_sales': 'harlfoxem/housesalesprediction',
		'California_housing': 'camnugent/california-housing-prices',
		'Used_car': 'adityadesai13/used-car-dataset-ford-and-mercedes', 
		'COVID_hospital': 'tanmoyx/covid19-patient-precondition-dataset', 
		'Uber_lyft': 'brllrb/uber-and-lyft-dataset-boston-ma', 
		'Hotel_review': 'jiashenliu/515k-hotel-reviews-data-in-europe'
	}

	assert dataset_name in dataset_names, "{} dataset is not implemented.".format(dataset_name)

	print("Downloading the {} dataset into the directiory : {}.".format(dataset_name, data_folder_dir))
	dataset_dir = oj(data_folder_dir, dataset_name)
	os.makedirs(dataset_dir, exist_ok=True)

	# Signature: dataset_download_files(dataset, path=None, force=False, quiet=True, unzip=False)
	api.dataset_download_files(kaggle_names[dataset_name], path=dataset_dir, unzip=True)
	return


def load_used_car(n_participants=3, s=2000, train_test_diff_distr=False):
    """     
	    Method to load the Used_car dataset.
        Args:
            n_participants (int): number of data subsets to generate
            s (int): number of data samples for each participant (equal)
            train_test_diff_distr (bool): whether to generate a test set that has a different distribution from the train set
        Returns:
            feature_datasets, labels, feature_datasets_test, test_labels: each a list containing the loaded dataset
    """

    if n_participants > 8:
        print("Used car dataset supports at most n=8. Setting n=8.")
        n_participants = 8

    PATH = '.data/Used_car/'
    
    brands_list = ['audi', 'ford', 'toyota', 'vw', 'bmw', 'mercedez', 'vauxhall', 'skoda']
    
    # Each participant would hold data of one car brand
    brands = brands_list[:n_participants]
    
    # Load data and shuffle
    audi_df = shuffle(pd.read_csv(PATH + 'audi.csv'))
    toyota_df = shuffle(pd.read_csv(PATH + 'toyota.csv'))
    ford_df = shuffle(pd.read_csv(PATH + 'ford.csv'))
    bmw_df = shuffle(pd.read_csv(PATH + 'bmw.csv'))
    vw_df = shuffle(pd.read_csv(PATH + 'vw.csv'))
    mercedez_df = shuffle(pd.read_csv(PATH + 'merc.csv'))
    vauxhall_df = shuffle(pd.read_csv(PATH + 'vauxhall.csv'))
    skoda_df = shuffle(pd.read_csv(PATH + 'skoda.csv'))

    # Identifier
    audi_df['model'] = 'audi'
    toyota_df['model'] = 'toyota'
    ford_df['model'] = 'ford'
    bmw_df['model'] = 'bmw'
    vw_df['model'] = 'vw'
    mercedez_df['model'] = 'mercedez'
    vauxhall_df['model'] = 'vauxhall'
    skoda_df['model'] = 'skoda'

    car_manufacturers = pd.concat([audi_df[:s],
                                   toyota_df[:s],
                                   ford_df[:s],
                                   bmw_df[:s],
                                   vw_df[:s], 
                                   mercedez_df[:s],
                                   vauxhall_df[:s],
                                   skoda_df[:s],
                                   ])
    
    # Remove invalid value rows
    car_manufacturers = car_manufacturers[car_manufacturers['year'] <= 2021]
    
    # Feature selection
    X = car_manufacturers[['model', 'year', 'mpg', 'mileage', 'tax', 'engineSize']].values
    y = car_manufacturers['price'].values.reshape(-1, 1)
    
    feature_datasets, feature_datasets_test, labels, test_labels = [], [], [], []
    
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    
    if train_test_diff_distr:
        # Train data from brands
        for brand in brands:
            feature_datasets.append(torch.from_numpy(np.array(X_train[X_train[:,0] == brand][:,1:], dtype=np.float32)))
            labels.append(torch.from_numpy(np.array(y_train[X_train[:,0] == brand], dtype=np.float32)))
        
        # Make test data a different brand (distribution) as the train
        test_brand = brands_list[n_participants]
        feature_datasets_test.append(torch.from_numpy(np.array(X_test[X_test[:,0] == test_brand][:,1:], dtype=np.float32)))
        test_labels.append(torch.from_numpy(np.array(y_test[X_test[:,0] == test_brand], dtype=np.float32)))
        
    else:
        # Train and set have the same distribution            
        for brand in brands:
            feature_datasets.append(torch.from_numpy(np.array(X_train[X_train[:,0] == brand][:,1:], dtype=np.float32)))
            labels.append(torch.from_numpy(np.array(y_train[X_train[:,0] == brand], dtype=np.float32)))
            feature_datasets_test.append(torch.from_numpy(np.array(X_test[X_test[:,0] == brand][:,1:], dtype=np.float32)))
            test_labels.append(torch.from_numpy(np.array(y_test[X_test[:,0] == brand], dtype=np.float32)))
        
    return feature_datasets, labels, feature_datasets_test, test_labels

def load_uber_lyft(n_participants=3, s=30, reduced=False):
    """     
	    Method to load the Uber_lyft dataset.
        Args:
            n_participants (int): number of data subsets to generate
            s (int): number of data samples for each participant (equal)
            reduced (bool): whether to use a reduced csv file for faster loading
        Returns:
            feature_datasets, labels, feature_datasets_test, test_labels: each a list containing the loaded dataset
    """    
    df = pd.read_csv('.data/Uber_lyft/rideshare_kaggle{}.csv'.format('_reduced' if reduced else ''))


    
    # Remove empty cell rows
    df = df[df['price'].isnull() == False]
    
    # Feature selection
    df = df[['price', 'distance', 'surge_multiplier', 'day', 'month', 'windBearing', 'cloudCover', 'name']]
    df = pd.get_dummies(df,columns=['name'], drop_first=True)
    df = df.drop(['name_Lux', 'name_Lux Black', 'name_Lux Black XL', 'name_Lyft', 'name_Lyft XL'], axis=1)

    # Shuffle and train test split
    data = df.copy()
    data = shuffle(data)
    X = data.drop(['price'], axis=1).values
    Y = data['price'].values
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1234)
    
    test_s = int(s * 0.2)
    feature_datasets, feature_datasets_test, labels, test_labels = [], [], [], []
    
    for i in range(n_participants):
        start_idx = i * s
        end_idx = (i + 1) * s
        test_start_idx = i * test_s
        test_end_idx = (i + 1) * test_s
        feature_datasets.append(torch.from_numpy(np.array(X_train[start_idx:end_idx], dtype=np.float32)))
        labels.append(torch.from_numpy(np.array(y_train[start_idx:end_idx], dtype=np.float32).reshape(-1, 1)))
        feature_datasets_test.append(torch.from_numpy(np.array(X_test[test_start_idx:test_end_idx], dtype=np.float32)))
        test_labels.append(torch.from_numpy(np.array(y_test[test_start_idx:test_end_idx], dtype=np.float32).reshape(-1, 1)))
    
    return feature_datasets, labels, feature_datasets_test, test_labels

def load_credit_card(n_participants=3, s=30, train_test_diff_distr=False):
    """     
	    Method to load the credit_card dataset.
        Args:
            n_participants (int): number of data subsets to generate
            s (int): number of data samples for each participant (equal)
            train_test_diff_distr (bool): whether to generate a test set that has a different distribution from the train set
        Returns:
            feature_datasets, labels, feature_datasets_test, test_labels: each a list containing the loaded dataset
    """
    test_s = int(s * 0.2)
    
    data = pd.read_csv(".data/Credit_card/creditcard.csv")
    
    # Use high amounts as hold out set
    hold_out_idx = list(data[data['Amount'] > 1000].index)
    hold_out = data.iloc[hold_out_idx[:test_s*n_participants]]
    data = data.drop(hold_out_idx)
    
    # Drop redundant features
    data = data.drop(['Class', 'Time'], axis = 1)
    hold_out = hold_out.drop(['Class', 'Time'], axis = 1)
    
    data = shuffle(data)
    X = data.iloc[:, data.columns != 'Amount']
    y = data.iloc[:, data.columns == 'Amount']

    # Feature selection
    cols = ['V1', 'V2', 'V5', 'V7', 'V10', 'V20', 'V21', 'V23']
    
    X_train, X_test, y_train, y_test = train_test_split(X[cols], y, test_size=0.2, random_state=1234)
    if train_test_diff_distr:
        X_test = hold_out.iloc[:, hold_out.columns != 'Amount'][cols]
        y_test = hold_out.iloc[:, hold_out.columns == 'Amount']
    
    feature_datasets, feature_datasets_test, labels, test_labels = [], [], [], []
    
    for i in range(n_participants):
        start_idx = i * s
        end_idx = (i + 1) * s
        test_start_idx = i * test_s
        test_end_idx = (i + 1) * test_s
        feature_datasets.append(torch.from_numpy(np.array(X_train[start_idx:end_idx], dtype=np.float32)))
        labels.append(torch.from_numpy(np.array(y_train[start_idx:end_idx], dtype=np.float32).reshape(-1, 1)))
        feature_datasets_test.append(torch.from_numpy(np.array(X_test[test_start_idx:test_end_idx], dtype=np.float32)))
        test_labels.append(torch.from_numpy(np.array(y_test[test_start_idx:test_end_idx], dtype=np.float32).reshape(-1, 1)))
    
    return feature_datasets, labels, feature_datasets_test, test_labels

def load_hotel_reviews(n_participants=3, s=30):
    """     
	    Method to load the hotel_reviews dataset.
        Args:
            n_participants (int): number of data subsets to generate
            s (int): number of data samples for each participant (equal)
        Returns:
            feature_datasets, labels, feature_datasets_test, test_labels: each a list containing the loaded dataset
    """
    
    # Note that here we load the extracted features. Code for extracting the features can be found in the notebooks folder.
    loaded = np.load('.data/TripAdvisor_hotel_reviews/extracted_features.npz')
    X = loaded['X']
    y = loaded['y']
    
    # Shuffle
    p = np.random.permutation(X.shape[0])
    X, y = X[p], y[p]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    
    test_s = int(s * 0.2)
    feature_datasets, feature_datasets_test, labels, test_labels = [], [], [], []
    
    for i in range(n_participants):
        start_idx = i * s
        end_idx = (i + 1) * s
        test_start_idx = i * test_s
        test_end_idx = (i + 1) * test_s
        feature_datasets.append(torch.from_numpy(np.array(X_train[start_idx:end_idx], dtype=np.float32)))
        labels.append(torch.from_numpy(np.array(y_train[start_idx:end_idx], dtype=np.float32).reshape(-1, 1)))
        feature_datasets_test.append(torch.from_numpy(np.array(X_test[test_start_idx:test_end_idx], dtype=np.float32)))
        test_labels.append(torch.from_numpy(np.array(y_test[test_start_idx:test_end_idx], dtype=np.float32).reshape(-1, 1)))
    
    return feature_datasets, labels, feature_datasets_test, test_labels


from .main_utils import get_synthetic_datasets, generate_linear_labels, friedman_function, hartmann_function, scale_normal
# from .volume import replicate

def get_datasets(dataset, M, train_sizes, D=5, test_sizes=[], ranges=[[0,1/3], [1/3, 2/3], [2/3, 1]]):
    '''
    M: number of players, i.e., number of datasets to return
    D: dimension of the feature space

    train_sizes: the size of each dataset
    test_sizes: the size of each test_dataset

    '''
    if len(test_sizes) == 0: test_sizes =  [200] * M


    if dataset in ['linear', 'friedman', 'hartmann']:

        feature_datasets = get_synthetic_datasets(n_participants=M, sizes=train_sizes, d=D, ranges=ranges)
        feature_datasets_test = get_synthetic_datasets(n_participants=M, sizes=test_sizes, d=D, ranges=ranges)

        if dataset == 'linear':
            labels, true_weights, true_bias = generate_linear_labels(feature_datasets, d=D)
            test_labels, _, _ = generate_linear_labels(feature_datasets_test, d=D, weights=true_weights, bias=true_bias)
        elif dataset == 'friedman':
            friedman_labels, friedman_noisy_labels, friedman_test_labels = [], [], []
            assert D >= 5 
            if D >= 5:
                for X in feature_datasets:
                    friedman_y = friedman_function(X)
                    friedman_labels.append(friedman_y)

                    friedman_y_noisy = friedman_y + torch.randn(friedman_y.shape) * 0.05
                    friedman_noisy_labels.append(friedman_y_noisy)

                for X in feature_datasets_test:
                    friedman_test_labels.append(friedman_function(X))
            labels, test_labels = friedman_noisy_labels, friedman_test_labels
        elif dataset == 'hartmann':
            hartmann_labels, hartmann_noisy_labels, hartmann_test_labels = [], [], []
            D = 4
            assert D in (3, 4, 6)
            if D in (3, 4, 6):
                for X in feature_datasets:
                    hartmann_y = hartmann_function(X)
                    hartmann_labels.append(hartmann_y)

                    hartmann_y_noisy = hartmann_y + torch.randn(hartmann_y.shape) * 0.0005
                    hartmann_noisy_labels.append(hartmann_y_noisy)

                for X in feature_datasets_test:
                    hartmann_test_labels.append(hartmann_function(X))
            labels, test_labels = hartmann_noisy_labels, hartmann_test_labels


    elif dataset == 'used_car':
        D = 5
        assert D == 5
        s = train_sizes[0]
        # train_sizes = []
        feature_datasets, labels, feature_datasets_test, test_labels = load_used_car(n_participants=M, s=s, train_test_diff_distr=False)
    elif dataset == 'uber_lyft':
        D = 12
        assert D == 12
        s = train_sizes[0]
        feature_datasets, labels, feature_datasets_test, test_labels = load_uber_lyft(n_participants=M, s=s, reduced=True)
    elif dataset == 'credit_card':
        D = 8
        assert D == 8
        s = train_sizes[0]

        feature_datasets, labels, feature_datasets_test, test_labels = load_credit_card(n_participants=M, s=50, train_test_diff_distr=False)
    elif dataset == 'hotel_reviews':
        D = 8
        assert D == 8
        feature_datasets, labels, feature_datasets_test, test_labels = load_hotel_reviews(n_participants=M, s=30)
    else:
        raise NotImplementedError('Dataset/function not implemented.')

    # if rep:
    #     feature_datasets_ = copy.deepcopy(feature_datasets)
    #     labels_ = copy.deepcopy(labels)
        
    #     for i in range(len(feature_datasets)):
    #         if rep_factors[i] == 1:
    #             continue
    #         to_replicate = torch.cat((feature_datasets[i], labels[i]), axis=1)
    #         replicated = replicate(to_replicate, c=rep_factors[i])
    #         feature_datasets_[i] = replicated[:,:-1]
    #         labels_[i] = replicated[:, -1:]
        
    #     feature_datasets, labels = feature_datasets_, labels_

    # if superset:
    #     # Create dataset such that party i is superset of party i-1
    #     feature_datasets_ = copy.deepcopy(feature_datasets)
    #     labels_ = copy.deepcopy(labels)
        
    #     for i in range(1, len(feature_datasets)):
    #         feature_datasets_[i] = torch.cat((feature_datasets[i], feature_datasets_[i-1]), axis=0)
    #         labels_[i] = torch.cat((labels[i], labels_[i-1]), axis=0)

    #     feature_datasets, labels = feature_datasets_, labels_

    # Standardize features to standard normal
    feature_datasets, feature_datasets_test = scale_normal(feature_datasets, feature_datasets_test)
    labels, test_labels = scale_normal(labels, test_labels)

    return feature_datasets, feature_datasets_test, labels, test_labels, D