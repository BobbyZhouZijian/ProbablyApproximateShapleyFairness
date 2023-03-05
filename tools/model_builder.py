import os, sys, warnings, inspect, pickle
import numpy as np
import pandas as pd
from glob import glob
import _pickle as pkl
from collections import defaultdict
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from scipy.linalg import eigh
from scipy.stats import kendalltau, ttest_ind
from sklearn.cluster import KMeans

random_state = 2022


def return_model(mode, **kwargs):
    if mode == 'logistic':
        solver = kwargs.get('solver', 'liblinear')
        n_jobs = kwargs.get('n_jobs', -1)
        C = kwargs.get('C', 0.05)
        max_iter = kwargs.get('max_iter', 5000)
        model = LogisticRegression(solver=solver, n_jobs=n_jobs, C=C, max_iter=max_iter, random_state=random_state)
    elif mode == 'linear':
        n_jobs = kwargs.get('n_jobs', -1)
        model = LinearRegression(n_jobs=n_jobs)
    elif mode == 'ridge':
        alpha = kwargs.get('alpha', 1.0)
        model = Ridge(alpha=alpha, random_state=random_state)
    elif mode == 'Tree':
        model = DecisionTreeClassifier(random_state=random_state)
    elif mode == 'RandomForest':
        n_estimators = kwargs.get('n_estimators', 50)
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    elif mode == 'GB':
        n_estimators = kwargs.get('n_estimators', 50)
        model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=random_state)
    elif mode == 'AdaBoost':
        n_estimators = kwargs.get('n_estimators', 50)
        model = AdaBoostClassifier(n_estimators=n_estimators, random_state=random_state)
    elif mode == 'SVC':
        kernel = kwargs.get('kernel', 'rbf')
        C = kwargs.get('C', 1.0)  # 1.
        max_iter = kwargs.get('max_iter', 5000)
        model = SVC(kernel=kernel, max_iter=max_iter, C=C, random_state=random_state)
    elif mode == 'LinearSVC':
        C = kwargs.get('C', 0.05)  # 1.
        max_iter = kwargs.get('max_iter', 5000)
        model = LinearSVC(loss='hinge', max_iter=max_iter, C=C, random_state=random_state)
    elif mode == 'GP':
        model = GaussianProcessClassifier(random_state=random_state)
    elif mode == 'KNN':
        n_neighbors = kwargs.get('n_neighbors', 1)
        n_jobs = kwargs.get('n_jobs', -1)
        model = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=n_jobs)
    elif mode == 'NB':
        model = MultinomialNB()
    else:
        raise ValueError("Invalid mode!")
    return model
