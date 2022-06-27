#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 12:09:33 2021

@author: christian
"""

from pathlib import Path 
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from joblib import Parallel, delayed

def load(dmdmean_file, force_file):
    dmd_mean = pd.read_csv(dmdmean_file, index_col = 0)
    df = pd.read_csv(force_file)
    vpn_to_select = dmd_mean.part.unique()
    df = df[df.part.isin(vpn_to_select)] #select only participans with corresponding eeg file
    le = LabelEncoder()
    labels = ['steady right', 'sine right', 'steady left', 'sine left']
    df = df.replace(labels,[1,2,3,4])
    return df

def cv_score(train_idx, test_idx, X, y,clf):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx] 
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    return(classification_report(y_test, predictions, output_dict = True))

def classify(df, group = 'n'):
    # load data and get X,y
    df = df[df.part.str.contains(group)]
    part_id = df.part.values
    y = df.label.values
    X = df[['dev','var']].values

    # define ML parameters
    param_grid = {'svc__C': [0.1, 1, 10, 100, 1000, 10000],
                'svc__gamma': ['scale'],
                'svc__kernel': ['rbf', 'poly','linear']}
    steps = [('scaler', StandardScaler()),('svc', SVC())]
    pipeline = Pipeline(steps)  
    cv = GroupShuffleSplit(n_splits=10, test_size = .2, random_state = 1234)
    clf = GridSearchCV(pipeline, param_grid = param_grid, n_jobs = -1)
    reports = Parallel(n_jobs=-1)(delayed(cv_score)(train_idx,test_idx,X,y,clf) for train_idx,test_idx in cv.split(X,y,part_id))

    with open(group + '_between_part_task_classification_report_force.pkl', 'wb') as f:
        pickle.dump(reports,f,protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    dmdmean_file = str(Path.cwd().parent / 'input' / 'preprocessed' / 'Pre' / 'mean_modes.csv')
    force_file = str(Path.cwd().parent / 'results' / 'force_features_trials.csv')
    force_file = load(dmdmean_file,force_file)
    for g in ['n','e']: 
        classify(force_file,g)
