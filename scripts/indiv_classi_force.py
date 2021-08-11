#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 12:09:33 2021

@author: christian
"""
import glob
from pathlib import Path 
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix

dmdmean_file = str(Path.cwd().parent / 'input' / 'preprocessed' / 'Pre' / 'mean_modes.csv')
force_file = str(Path.cwd().parent / 'results_used' / 'force_features_trials.csv')
dmd_mean = pd.read_csv(dmdmean_file, index_col = 0)
df = pd.read_excel(force_file)
vpn_to_select = dmd_mean.part.unique()
df = df[df.part.isin(vpn_to_select)] #select only participans with corresponding eeg file
le = LabelEncoder()
labels = ['steady right', 'sine right', 'steady left', 'sine left']
df = df.replace(labels,[1,2,3,4])
parts = pd.unique(df.part)
param_grid = {'svc__C': [0.1, 1, 10, 100, 1000, 10000],
              'svc__gamma': ['scale'],
              'svc__kernel': ['rbf', 'poly','linear']}
steps = [('scaler', StandardScaler()),('svc', SVC())]
pipeline = Pipeline(steps)  
results = {}
conf_mat = {}
for part in parts: 
    cv_inner = StratifiedShuffleSplit(n_splits=10, test_size = .2,  random_state= 123)
    cv_outer = StratifiedShuffleSplit(n_splits=10, test_size = .2,  random_state= 345)
    df_part = df[df.part == part]
    y = df_part.label.values
    X = df_part[['dev','var']].values   
    results_per_fold = []
    confmat_per_fold = []
    for train,test in cv_outer.split(X,y):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test] 
        grid = GridSearchCV(pipeline, param_grid = param_grid, cv = cv_inner, n_jobs = -1)
        grid.fit(X_train, y_train)
        grid_predictions = grid.predict(X_test)
        results_per_fold.append(classification_report(y_test, grid_predictions, output_dict = True))
        confmat_per_fold.append(confusion_matrix(y_test, grid_predictions, normalize = 'true'))
    results[part] = results_per_fold
    conf_mat[part] = confmat_per_fold

with open(str(Path.cwd().parent / 'results_used' / 'indiv_classification_tasks_cf_force.pkl'), 'wb') as handle:
    pickle.dump(conf_mat, handle, protocol=pickle.HIGHEST_PROTOCOL) 

with open(str(Path.cwd().parent / 'results_used' / 'indiv_classification_tasks_report_force.pkl'), 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL) 