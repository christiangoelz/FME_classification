#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 11:31:26 2021

@author: christian
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pathlib import Path
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from ml_group import classify

labels = {'all': 0,'steady_left':3}

pipe_lda = Pipeline([('lda', LinearDiscriminantAnalysis())])
pipe_lda_other_feature =  Pipeline([('scaler', StandardScaler()),('lv', VarianceThreshold()),('lda', LinearDiscriminantAnalysis())])
param_grid_lda = {'lda__solver': ['lsqr', 'eigen'],
                    'lda__shrinkage': ['auto']}

pipe_rf = Pipeline([('rf',  RandomForestClassifier())])
pipe_rf_other_feature =  Pipeline([('scaler', StandardScaler()),('lv', VarianceThreshold()),('rf',  RandomForestClassifier())])
param_grid_rf = {'rf__n_estimators': np.arange(10, 230, 30)}

pipe_svc_other_feature = Pipeline([('scaler', StandardScaler()),('lv', VarianceThreshold()), ('svc', SVC())])
param_grid_svc_plus = { 'svc__C': [0.1, 1, 10, 100, 1000, 10000] ,
                        'svc__gamma': ['scale'],
                        'svc__kernel': ['rbf', 'poly']}

pipe_svc_plus = Pipeline([('scaler', StandardScaler()),('lv', VarianceThreshold()),('skb', SelectKBest(f_classif)), ('svc', SVC())])
param_grid_svc_plus = { 'svc__C': [0.1, 1, 10, 100, 1000, 10000] ,
                        'svc__gamma': ['scale'],
                        'svc__kernel': ['rbf', 'poly'],
                        'skb__k': [10, 20, 30, 40, 50, 100]}

pipe_lda_plus = Pipeline([('scaler', StandardScaler()),('lv', VarianceThreshold()),('skb', SelectKBest(f_classif)), ('lda', LinearDiscriminantAnalysis())])
param_grid_lda_plus = {'lda__solver': ['lsqr', 'eigen'],
                        'lda__shrinkage': ['auto'],
                        'skb__k': [10, 20, 30, 40, 50, 100]}

pipes = [pipe_lda, pipe_rf]
param_grids = [param_grid_lda, param_grid_rf] 

pipes_other_features = [pipe_lda_plus, pipe_svc_plus]
param_grids_other_features = [param_grid_lda_plus, param_grid_svc_plus] 


# input directories 
dmd_file = str(Path.cwd().parent / 'results' / 'mean_modes.csv')
force_file = str(Path.cwd().parent / 'results' / 'force_features_trials.csv')
eeg_features_file = str(Path.cwd().parent / 'results' / 'eeg_other_features.csv')

#feature sets: 
dmd = pd.read_csv(dmd_file, index_col= 0) #F0
eeg_features = pd.read_csv(eeg_features_file, index_col= 0, header = [0,1]) #F1
eeg_features.columns = eeg_features.columns.droplevel(1)
eeg_features['trial'] = dmd.trial.values 

force = pd.read_csv(force_file, index_col= 0)
force_performance = force[['trial','label', 'dev', 'var', 'part']]#F2
force_other = force.drop(['dev', 'var'], axis = 1) #F3

#F1 + F3
force_performance['key'] = [p + str(t) for p,t in zip(force_performance.part, force_performance.trial)]
dmd['key'] = [p + str(t) for p,t in zip(dmd.part, dmd.trial)]
out_both = Path.cwd().parent / 'results' / 'group_classification' / 'both'
both = pd.merge(force_performance, dmd, how="inner", on=["key"], suffixes = ["_x", None])
both.drop(['key', 'part_x', 'trial_x', 'label_x'], axis = 1, inplace = True)
dmd.drop(['key'], axis = 1, inplace = True)

out_force_performance = Path.cwd().parent / 'results' / 'group_classification' / 'force' / 'performance'
out_force_other = Path.cwd().parent / 'results' / 'group_classification' / 'force' / 'other_features'
out_eeg_dmd = Path.cwd().parent / 'results' / 'group_classification' / 'eeg' / 'dmd'
out_eeg_other = Path.cwd().parent / 'results' / 'group_classification' / 'eeg' / 'other_features'
out_both = Path.cwd().parent / 'results' / 'group_classification' / 'both'

# for pipe, grid in zip(pipes, param_grids): 
#     classify(force_performance, out_force_performance , pipe, grid, labels)
#     classify(dmd, out_eeg_dmd , pipe, grid, labels)
#     classify(both, out_both , pipe, grid, labels)
# classify(both, out_both, labels=labels)    
    
for pipe, grid in zip(pipes_other_features, param_grids_other_features):
    classify(force_other, out_force_other, pipe, grid, labels) 
    classify(eeg_features, out_eeg_other, pipe, grid, labels)