#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 11:31:26 2021

@author: christian
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupShuffleSplit
from sklearn import preprocessing 
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from pathlib import Path
import pickle 

y_true = []
y_pred = []
report_force = []
confmat_force = []
svm_models_force = []
le = preprocessing.LabelEncoder()
force_file = str(Path.cwd().parent / 'results_used' / 'force_features_trials.csv')
df = pd.read_csv(force_file, index_col = 0) 
df = df[df.label == 'steady left']
part = le.fit_transform(df['part'])
y = np.array([1 if 'e' in lab 
              else 0 for lab in df['part']])
X = df[['dev','var']]
cv_inner = GroupShuffleSplit(n_splits=1, test_size = .2, random_state = 123)
cv_outer = GroupShuffleSplit(n_splits=1, test_size = .2, random_state = 456)
steps = [('scale', StandardScaler()), ('svc', SVC())]
param_grid = { 'svc__C': [0.1, 1, 10, 100, 1000, 10000] ,
              'svc__gamma': ['scale'],
              'svc__kernel': ['rbf', 'poly','linear']}
pipeline = Pipeline(steps)
i = 0 

for train,test in cv_outer.split(X,y,part):
    X_train, X_test = X.iloc[train,:], X.iloc[test,:]
    y_train, y_test = y[train], y[test]
    part_train = part[train]
    
    grid_sv = GridSearchCV(pipeline, param_grid = param_grid, cv = cv_inner, 
                            scoring = 'accuracy', n_jobs = -1)
    grid_sv.fit(X_train, y_train, groups = part_train)
    grid_predictions = grid_sv.predict(X_test)
    
    y_pred.append(grid_sv.predict(X_test))
    y_true.append(y_test)
    report_force.append([classification_report(y_test, grid_predictions, output_dict = True)])
    confmat_force.append(confusion_matrix(y_test, grid_predictions, normalize = 'true'))
    svm_models_force.append(grid_sv)

    print(i)
    i += 1

with open(str(Path.cwd().parent / 'results_used' / 'group_classification'/ 'steady_left' / 'classification_report_force.pkl'),'wb') as handle:
    pickle.dump(report_force,handle,protocol=pickle.HIGHEST_PROTOCOL)

with open(str(Path.cwd().parent / 'results_used' / 'group_classification'/ 'steady_left' / 'svm_models_force.pkl'),'wb') as handle:
    pickle.dump(svm_models_force,handle,protocol=pickle.HIGHEST_PROTOCOL)

with open(str(Path.cwd().parent / 'results_used' / 'group_classification'/ 'steady_left' / 'confmat_force.pkl'),'wb') as handle:
    pickle.dump(confmat_force,handle,protocol=pickle.HIGHEST_PROTOCOL)

with open(str(Path.cwd().parent / 'results_used' / 'group_classification'/ 'steady_left' /  'classification_y_true_force.pkl'),'wb') as handle:
    pickle.dump(y_true,handle,protocol=pickle.HIGHEST_PROTOCOL)

with open(str(Path.cwd().parent / 'results_used' / 'group_classification'/ 'steady_left' /  'classification_y_pred_force_.pkl'),'wb') as handle:
    pickle.dump(y_pred,handle,protocol=pickle.HIGHEST_PROTOCOL)