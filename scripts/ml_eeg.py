#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 10:40:25 2021

@author: christian
"""
from pathlib import Path
import pandas as pd
import numpy as np
import pickle 
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing 
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, KFold, permutation_test_score
from sklearn.pipeline import Pipeline

y_true = []
y_pred = []
report_force = []
confmat_force = []
svm_models_force = []
le_eeg = preprocessing.LabelEncoder()
dmd_file = str(Path.cwd().parent / 'input' / 'preprocessed' / 'Pre' / 'mean_modes.csv')
dmd = pd.read_csv(dmd_file, index_col=1)
dmd = dmd[dmd.label == 4]
part = le_eeg.fit_transform(dmd['part'])
y_eeg = np.array([1 if 'e' in lab
              else 0 for lab in dmd['part']])
X_eeg = dmd.drop(['part','label'], axis = 1)

#Configure Cross Validation and define parameter_grid
cv_inner = GroupShuffleSplit(n_splits=10, test_size = .2, random_state = 123)
cv_outer = GroupShuffleSplit(n_splits=10, test_size = .2, random_state = 456)
steps = [('svc', SVC())]
param_grid = { 'svc__C': [0.1, 1, 10, 100, 1000, 10000] ,
              'svc__gamma': ['scale'],
              'svc__kernel': ['rbf', 'poly','linear']}

pipeline = Pipeline(steps)
i = 0
for train,test in cv_outer.split(X_eeg,y_eeg,part):
    X_train, X_test = X_eeg.iloc[train,:], X_eeg.iloc[test,:]
    y_train, y_test = y_eeg[train], y_eeg[test]
    part_train = part[train]
    grid_sv = GridSearchCV(pipeline, param_grid = param_grid, cv = cv_inner, 
                            scoring = 'accuracy', n_jobs = -1, refit = True)
    grid_sv.fit(X_train, y_train, groups = part_train)
    grid_predictions = grid_sv.predict(X_test)
    y_pred.append(grid_sv.predict(X_test))
    y_true.append(y_test)
    report_force.append([classification_report(y_test, grid_predictions, output_dict = True)])
    confmat_force.append(confusion_matrix(y_test, grid_predictions, normalize = 'true'))
    svm_models_force.append(grid_sv)

    print(i)
    i += 1

with open(str(Path.cwd().parent / 'results_used' / 'group_classification'/ 'sine_left' / 'classification_report_eeg.pkl'),'wb') as handle:
    pickle.dump(report_force,handle,protocol=pickle.HIGHEST_PROTOCOL)

with open(str(Path.cwd().parent / 'results_used' / 'group_classification'/ 'sine_left' / 'svm_models_eeg.pkl'),'wb') as handle:
    pickle.dump(svm_models_force,handle,protocol=pickle.HIGHEST_PROTOCOL)

with open(str(Path.cwd().parent / 'results_used' / 'group_classification'/ 'sine_left' / 'confmat_eeg.pkl'),'wb') as handle:
    pickle.dump(confmat_force,handle,protocol=pickle.HIGHEST_PROTOCOL)

with open(str(Path.cwd().parent / 'results_used' / 'group_classification'/ 'sine_left' /  'classification_y_true_eeg.pkl'),'wb') as handle:
    pickle.dump(y_true,handle,protocol=pickle.HIGHEST_PROTOCOL)

with open(str(Path.cwd().parent / 'results_used' / 'group_classification'/ 'sine_left' /  'classification_y_pred_eeg_.pkl'),'wb') as handle:
    pickle.dump(y_pred,handle,protocol=pickle.HIGHEST_PROTOCOL)