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
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold

def classify(data, out_folder, pipe = [], param_grid = [], labels = {'all': 0,'steady_right': 1, 'sine_right': 2, 'steady_left':3, 'sine_left': 4}):

    #Configure Cross Validation
    cv_inner = GroupShuffleSplit(n_splits=10, test_size = .2, random_state = 123)
    cv_outer = GroupShuffleSplit(n_splits=10, test_size = .2, random_state = 456)

    #predfined pipelines:
    if pipe == []: 
        pipe = Pipeline([('svc', SVC())])
        param_grid = ({ 'svc__C': [0.1, 1, 10, 100, 1000, 10000],
                    'svc__gamma': ['scale'],
                    'svc__kernel': ['rbf', 'poly','linear']})

    #run all classifier on all data and single tasks :
    for k in labels:
        y_true = []
        y_pred = []
        report = []
        confmat = []
        models  = []
        
        if k != 'all': 
            data_sel = data[data.label == labels[k]]
        else: 
            data_sel = data
        
        part = preprocessing.LabelEncoder().fit_transform(data_sel['part'])
        y = np.array([1 if 'e' in lab
                else 0 for lab in data_sel['part']])
        X = data_sel.drop(['part','label','trial'], axis = 1)

        out = str(out_folder / k / pipe.steps[-1][0]) 
        i = 0
        for train,test in cv_outer.split(X,y,part):
            print(k + ': ' + pipe.steps[-1][0] + str(param_grid) + ' iteration:' + str(i))
            X_train, X_test = X.iloc[train,:], X.iloc[test,:]
            y_train, y_test = y[train], y[test]
            part_train = part[train]
            grid_sv = GridSearchCV(pipe, param_grid = param_grid, cv = cv_inner, 
                                    scoring = 'accuracy', n_jobs = -1, refit = True)
            grid_sv.fit(X_train, y_train, groups = part_train)
            grid_predictions = grid_sv.predict(X_test)
            y_pred.append(grid_sv.predict(X_test))
            y_true.append(y_test)
            report.append([classification_report(y_test, grid_predictions, output_dict = True)])
            confmat.append(confusion_matrix(y_test, grid_predictions, normalize = 'true'))
            models.append(grid_sv)
            i += 1

        with open(out + '/classification_report.pkl','wb') as handle:
            pickle.dump(report,handle,protocol=pickle.HIGHEST_PROTOCOL)

        with open(out + '/models.pkl','wb') as handle:
            pickle.dump(models,handle,protocol=pickle.HIGHEST_PROTOCOL)

        with open(out + '/confmat.pkl','wb') as handle:
            pickle.dump(confmat,handle,protocol=pickle.HIGHEST_PROTOCOL)

        with open(out + '/classification_y_true.pkl','wb') as handle:
            pickle.dump(y_true,handle,protocol=pickle.HIGHEST_PROTOCOL)

        with open(out + '/classification_y_pred.pkl','wb') as handle:
            pickle.dump(y_pred,handle,protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":
    #import eeg and force features
    dmd_file = str(Path.cwd().parent / 'results_used' / 'mean_modes.csv')
    force_file = str(Path.cwd().parent / 'results_used' / 'force_features_trials.csv')

    dmd = pd.read_csv(dmd_file, index_col= 0)
    out_eeg = Path.cwd().parent / 'results_used' / 'group_classification' / 'eeg' / 'dmd'
    classify(dmd,out_eeg) 

    force = pd.read_csv(force_file)
    force = force[['trial','label', 'dev', 'var', 'part']]
    out_force = Path.cwd().parent / 'results_used' / 'group_classification' / 'force' / 'performance'
    classify(force, out_force)