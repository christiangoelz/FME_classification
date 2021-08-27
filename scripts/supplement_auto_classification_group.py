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
import autosklearn.classification
from sklearn.metrics import classification_report
from sklearn import preprocessing 
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupShuffleSplit


def classify(data, out_folder):
    labels = {'all': 0,
        'steady_right': 1, 
        'sine_right': 2, 
        'steady_left':3, 
        'sine_left': 4}

    #Configure Cross Validation
    cv_outer = GroupShuffleSplit(n_splits=10, test_size = .2, random_state = 456)

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
        out = str(out_folder / k / 'automl') 
        i = 0
        for train,test in cv_outer.split(X,y,part):
            print(' iteration:' + str(i))
            X_train, X_test = X.iloc[train,:], X.iloc[test,:]
            y_train, y_test = y[train], y[test]
            automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task = 120,
                                                                    per_run_time_limit = 30)
            automl.fit(X_train.copy(), y_train.copy(), dataset_name = 'auto_sklearn')
            automl.refit(X_train.copy(), y_train.copy())
            predictions = automl.predict(X_test)
            y_pred.append(predictions)
            y_true.append(y_test)
            report.append([classification_report(y_test, predictions, output_dict = True)])
            confmat.append(confusion_matrix(y_test, predictions, normalize = 'true'))
            i += 1
        print(classification_report(y_test, predictions))
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

#import eeg and force features
dmd_file = str(Path.cwd().parent / 'results' / 'mean_modes.csv')
force_file = str(Path.cwd().parent / 'results' / 'force_features_trials.csv')

dmd = pd.read_csv(dmd_file, index_col= 0)
out_eeg = Path.cwd().parent / 'results' / 'group_classification' / 'eeg'
classify(dmd,out_eeg) 

force = pd.read_csv(force_file)
force = force[['trial','label', 'dev', 'var', 'part']]
out_force = Path.cwd().parent / 'results' / 'group_classification' / 'force'
classify(force, out_force)

force['key'] = [p + str(t) for p,t in zip(force.part, force.trial)]
dmd['key'] = [p + str(t) for p,t in zip(dmd.part, dmd.trial)]
out_both = Path.cwd().parent / 'results' / 'group_classification' / 'both'
both = pd.merge(force, dmd, how="inner", on=["key"], suffixes = ["_x", None])
both.drop(['key', 'part_x', 'trial_x', 'label_x'], axis = 1, inplace = True)
classify(both, out_both)
