#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 21:44:28 2020

@author: christiangolz
"""
import numpy as np
from random import sample, shuffle
import copy as cp
from csp_DMD import CSP
import warnings
import gc

warnings.filterwarnings("ignore")

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix


class DMD_FBCSP:
    """Object for EEG signal classification using Dynamic Mode Decomposition (DMD)
    and (Filter Bank) Common Spatial Patterns. DMD modes are prijected into CSP space 
    and classified with LDA classifier. nxn fold Cross Validation is used in classification
    in combination with Gridsearch for LDA parameter selection 
    
    Parameters
    ----------
    dmd, : object of DMD module, 
        contains dmd decomposed EEG data with information of labels windows and epochs 
    fbands: list, shape [[a,b],[c,d]], optional
        list of 2 element list with a,b,c,d are floats defining freqeuncy bands 
        lower and upper frquency, default: [[4,8],[8,12],[12,16],[16,30]]
    n_components: int, optional
        defines how many CSP components to keep as projector 
    folds: int, optional
        folds used in stratified crossvalidation, default is 10
    test_size: float, optional
        defines test daatset, default is 20% (0.2)
    use_cases_per_class, int, optional 
        defines how many cases per class should be used in case data labels are
        merged, default is 40
    classifier: str, 
        which classifier to use. LDA or RF (randomforrest), default is LDA
    select_labels: list, optional 
        contains label indices of labels which should be selected, default is [] 
    merge_labels: list, shape [[a,b],[c,d]] 
        list of 2 element list with a,b and c,d are integers defining which 
        labels to merge, default is [] 
        
    
    Attributes
    ----------
    dmd: obj of DMD module 
        dmd adjusted if labels were merged or selected 
    auc_score: list, shape(n_folds)
        classification ROCAUC score for each fold, multiclass is implemented as 
        macro average in one vs. one classification
    conf_mat: ist, shape(n_folds)
        classification conf mat score for each fold 
    fpr: list, shape(n_folds)
        false positive rate fore each fold
    tpr: list, shape(n_folds)
        true positive rate fore each fold
    metrics: list, shape(n_folds)
        metrics fore each fold
    importanes: list, shape(n_folds)
        feature importances in each fold (only when RF is selected)
        
    Methods
    ----------
    classify(self): 
        run classification with specified parameters
        returns: modified self instance of object 
    get_get_csp_patterns(self)
        calculates CSP patterns and returns patterns of n_components
    """
        
    def __init__(self, 
                 dmd,
                 fbands = [[4,8],[8,12],[12,16],[16,30]],
                 n_components = 2,
                 folds = 10,
                 test_size = 0.2,
                 use_cases_per_class = 40,
                 classifier = 'LDA',
                 select_labels = [],
                 merge_labels = []):
        self.dmd = cp.deepcopy(dmd)
        self.fbands = fbands
        self.n_components = n_components
        self.folds = folds 
        self.test_size = test_size
        self.use_cases_per_class = use_cases_per_class
        self.classifier = classifier
        self.select_labels = select_labels
        self.labels = cp.deepcopy(dmd.y)
        self.merge_labels = merge_labels
        y = self.dmd.results['label']
        
        # merge labels and create new labels 
        if len(merge_labels) > 0:
            for n,l in enumerate(merge_labels):
                y[(y == l[0]) | (y == l[1])] = n
        # select defined labels
        if len(select_labels) > 0:
            self.dmd.results = dmd.results[dmd.results.label.isin(select_labels)]
        
        self.n_cases = len(np.unique(y))
            
    def classify(self):
        '''
        classification based on LDA classifier with nxn fold cv 
        if labels are merged teh same portion per label are selected based on 
        number of cases per class defined in init 

        Returns
        -------
        self: modified instance of object 
        # '''
        # accuracy = [] 
        auc_score = [] 
        metrics_ = []
        importances_ = []
        fpr = []
        tpr = [] 
        conf_mat = [] 
        cv = StratifiedShuffleSplit(self.folds, test_size= self.test_size, random_state=0)
        
        if self.classifier == 'LDA':
            param_grid = {'LDA__solver': ['lsqr','eigen'], 
                              'LDA__shrinkage': ['auto']} 
                            #   'LDA__n_components' : np.unique(self.dmd.y)[:-1]}
            steps = [('standardscaler', StandardScaler()),
                  ('LDA',  LinearDiscriminantAnalysis())]
        
        elif self.classifier == 'RF':
            steps = [('standardscaler', StandardScaler()),('RF',  RandomForestClassifier(random_state=0))]
            param_grid = {'RF__n_estimators': np.arange(10, 100, 30)}
            
        if len(self.merge_labels) > 0:
            # calculate labels p
            samp_size = self.use_cases_per_class * self.n_cases
            train_size = int((samp_size/len(np.unique(self.labels)))*(1-self.test_size))
            test_size = int((samp_size/len(np.unique(self.labels)))*(self.test_size))
        
        for train,test in cv.split(self.dmd.X, self.labels):
            
            # in case of merging get the same number per condition in merged conditions 
            if len(self.merge_labels) > 0:
                train_picks = [] 
                test_picks = [] 
                for n in np.unique(self.labels):
                    train_entry = np.where(self.labels[train]==n)
                    train_pick = sample(list(train_entry[0]), 
                                        k = train_size)
                    
                    test_entry = np.where(self.labels[test]==n)
                    test_pick = sample(list(test_entry[0]), 
                                       k = test_size)
                    
                    train_picks.append(train_pick); test_picks.append(test_pick)
                train_picks = np.ravel(train_picks); test_picks = np.ravel(test_picks);
                train = train[train_picks]; test = test[test_picks]
                shuffle(train); shuffle(test)
                
            #select train and test data 
            dmd_train = self.dmd.select_trials(train)
            dmd_test = self.dmd.select_trials(test)
            
            #Feature generation (FBCSP) fit on train data and transform train and test 
            X_train = []
            X_test = []            
            for band in self.fbands:
                csp = CSP(n_components= self.n_components, reg=None, log=True, norm_trace=False)
                x_train  = dmd_train.get_PSI(fband = [band[0],band[1]])      
                csp.fit(x_train.abs())
                x_train, y_train = csp.transform(x_train.abs())
                X_train.append(x_train)
            
                x_test = dmd_test.get_PSI(fband = [band[0],band[1]])    
                x_test, y_test = csp.transform(x_test.abs())
                X_test.append(x_test)
                
            X_train = np.concatenate(X_train, axis = 1)
            X_test = np.concatenate(X_test, axis = 1)
            
            #LDA Classifier
            pipeline = Pipeline(steps)
            GS = GridSearchCV(pipeline, param_grid = param_grid, cv = cv)
            GS.fit(X_train, y_train)
            y_pred = GS.predict(X_test)
            metrics_.append(metrics.classification_report(y_test, y_pred, output_dict=True))
            conf_mat.append(confusion_matrix(y_test, y_pred, normalize = 'true'))
            if self.classifier == 'RF':
                importances_.append(GS.best_estimator_.named_steps["RF"].feature_importances_)

            #adjust labels in case they are not compatible to f1 metrics calculation 
            for a,b in enumerate(np.unique(y_test)):
                y_train[y_train == b] = a
                y_test[y_test == b] = a
                y_pred[y_pred == b] = a
            # f1.append(metrics.f1_score(y_test, y_pred, average = 'macro'))
            
            try:
                score = GS.fit(X_train, y_train).decision_function(X_test)
            except:
                score = GS.fit(X_train, y_train).predict_proba(X_test)
                if self.classifier == 'RF':
                    score = score[:,1]
            try:
                auc_score.append(metrics.roc_auc_score(y_test, score, average = 'macro'))
                fpr1, tpr1, _ = roc_curve(y_test, score)
                fpr.append(fpr1); tpr.append(tpr1)
            except:
                score = GS.fit(X_train, y_train).predict_proba(X_test)
                auc_score.append(metrics.roc_auc_score(y_test, score, average = 'macro', multi_class = 'ovo'))

            gc.collect() #release memory 
        self.auc_score = auc_score
        self.conf_mat = conf_mat
        self.metrics_ = metrics_
        self.tpr = tpr
        self.fpr = fpr
        if self.classifier == 'RF':
            self.importances_ = importances_
        return(self)
            
    def get_csp_patterns(self):
        '''
        calculates and returns csp patterns over whiole dataset 
        
        Returns
        -------
        csp_patterns 

        '''
        labels = list(np.unique(self.dmd.results['label']))
        csp_patterns = [] 
        csp_features = [] 
        for band in self.fbands:
            csp = CSP(n_components = self.n_components, reg=None, log=True, norm_trace=False)
            x  = self.dmd.get_PSI(fband = [band[0],band[1]], labels = labels)
            csp.fit(x.abs())
            csp_features.append(csp.transform(x.abs()))
            csp_patterns.append(csp.patterns_[:self.n_components,:])
        return(csp_patterns,csp_features)
