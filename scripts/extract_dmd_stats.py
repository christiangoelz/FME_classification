#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 19:16:18 2021

@author: christian
""" 
import glob
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

def extract_stats(files, freqs, fbands, groupby = ['label', 'trial']):
    mean_modes = []
    for file in files:  
        with open(file, 'rb') as handle:
            dmd = pickle.load(handle)
        part = file.split('/')[-1][:4]
        print('processing file:' + part) 
        epoch_file = (part + '_both.pkl')
        epochs = str(Path.cwd().parent / 'input' / 'preprocessed' / 'Pre'/ 'EEG' / epoch_file)
        
        
        with open(epochs, 'rb') as handle:
            epochs = pickle.load(handle)
        
        cols_sel = [col for col in dmd.results.columns if 'PSI' in col]
        mean_modes_p = [dmd.get_PSI(b, unit_length = False).abs().groupby(groupby).mean()[cols_sel] for b in fbands]
        df = pd.concat(mean_modes_p, axis = 1)
        cols_new = [c + '_' + b for b in freqs for c in cols_sel]
        df.columns = cols_new
        df['part'] = part
        df.reset_index(inplace = True)
        df = df[df.label < 5]
        
        if 'trial' in groupby:
            #correct trial numbers to merge later 
            trials = pd.DataFrame(epochs.epochs.drop_log)
            trials['nr'] = np.arange(1,161)
            if len(trials.columns) == 2:
                df.trial = trials[trials.iloc[:,0].isna()].nr.values
            else: 
                df.trial = df.trial + 1
        mean_modes.append(df)
    return(mean_modes)

def run(in_name, freqs, fbands, groupby):
    files = glob.glob(in_name)
    mean_modes = extract_stats(files, freqs, fbands, groupby)
    return(mean_modes)


in_name = str(Path.cwd().parent / 'input' / 'preprocessed' / 'Pre'/ 'DMD' / 'nonorm' / '*both.pkl')
freqs = ['theta','alpha','beta1','beta2']
fbands = [[4,8],[8,12],[12,16],[16,30]]
#mean per trial
mean_modes_trial = run(in_name, freqs, fbands, groupby = ['trial', 'label']) 
pd.concat(mean_modes_trial).to_csv(str(Path.cwd().parent / 'results'/ 'mean_modes.csv')) 

#mean per task
mean_modes_task = run(in_name, freqs, fbands, groupby = ['label']) 
pd.concat(mean_modes_task).to_csv(str(Path.cwd().parent / 'results'/ 'meanTask_modes.csv')) 
