#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis of behavioural data - isometric force control
Classification of Fine Motor Expertise

Authors
Christian Gölz
Sportmedizinisches Institut
Universität Paderborn
goelz@sportmed.upb.de

Roman Gaidai
Sportmedizinisches Institut
Universität Paderborn
gaidai@sportmed.upb.de
"""

#%% import 
import glob
import pickle
import numpy as np
from scipy import signal
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path 
import tsfel

#%% define input and output folders and get files
current_dir = Path.cwd()
results_path = glob.glob(str(current_dir.parent / 'results_used')) # where to put te reults 
force_path = glob.glob(str(current_dir.parent.parent / 'data' / 'BHS' / 'Verhalten_Motorik_Pre_Post' / 'Pre')) #location of force files 
files = glob.glob(force_path[0] + '/e*') + glob.glob(force_path[0] + '/na*') + glob.glob(force_path[0] + '/nm*')

#%% set fixed variables
n_trials = 160
fs = 120  # sampling rate
length = 120 * 3 # 3sec trial length
b, a = signal.butter(4, 30/(fs/2), 'low')
diff_signal = {}
part_all = []
mean_all_trial = []
exclude = []
df_ml = []
cfg_file = tsfel.get_features_by_domain()

#%% analyse each participant
for parti in files:
    diff = []
    label = []
    t = []
    ex = []
    part_all.append(parti[-4:])

    for trial in range(1,n_trials+1):
        # Import and filtering
        this_part_files = glob.glob(parti + '/*r' + str(trial) + 'axis0.dat')
        data = pd.read_csv(this_part_files[0], skiprows = 2)
        raw = data['Pinch 1'].values[120:-120]
        target = data['MVC 1'].values[120:-120]
        filt = signal.filtfilt(b, a, raw)
        d = target - filt # deviation from target
        diff.append(d)

        t.append(trial)
        if trial <= 40: 
            label.append(['steady right'])
        elif trial > 40 and trial <= 80:
            label.append(['sine right'])
        elif trial > 80 and trial <= 120 :
            label.append(['steady left'])
        else: 
            label.append(['sine left'])

    diff = np.asarray(diff)
    label = np.ravel(label)

    df = tsfel.time_series_features_extractor(cfg_file, diff, fs=fs) 
    df['dev'] = np.mean(abs(diff), axis = 1)
    df['var'] = np.std(abs(diff), axis = 1)
    df['trial'] = t
    df['label'] = label
    df['part'] = parti[-4:]
    
    # outlier detection and documentation
    # Z-score outlier detection
    zAll = df.groupby(['label'], sort = False).dev.apply(stats.zscore)
    zAll = np.concatenate(zAll.to_list())
    ex = np.where(np.abs(zAll > 3))
    df = df.drop(ex[0])
    exclude.append(np.array(ex[0]))

    this_result = df.groupby(['label'], sort = False).mean()
    this_result['int var'] = df.groupby(['label'])['dev'].std()
    this_result.drop(['trial'], axis = 1, inplace = True)
    this_result['part'] = parti[-4:]
    mean_all_trial.append(this_result)

    # data for trial classification
    df_ml.append(df)


# save results
exclude = pd.DataFrame(data = exclude, index = part_all)
mean_all = pd.concat(mean_all_trial)
df_ml = pd.concat(df_ml, ignore_index=True)
exclude.to_csv('Exclude.csv')
mean_all.to_csv(results_path[0] + '/Force_results.csv')
df_ml.to_csv(results_path[0] + '/force_features_trials.csv')
