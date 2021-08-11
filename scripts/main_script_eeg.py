#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis of electrophysiological data

Classification of Fine Motor Expertise
Master Thesis - Intelligenz und Bewegung (M.Sc.)

Christian Gölz
Sportmedizinisches Institut
Universität Paderborn
goelz@sportmed.upb.de
Source Code of Preprocessing & DMD Objects:
https://github.com/christiangoelz/Code-christiangoelz-Code-Task-classification-and-brain-network-characteristics-of-fine-motor-moveme

modified by:
Roman Gaidai
Sportmedizinisches Institut
Universität Paderborn
gaidai@sportmed.upb.de
"""

import glob
from preprocessing import PREPRO # Gölz - GitHub Repository
from dmdeeg import DMD # Gölz - GitHub Repository
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

current_dir = Path.cwd()
output_dir = str(current_dir.parent / 'input' / 'preprocessed' )
EEGfiles  = str(current_dir.parent.parent / 'data' / 'BHS' / 'EEG_Motorik_Pre_Post' / 'Pre') 
forcefiles = str(current_dir.parent.parent / 'data' / 'BHS' / 'Verhalten_Motorik_Pre_Post' / 'Pre')
eeg_files = (glob.glob(EEGfiles + '/em00*.edf') +
            glob.glob(EEGfiles + '/E000*.edf') +
            glob.glob(EEGfiles + '/NA000*.edf') +
            glob.glob(EEGfiles + '/na09*.edf') + 
            glob.glob(EEGfiles + '/nm000*.edf'))

# del(eeg_files[45])  # delete nm15 first marker fault
# del(eeg_files[2])   # delete ea03 broken file
# del(eeg_files[35])  # delete nm01 

misc =list(range(32,47)); del misc[2]
exclude = pd.read_csv('Exclude.csv', index_col = 0)
fault = []
for file in eeg_files: # choose participants to solve problems with markers manually
    try:
        participant = file[-18:-14].lower()
        if participant in exclude.index:
            bads = np.asarray(exclude.loc[participant,:].dropna()) # exclude bad trials based on behavioural data
        else: 
            bads = []

        ###### PREPROCESSING ######
        data = PREPRO(participant = participant,
                      file = file,
                      trial_length = 5,
                      t_epoch = [-4, 4],
                      eog_channels = [34],
                      misc_channels = misc,
                      stim_channel = [42],
                      montage = 'biosemi32',
                      event_detection = True,
                      ext_file_folder = forcefiles,
                      event_dict = {'Steady_right':1, 'Sinus_right':2, 
                                    'Steady_left':3, 'Sinus_left':4,},
                      sr_new = 200,
                      ref_new = ['EXG5','EXG6'],
                      filter_freqs = [4,30],
                      ICA = True,
                      Autoreject = True,
                      bads = bads)
        data.run()
        
         ###### TASK EPOCHS ######
        epochs_cp = data.epochs.copy() 
        X_rest = epochs_cp.crop(-4,-1).get_data()*(1e6)
        X_task = data.epochs.crop(1,4).get_data()*(1e6)
        channels = data.epochs.info['ch_names'][:32]
        labels = data.epochs.events[:,-1]
        labels_rest = labels + 4

        with open(output_dir + '/Pre/EEG/' + participant + '_both.pkl', 'wb') as handle:
               pickle.dump(data,handle, protocol=pickle.HIGHEST_PROTOCOL)
    
        X = np.concatenate([X_task, X_rest],0)
        labels = np.concatenate([labels,labels_rest])
        
        ###### DMD ######
        dmd = DMD(X, labels,
                  channels = channels,
                  dt = 1/200,
                  win_size = 100,
                  datascale = 'none',
                  overlap = 50,
                  stacking_factor = 4,
                  truncation = True,
                  ).DMD_win()
    
        with open(output_dir + '/Pre/DMD/' + participant + '_dmd_both.pkl', 'wb') as handle:
               pickle.dump(dmd,handle, protocol=pickle.HIGHEST_PROTOCOL)

    except: 
        fault.append(participant) 
        pass
