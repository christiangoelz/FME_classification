import glob
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from mne_features.feature_extraction import extract_features

current_dir = Path.cwd()
files = glob.glob(str(current_dir.parent / 'input' / 'preprocessed' / 'Pre' / 'EEG' / '*both.pkl'))
out = str(current_dir.parent / 'results' / 'eeg_other_features.csv')
v = []
for file in files:  
    with open(file, 'rb') as handle:
        epochs = pickle.load(handle)
        data = epochs.epochs.get_data()
        selected_funcs = {'mean', 
                            'variance',
                            'samp_entropy',
                            'pow_freq_bands',
                            'zero_crossings',
                            'phase_lock_val'}
        funcs_params = {'pow_freq_bands__freq_bands' :np.array([4,8,12,16,30])}
        df = extract_features(data, 200, selected_funcs, funcs_params, return_as_df = True)
        df['label'] = epochs.epochs.events[:,-1]
        df['part'] =  file.split('/')[-1][:4]
        v.append(df)
        print('processing: ' + file.split('/')[-1][:4])
pd.concat(v).to_csv(out)
        
