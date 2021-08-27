import mne 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from permute.core import two_sample
from statsmodels.stats.multitest import fdrcorrection

m = mne.channels.make_standard_montage('biosemi32') 
info = mne.create_info(
        ch_names=m.ch_names, sfreq=200., ch_types='eeg')
info.set_montage(m)

dmdmean_file = str(Path.cwd().parent / 'results'/ 'meanTask_modes.csv') 
df = pd.read_csv(dmdmean_file, index_col=0)
# df.drop(['trial'], axis = 1, inplace = True)
df['exnov']= np.array([1 if 'e' in lab else 0 for lab in df['part']])
df.reset_index(inplace = True)
df = df[df.label <= 4]
df.label = df.label.replace([1,2,3,4],['steady right','sine right','steady left','sine left'] )
freqs = ['theta', 'alpha', 'beta1', 'beta2']
p_all = pd.DataFrame()
t_all = pd.DataFrame()
df_tasks = df.groupby("label")

for task, df_t in df_tasks:
    for f in freqs:
        df_t_f_exp =  df_t[df_t.exnov == 1].iloc[:,df_t.columns.str.contains(f)]
        df_t_f_nov =  df_t[df_t.exnov == 0].iloc[:,df_t.columns.str.contains(f)]
        t = []
        p = []
        for i in range(32): 
            x = df_t_f_exp.iloc[:,i].values
            y = df_t_f_nov.iloc[:,i].values
            p_temp, t_temp = two_sample(x,y, reps = 10000, stat='t',alternative="two-sided", seed=4)
            p.append(p_temp), t.append(t_temp) 
        p_all[task + '_' + f] = p
        t_all[task + '_' + f] = t

_, pcorr = fdrcorrection(np.ravel(p_all.values))
p_corr = pd.DataFrame(data = np.reshape(pcorr, (32,16)), columns=p_all.columns)

p_corr.to_csv(str(Path.cwd().parent / 'results'/ 'mean_modes_perm_p.csv'))
t_all.to_csv(str(Path.cwd().parent / 'results'/ 'mean_modes_perm_t.csv'))
