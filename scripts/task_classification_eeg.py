#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 12:09:33 2021

@author: christian
"""
import glob
import pickle 
from pathlib import Path 
from dmd_fbcsp import DMD_FBCSP 
import seaborn as sns
import numpy as np 
f = []
current_dir = Path.cwd()
files = glob.glob(str(current_dir.parent / 'input' / 'preprocessed'/ 'Pre' / 'DMD'  / 'nonorm' / '*both.pkl'))
out = str(current_dir.parent / 'input' / 'preprocessed' / 'DMD' / 'task.pkl')

results = {}
for file in files:
   part = file.split('/')[-1][:4]
   with open(file, 'rb') as handle:
       dmd = pickle.load(handle)
   try:
       dmd.results = dmd.results[dmd.results.label < 5]
       alle = DMD_FBCSP(dmd).classify() 
       results[part] = alle
   except:
       f.append(part)

with open(str(Path.cwd().parent / 'results' / 'indiv_classification_tasks.pkl'), 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL) 

print(f)
