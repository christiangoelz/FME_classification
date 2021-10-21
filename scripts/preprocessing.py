#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Chrisitan Goelz <goelz@sportmed.upb.de>
# Description: Preprocessing Functions EEG using MNE and custom functions
# Dependencies: see .yml 
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths
from os import path
import copy as cp
import glob

import mne
from mne import pick_channels
from mne.io import read_raw_edf
from mne.channels import make_standard_montage
from mne.preprocessing import ICA
from autoreject import AutoReject

class PREPRO: 
    """
    Object can be used for preprocessing of EEG data
    includes: 
        - rereferencing 
        - downsampling 
        - filtering 
        - ica occular correction 
        - autorejection of bad epochs 
        - user defined rejection 
        
    
    Parameters
        ----------
        participant: str
            string of participant name. Should match name of external folder to 
            match external signal for event calculation
        file : str
            path of file to preprocess
        trial_length : int, optional
            length of each trial contained in the eeg file. The default is None.
        nr_trials: list, optional 
            list of trials to match with external files, trials should be in consecutive order!
        eog_channels : list, optional 
            indices of eog channel recorded
        misc_channels : list, optional 
            indices of miscellaneous channel recorded 
        eeg_channels : list, optional 
            indices of miscellaneous channel recorded 
        stim_channels : list, optional 
            indices of trigger channel recorded     
        montage: str, optional 
            name of montage to read in, see MNE Documentation for possible 
            choices 
        ext_file_folder : str, optional 
            path of folder with external files to fit events with  
        event_dict : dict
            dictionary with event types of form {event_name: event_id}
        t_epoch: list of int
            cutting points of epochs     
            first element corresponds to minimum 
            second element corresponds to maximum
        sr_new : int, optional 
            sampling rate for resampling
        ref_new : list of strings
            list of channel names for rereferencing, default emtpy --> average ref.
        filter_freqs : list of int
            first element corresponds to lowfrequency cutoff 
            second element corresponds to highfrequency cutoff
            DESCRIPTION. The default is None
        ICA: bool
            If set to True ICA based occular correction will be applied
        Autoreject: bool 
            If set to True autoreject willbe applied based on autoreject 
            toolbox
        bads: indicesarray of int or bool
            Set epochs to remove by specifying indices to remove or a boolean
            mask to apply (where True values get removed). Events are correspondingly modified.
            
    Attributes
        ----------
        raw: class 
            MNE raw object (without data loded)
        epochs: class
            MNE data object containing epoched cleaned data 
        bads: array, shape(nbads,)
            contains number of excluded epochs
        ica: class
            MNE ICA object of MNE 
            
    Methods
        ----------    
        run(self) 
            run preprocessing
     """
    def __init__(self, 
                 participant,
                 file,
                 trial_length = None,
                 trials = None,
                 t_epoch = None,
                 eog_channels = [], 
                 misc_channels = [],
                 eeg_channels = [],
                 stim_channel = [],
                 montage = None,
                 marker_detection = None, 
                 event_detection = True, 
                 ext_file_folder = None, 
                 event_dict = None, 
                 sr_new = None, 
                 ref_new = None, 
                 filter_freqs = None, 
                 ICA = True,
                 Autoreject = True,
                 bads = None): 
        
        self.participant = participant
        self.file = file
        self.bads = bads
        self.info = {}  
        self.info['trial_length'] = trial_length
        self.info['trials'] = trials
        self.info['channel_info'] = {'EEG': eeg_channels,
                                     'EOG': eog_channels,
                                     'Misc': misc_channels,
                                     'Stim': stim_channel}
        self.info['event_detection'] = event_detection
        self.info['marker_detection'] = marker_detection            
        self.info['ext_file_folder'] = ext_file_folder
        self.info['event_dict'] = event_dict
        self.info['ICA'] = ICA
        self.info['Autoreject'] = Autoreject
        self.sr_new = sr_new
        self.filter_freqs = filter_freqs
        self.t_epoch = t_epoch
        self.montage = montage
        
    def run(self) :
        
        eog = self.info['channel_info']['EOG']
        misc = self.info['channel_info']['Misc']
        stim = self.info['channel_info']['Stim']
        
        try:
            ext_files = glob.glob(self.info['ext_file_folder'] + '/'+ self.participant + '/*axis0.dat')
        except:
            pass
                
        tmin = self.t_epoch[0]
        tmax = self.t_epoch[1]
        

        raw = read_raw_edf(self.file,  eog = eog, misc = misc)
        self.raw = cp.deepcopy(raw)
        raw.load_data()
        
        # marker detection (one marker continous trial)
        if self.info['marker_detection'] == True :
            starts = (find_trialstart(raw, stim_channel = raw.ch_names[stim[0]]))/2048 #convert to seconds
            try:
                starts[1] = starts[0] + 30
            except: 
                starts = np.r_[starts,(starts[0] + 30)]
            # events = np.zeros((len(starts),3))
            # events[:,0] = starts 
            # events[:,2] = list(self.info['event_dict'].values())
            # events = events.astype(np.int)
            eo = cp.deepcopy(raw).crop(tmin = starts[0],tmax = starts[1])
            ec = cp.deepcopy(raw).crop(tmin = starts[1],tmax = starts[1] + 30)
            events_eo = mne.make_fixed_length_events(eo, duration = 3, overlap = 0, id = 1)
            events_ec = mne.make_fixed_length_events(ec, duration = 3, overlap = 0, id = 2)
            events = np.concatenate([events_eo, events_ec]) # fit to new samplich rate
            events[:,0] = np.round(events[:,0]/2048 * 200)
            
            
        # event detection (one marker regular events)
        if self.info['event_detection'] ==  True:
            starts = find_trialstart(raw, stim_channel = raw.ch_names[stim[0]],
                                     new_samplin_rate = self.sr_new) 
        
            events = force_events(ext_files,
                                  self.info['event_dict'], 
                                  self.sr_new, 
                                  self.info['trial_length'], 
                                  self.info['trials'], 
                                  starts)
        
        
        if self.info['ICA']== True :
            ica = ICA(method = 'fastica')
        
        if self.info['Autoreject'] == True:
            ar = AutoReject()
        
        ## EEG preprocessing options will applied if parameters are set in object    
        
        #read montage 
        try:
            montage = make_standard_montage(self.montage)
            raw.set_montage(montage)
        except:
            pass
       
        #resampling
        try:
            raw.resample(sfreq = self.sr_new)
        except: 
            pass
        
        #rereferencing
        try:
            raw, _ = mne.set_eeg_reference(raw,ref_channels = ['EXG5','EXG6'])
        except: 
            pass
        
        #filter    
        try:
            low = self.filter_freqs[0]
            high = self.filter_freqs[1]
            raw.filter(low, high, fir_design='firwin')
        except:
            pass
        
        # occular correction
        try:
            ica.fit(raw)
            ica.exclude = []
            eog_indices, eog_scores = ica.find_bads_eog(raw)
            ica.exclude = eog_indices
            ica.apply(raw) 
            self.ica = ica
        except: 
            pass
        
        picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')
        
        event_id = self.info['event_dict']
        epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                            baseline=None, preload=True, picks=picks)
        
        #epoch rejection
        try: 
            epochs = epochs.drop(indices= self.bads)
        except:
            pass
        
        try:
            epochs, self.autoreject_log = ar.fit_transform(epochs, return_log = True)
        except:
            pass
        
        bads = np.asarray([l == ['USER'] or l == ['AUTOREJECT'] for l in epochs.drop_log])
        self.bads = np.where(bads == True)                 
        self.epochs = epochs
        return(self)
################### Custom Functions ################################
 
def find_trialstart(raw, stim_channel, threshold = 0.01, 
                    distance=None, prominence=None, width=None, wlen=None, rel_height=0.5, 
                    plateau_size=None, new_samplin_rate = None):
    """
    Function to find peaks in Stimulus channels. Based on scipy.signal.find_peaks
    Returns first detectet peak of peaksignal if it is plateau 

    Parameters
    ----------
    raw : class
          mne raw object
    stim_channel : str
        Required physical channel number in mne raw which represents 
        the stimulus signal. If not set here, it will be 
        looked up in mne raw description. The default is None.
    threshold : number | ndarray| sequence, optional
        Required threshold of peaks, the vertical distance to its neighbouring 
        samples. Either a number, None, an array matching x or a 2-element se-
        quence of the former. The first element is always interpreted as the 
        minimal and the second, if supplied, as the maximal required threshold. 
    distance :number, optional
        Required minimal horizontal distance (>= 1) in samples between neigh-
        bouring peaks. Smaller peaks are removed first until the condition is 
        fulfilled for all remaining peaks.
    prominence : number | ndarray | sequence, optional
        Required prominence of peaks. Either a number, None, an array matching 
        x or a 2-element sequence of the former. The first element is always 
        interpreted as the minimal and the second, if supplied, as the maximal 
        required prominence.
    width : number | ndarray | sequence, optional
        DESCRIPTION. The default is None.
    wlen : number | ndarray | sequence, optional
        DESCRIPTION. The default is None.
    rel_height :float, optional
        DESCRIPTION. The default is 0.5.
    plateau_size : number or ndarray or sequence, optional
        Required size of the flat top of peaks in samples. Either a number, 
        None, an array matching x or a 2-element sequence of the former. 
        The first element is always interpreted as the minimal and the second, 
        if supplied as the maximal required plateau size.
    new_samplin_rate : float, optional
        Used if peaks should be fitted to new sampling rate

    Raises
    ------
    ValueError
        raised if no channels are present  

    Returns
    -------
    ndarray: 
        indices of peaks in x that satisfy all given conditions, fitted to 
        new samplingh rate if provided

    """
    picks = pick_channels(raw.info['ch_names'], include = [stim_channel])
    
    if len(picks) == 0 :
        raise ValueError('No Stim Channel present in dataset! Please specify in class')
    
    data, _ = raw[picks]
    data = np.ravel(data)

    peaks, _ = find_peaks(data, threshold, distance, 
                       prominence, width, wlen, rel_height, 
                       plateau_size)
    widths = peak_widths(data, peaks, rel_height=1/raw.info['sfreq'])
    widths = widths[0]/2
    peaks = peaks - widths.astype(np.int)
    
    if new_samplin_rate != None:
        peaks = peaks/raw.info['sfreq'] * new_samplin_rate
    
    peaks = peaks.astype(np.int)
        
    return(peaks)


def force_events(ext_files, event_dict, sr_new, trial_length, trials = None, startpoints = 0):
    """
    Calaculation of events based on time stamps of external recorded files
    such as force signal files. Takes time stamps and 
    
    Parameters
    ----------
    ext_files : list
        containing path information of all files which should be accounted for
        as string
    event_dict : dict
        dictionary with event types of form {event_name: event_id}
    trial_length: int 
        length of trial
    trials: list, otional 
        list of which trials to consider
    startpoints : ndarray, optional
        Starting point from which events will be calculated. The default is 0.

    Returns
    -------
    events: ndarray, shape = (n_events, 3)
        All events that were found. The first column contains the event time 
        in samples and the third column contains the event id
    """

    nr_conditions = int(len(event_dict))
    if trials is not None: 
        nr_trials = len(trials)
        start = min(trials) - 1
        end = max(trials)
    else:
        nr_trials = int(len(ext_files)/nr_conditions)
        
    events = np.zeros((nr_trials*len(event_dict),3))
    ev = []
    ff = []
    
    for f in ext_files:
        ff.append(path.getmtime(f))
    
    times = pd.Series(ff, name='DateValue').sort_values()
    if trials is not None:
        times = times[start:end]
        
    diff_t = times.diff() 
    diff_t.iloc[0] = trial_length
    diff_t = diff_t.values.reshape(nr_conditions,nr_trials).T
    
    diff_t[0,:] = trial_length
    trial_end = np.cumsum(diff_t, axis = 0)
    trial_start = (trial_end - trial_length)* sr_new
   
    for i,start in enumerate(startpoints):
        ev.append(trial_start[:,i] + start)
        
    events[:,0] = np.ravel(np.asarray(ev))
    l = nr_trials*[list(event_dict.values())]
    l = np.ravel(l)
    l = np.sort(l)
    events[:,2] =  l
    return(events.astype(np.int))




    