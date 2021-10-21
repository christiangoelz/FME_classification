
# Paper: Classification characteristics of fine motor experts based on electroencephalographic and force tracking data 

This repository contains the code to generate results, summary statistics, and raw figures for the manuscript "Classification characteristics of fine motor experts based on electroencephalographic and force tracking data"\
Code dependencies:

- Code was developed in Anaconda in the following environment: fme_classification.yml
- feature sets used are available under /results

Recomputation: 

Note: For recomputation all input paths should be adjusted 

1. Force preprocessing and feature extraction: 
    - force_analysis.py  

2. EEG preprocessing and feature extraction: 
    - eeg_analysis.py - preprocessing and dmd of EEG data
    - extract_dmd_stats.py - feature extraction (EEG features (main text))

3. Descriptive results and group statistics: 
    - results_group_characteristics.ipynb - group characteristics (NOV and FME group)
    - results_eeg_perm_test.ipynb / results_group_differences -  group differences in force control and EEG activity

4. Classification of Tasks: 
    - task_classification_eeg.py / task_classification_force - classification of tasks based on EEG / force control features
    - results_ml_tasks.ipynb - stats of classification of tasks based on EEG / force control features

5. Classification of Groups: 
    - group_classiication_force_eeg.py - classification of groups based on EEG / force control features
    - results_ml_groups.ipynb - stats of classification of tasks based on EEG / force control features

6. Feature Space Characteristics:
    - results_feature_space_characteristics.ipynb - UMAP visualzation of EEG feature space (feature space characteristics) 

### Supplement: 

#### S1 Group classification: additional classifiers and feature sets: 
- extract_eeg_otherfeatures.py. - feature extraction additional EEG features
- supplement_classification_group.py - additional classification pipelines (other classifier and feature sets)
- supplement_auto_classification_group.py - classification using automatic machine learning (Auto sklearn: https://automl.github.io/auto-sklearn/)
- supplement_ml_groups - results additional classification pipelines (other classifier and feature sets)

#### S2 Task classification: additional feature set:
- supplement_task_classification_force_otherfeatures.py - task classification based on additional force control features
- supplement_ml_tasks_other_features.ipynb - results additional additional force control features
